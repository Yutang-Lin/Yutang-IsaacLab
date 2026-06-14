# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anchored FB-CPR-Aux algorithm (Global-through-Anchoring) for BFM-One.

Extends :class:`FBCprAux` so a single fixed latent ``z = [z_local | z_spatial]``
encodes a complete global-local goal, and the policy reaches it closed-loop
without latent replanning.

Per update, four things are independently relabeled (all off-policy):

  * the replay transition ``(s_t, a_t, s_{t+1})``;
  * the task goal ``s_h ~ p_goal`` (future / nearby / replay / composed);
  * the FB successor-query state ``s_+ ~ rho`` (INDEPENDENT of s_h — the spec
    warns that ``s_+ = s_h`` collapses to goal-conditioned value and loses the
    full FB successor representation);
  * the coordinate anchor ``A ~ p_A`` (mix of ``g_t``, ``g_h`` and random),
    shared across this sample's current/next/goal/query states.

Anchoring is implemented as a RELABEL of the ``anchored_pose`` obs key: the
replay stores the raw world SE(2) pose ``(root_xy, root_yaw)`` per transition,
and at update time we recompute ``anchored_pose = enc(A^-1 g)`` for the
current / next / goal / query states under a fresh anchor ``A``. B_spatial reads
``anchored_pose``; F / actor / critic also read it. The spatial block of ``z``
is set from ``B_spatial`` of the anchored task goal.

The implicit Q ``<F, z>`` and all FB Bellman machinery are unchanged — only the
obs content (anchored pose) and the goal/query relabeling differ.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .fb_cpr import FBCprAux


def _yaw_to_cos_sin(yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cos(yaw), torch.sin(yaw)


class AnchoredFBCprAux(FBCprAux):
    """FB-CPR-Aux with Global-through-Anchoring relabeling."""

    # ------------------------------------------------------------------ #
    # Split CPR: local discriminator on z_local + spatial discriminator on
    # z_spatial (both judge the SAME local body motion). The spatial channel
    # asks "is this local motion plausible for the commanded spatial goal?".
    # ------------------------------------------------------------------ #
    def _discriminator_opt_params(self):
        params = list(self.policy._discriminator.parameters())
        ds = getattr(self.policy, "_discriminator_spatial", None)
        if ds is not None:
            params += list(ds.parameters())
        return params

    def _cpr_reward(self, obs, z):
        p = self.policy
        r_local = p._discriminator.compute_reward(obs, p.z_local(z))
        ds = getattr(p, "_discriminator_spatial", None)
        if ds is None:
            return r_local
        lam = float(getattr(self.cfg, "spatial_cpr_coeff", 1.0))
        r_spatial = ds.compute_reward(obs, p.z_spatial(z))
        # Stash channel magnitudes for logging (merged into critic metrics).
        self._cpr_reward_logs = {
            "cpr_reward/local": r_local.mean().detach(),
            "cpr_reward/spatial": r_spatial.mean().detach(),
            "cpr_reward/spatial_weighted": (lam * r_spatial).mean().detach(),
        }
        return r_local + lam * r_spatial

    def backward_discriminator(self, expert_obs, expert_z, train_obs, train_z,
                               grad_penalty, expert_mask=None):
        p = self.policy
        # Local channel: reuse the base disc trainer, but conditioned on the
        # LOCAL z block only (the base operates on self.policy._discriminator,
        # whose z_dim is z_local_dim in the anchored policy).
        metrics, handle = super().backward_discriminator(
            expert_obs, p.z_local(expert_z), train_obs, p.z_local(train_z),
            grad_penalty, expert_mask=expert_mask,
        )
        # Prefix the local-channel metrics for clarity in logs.
        metrics = {f"disc_local/{k}": v for k, v in metrics.items()}

        ds = getattr(p, "_discriminator_spatial", None)
        if ds is not None:
            sp_metrics = self._backward_discriminator_spatial(
                expert_obs, p.z_spatial(expert_z), train_obs, p.z_spatial(train_z),
                expert_mask=expert_mask,
            )
            metrics.update(sp_metrics)
        return metrics, handle

    def backward_critic(self, *args, **kwargs):
        metrics, handle = super().backward_critic(*args, **kwargs)
        logs = getattr(self, "_cpr_reward_logs", None)
        if logs:
            metrics.update(logs)
            self._cpr_reward_logs = None
        return metrics, handle

    def _backward_discriminator_spatial(self, expert_obs, expert_zs, train_obs,
                                        train_zs, expert_mask=None):
        """Train the spatial discriminator D_spatial(local_obs, z_spatial).

        Standard non-saturating GAN loss (no WGAN-GP; the local disc carries
        the gradient-penalty regularisation). Same merged real+fake forward.
        """
        import torch.nn.functional as F
        ds = self.policy._discriminator_spatial
        if expert_mask is not None:
            if isinstance(expert_obs, dict):
                eo = {k: v[expert_mask] for k, v in expert_obs.items()}
            else:
                eo = expert_obs[expert_mask]
            ez = expert_zs[expert_mask]
        else:
            eo, ez = expert_obs, expert_zs
        n_real = ez.shape[0]
        if isinstance(eo, dict):
            merged = {k: torch.cat([eo[k], train_obs[k]], dim=0) for k in eo}
        else:
            merged = torch.cat([eo, train_obs], dim=0)
        merged_z = torch.cat([ez, train_zs], dim=0)
        logits = ds.compute_logits(merged, merged_z)
        exp_logits, unl_logits = logits[:n_real], logits[n_real:]
        loss = (-F.logsigmoid(exp_logits)).mean() + F.softplus(unl_logits).mean()
        # IMPORTANT: do NOT zero_grad here — the local channel (super) already
        # zero-grad'd the shared discriminator_optimizer and accumulated the
        # local disc's grads. We backward into the spatial disc's (disjoint)
        # params; the single step in step_discriminator then applies BOTH.
        loss.backward()
        # The spatial disc is not DDP-wrapped (like the local disc); reduce its
        # grads manually under DDP and block before the shared optimizer steps.
        if self.is_distributed:
            from ..utils import finish_async_reduce, reduce_gradients_async
            finish_async_reduce(reduce_gradients_async(ds))
        with torch.no_grad():
            out = {
                "disc_spatial/disc_loss": loss.detach(),
                "disc_spatial/expert_logit": exp_logits.mean().detach(),
                "disc_spatial/train_logit": unl_logits.mean().detach(),
                "disc_spatial/expert_acc": (exp_logits > 0).float().mean().detach(),
                "disc_spatial/train_acc": (unl_logits < 0).float().mean().detach(),
            }
        return out

    # ------------------------------------------------------------------ #
    # Anchored-pose encoding
    # ------------------------------------------------------------------ #
    def _anchor_cfg(self):
        # Static config — build once and cache (called twice per update).
        cached = getattr(self, "_anchor_cfg_cache", None)
        if cached is not None:
            return cached
        c = self.cfg
        cached = {
            "clamp": float(getattr(c, "anchor_pose_clamp", 10.0)),
            "alpha_gt": float(getattr(c, "anchor_alpha_gt", 0.34)),
            "beta_gh": float(getattr(c, "anchor_beta_gh", 0.33)),
            # remainder -> random anchor
            "rand_xy": float(getattr(c, "anchor_random_xy_range", 10.0)),
            "goal_future": float(getattr(c, "goal_future_ratio", 0.4)),
            "goal_nearby": float(getattr(c, "goal_nearby_ratio", 0.2)),
            "goal_replay": float(getattr(c, "goal_replay_ratio", 0.2)),
            "goal_composed": float(getattr(c, "goal_composed_ratio", 0.2)),
            "nearby_radius": float(getattr(c, "goal_nearby_radius", 2.0)),
            "key": str(getattr(c, "anchored_pose_key", "anchored_pose")),
        }
        self._anchor_cfg_cache = cached
        return cached

    @staticmethod
    def _encode_anchored_pose(
        g_xy: torch.Tensor, g_yaw: torch.Tensor,
        a_xy: torch.Tensor, a_yaw: torch.Tensor, clamp: float,
    ) -> torch.Tensor:
        """Encode A^-1 g for SE(2) poses g, A.

        ``g_*``: world pose to encode; ``a_*``: anchor pose. Returns
        ``[clamp(px,±R), clamp(py,±R), cos(theta), sin(theta)]`` where
        (px,py,theta) = A^-1 g (g expressed in anchor frame).
        """
        if g_yaw.dim() == 2:
            g_yaw = g_yaw.squeeze(-1)
        if a_yaw.dim() == 2:
            a_yaw = a_yaw.squeeze(-1)
        d = g_xy - a_xy                     # world delta
        ca, sa = torch.cos(-a_yaw), torch.sin(-a_yaw)
        px = ca * d[:, 0] - sa * d[:, 1]
        py = sa * d[:, 0] + ca * d[:, 1]
        theta = g_yaw - a_yaw
        px = px.clamp(-clamp, clamp)
        py = py.clamp(-clamp, clamp)
        c, s = torch.cos(theta), torch.sin(theta)
        return torch.stack([px, py, c, s], dim=-1)

    def _normalize_key(self, key: str, val: torch.Tensor) -> torch.Tensor:
        """Run the single obs-normalizer sub-module for ``key`` (eval mode, no
        running-stat update)."""
        from ..modules.fb_cpr_policy import eval_mode
        norms = self.policy._obs_normalizer._normalizers
        if key not in norms:
            return val
        with torch.no_grad(), eval_mode(self.policy._obs_normalizer):
            return norms[key](val)

    # ------------------------------------------------------------------ #
    # The anchoring relabel (overrides FBCprAux no-op seam)
    # ------------------------------------------------------------------ #
    def _anchor_relabel(self, train_batch, train_obs, train_next_obs, train_z,
                        mixed_z, expert_z):
        cfg = self._anchor_cfg()
        dev = self.device
        B = train_z.shape[0]
        key = cfg["key"]

        extras = train_batch.get("extras", None)
        if extras is None or "root_xy" not in extras:
            # No world pose stored — behave like the base (identity). This
            # path should not hit in the anchored task (store_world_pose=True).
            return train_obs, train_next_obs, train_next_obs, train_z

        # Raw world SE(2) poses (current g_t, next g_{t+1}) from replay.
        gt_xy = extras["root_xy"].to(dev).float()
        gt_yaw = extras["root_yaw"].to(dev).float().view(-1)
        nx = train_batch["next"].get("extras", {})
        gtn_xy = nx.get("root_xy", extras["root_xy"]).to(dev).float()
        gtn_yaw = nx.get("root_yaw", extras["root_yaw"]).to(dev).float().view(-1)

        # ---- Task goal s_h ~ p_goal (mix future/nearby/replay/composed) ---
        # All four modes resolve to (goal world pose g_h, goal obs dict for the
        # LOCAL body part). We draw goal *body* states + goal *poses*, possibly
        # from different samples (composed).
        gh_xy, gh_yaw, goal_obs = self._sample_goal(train_batch, train_obs, gt_xy, gt_yaw, cfg)

        # ---- Sample coordinate anchor A ~ p_A (shared per sample) ----------
        a_xy, a_yaw = self._sample_anchor(gt_xy, gt_yaw, gh_xy, gh_yaw, cfg, B, dev)

        # ---- Inject anchored_pose into every obs dict under anchor A -------
        def _set_anchored(obs_dict, g_xy, g_yaw):
            ap = self._encode_anchored_pose(g_xy, g_yaw, a_xy, a_yaw, cfg["clamp"])
            ap = self._normalize_key(key, ap)
            new = dict(obs_dict)
            new[key] = ap
            return new

        train_obs = _set_anchored(train_obs, gt_xy, gt_yaw)
        train_next_obs = _set_anchored(train_next_obs, gtn_xy, gtn_yaw)
        goal_obs = _set_anchored(goal_obs, gh_xy, gh_yaw)

        # ---- FB successor goal = the (anchored) NEXT obs. The FB loss uses a
        # batch-matrix contrastive trick (Ms = F @ B.T): the DIAGONAL pairs
        # F(s_i,a_i,z_i) with B(next_obs_i) — the actual transition reward — and
        # the OFF-DIAGONAL rows are already the independent rho/successor-query
        # negatives. So ``goal`` MUST be next_obs (NOT a separate permutation;
        # an unrelated goal destroys the diagonal signal and F never learns →
        # Q_fb collapses to ~0). The independent-s_+ the spec asks for is the
        # off-diagonal batch, supplied for free here.
        fb_goal = train_next_obs

        # ---- z: local block keeps relabeled train_z's local part; spatial
        # block = B_spatial(anchored task goal). z = Normalize per-block. ----
        train_z = self._set_spatial_z(train_z, goal_obs)

        # Stash anchor context for the actor's anchor-consistency loss.
        self._anchor_ctx = {
            "a_xy": a_xy, "a_yaw": a_yaw,
            "gt_xy": gt_xy, "gt_yaw": gt_yaw,
            "gh_xy": gh_xy, "gh_yaw": gh_yaw,
            "goal_obs_local": {k: v for k, v in goal_obs.items() if k != key},
            "base_obs": {k: v for k, v in train_obs.items() if k != key},
            "z": train_z,
            "cfg": cfg,
        }
        return train_obs, train_next_obs, fb_goal, train_z

    # ------------------------------------------------------------------ #
    def _set_spatial_z(self, z, goal_obs):
        """Set z's spatial block = B_spatial(anchored goal), keep local block.

        Runs only the spatial backward head (B_local would be discarded)."""
        p = self.policy
        with torch.no_grad():
            z_spatial = p.backward_spatial(goal_obs)   # [B, z_spatial_dim]
        return torch.cat([z[:, : p.z_local_dim], z_spatial], dim=-1)

    def _sample_anchor(self, gt_xy, gt_yaw, gh_xy, gh_yaw, cfg, B, dev):
        """p_A = alpha·delta_{g_t} + beta·delta_{g_h} + (1-..)·random."""
        a_xy = gt_xy.clone()
        a_yaw = gt_yaw.clone()
        u = torch.rand(B, device=dev)
        alpha, beta = cfg["alpha_gt"], cfg["beta_gh"]
        is_gh = (u >= alpha) & (u < alpha + beta)
        is_rand = u >= (alpha + beta)
        # g_h anchor
        a_xy = torch.where(is_gh.unsqueeze(-1), gh_xy, a_xy)
        a_yaw = torch.where(is_gh, gh_yaw, a_yaw)
        # random anchor (around g_t)
        r = cfg["rand_xy"]
        rand_xy = gt_xy + (torch.rand(B, 2, device=dev) * 2 - 1) * r
        rand_yaw = (torch.rand(B, device=dev) * 2 - 1) * math.pi
        a_xy = torch.where(is_rand.unsqueeze(-1), rand_xy, a_xy)
        a_yaw = torch.where(is_rand, rand_yaw, a_yaw)
        return a_xy, a_yaw

    def _sample_goal(self, train_batch, train_obs, gt_xy, gt_yaw, cfg):
        """Return (goal world xy, goal world yaw, goal LOCAL-body obs dict).

        Mix of future / nearby / replay / composed. With per-transition replay
        we approximate ``future`` and ``replay`` by permutations of the current
        batch (the batch is i.i.d. across the buffer), ``nearby`` by selecting
        spatially-close rows, and ``composed`` by pairing a body state from one
        row with a spatial pose from another.
        """
        dev = self.device
        B = gt_xy.shape[0]
        probs = torch.tensor(
            [cfg["goal_future"], cfg["goal_nearby"], cfg["goal_replay"], cfg["goal_composed"]],
            device=dev, dtype=torch.float32,
        )
        probs = probs / probs.clamp_min(1e-8).sum()
        mode = torch.multinomial(probs, B, replacement=True)  # [B] in {0,1,2,3}

        # Base permutations.
        perm_body = torch.randperm(B, device=dev)     # body source
        perm_pose = torch.randperm(B, device=dev)     # spatial source (composed)

        # nearby: pick, per row, the spatially-closest OTHER row in the batch.
        with torch.no_grad():
            d = torch.cdist(gt_xy, gt_xy)              # [B,B]
            d.fill_diagonal_(float("inf"))
            nearby_idx = torch.argmin(d, dim=1)        # [B]

        # body source index per mode:
        #  future/replay -> perm_body (any other row); nearby -> nearby_idx;
        #  composed -> perm_body for body, perm_pose for pose.
        body_idx = perm_body.clone()
        body_idx = torch.where(mode == 1, nearby_idx, body_idx)
        pose_idx = body_idx.clone()
        pose_idx = torch.where(mode == 3, perm_pose, pose_idx)

        goal_obs = {k: v[body_idx] for k, v in train_obs.items()}
        gh_xy = gt_xy[pose_idx]
        gh_yaw = gt_yaw[pose_idx]
        return gh_xy, gh_yaw, goal_obs

    # ------------------------------------------------------------------ #
    # Anchor/gauge consistency (spec §7) — split across the two updates that
    # actually own the relevant parameters:
    #   * policy-KL consistency  -> actor update (_actor_extra_loss)
    #   * value (Q) consistency  -> F update     (_fb_extra_loss)
    # Putting the Q term in the actor update would be a NO-OP: it depends only
    # on F (Q=<F,z> on a detached action) and step_actor only steps the actor.
    # ------------------------------------------------------------------ #
    def _build_second_anchor(self, z):
        """Build the A2 branch (obs2, z2) for the SAME physical task as the
        A1-anchored ``obs``/``z`` baked this update. Returns (obs2, z2) or None
        when there's no stashed anchor context. Does NOT clear the context."""
        ctx = getattr(self, "_anchor_ctx", None)
        if ctx is None:
            return None
        p = self.policy
        cfg = ctx["cfg"]; key = cfg["key"]; dev = self.device
        B = z.shape[0]
        gt_xy, gt_yaw = ctx["gt_xy"], ctx["gt_yaw"]
        gh_xy, gh_yaw = ctx["gh_xy"], ctx["gh_yaw"]
        a2_xy, a2_yaw = self._sample_anchor(gt_xy, gt_yaw, gh_xy, gh_yaw, cfg, B, dev)

        def _anchored(obs_local, g_xy, g_yaw):
            ap = self._encode_anchored_pose(g_xy, g_yaw, a2_xy, a2_yaw, cfg["clamp"])
            ap = self._normalize_key(key, ap)
            d = dict(obs_local); d[key] = ap
            return d

        obs2 = _anchored(ctx["base_obs"], gt_xy, gt_yaw)
        goal2 = _anchored(ctx["goal_obs_local"], gh_xy, gh_yaw)
        with torch.no_grad():
            zs2 = p.backward_spatial(goal2)            # spatial head only
        z2 = torch.cat([z[:, : p.z_local_dim], zs2], dim=-1)
        return obs2, z2

    def _fb_extra_loss(self, obs, z):
        """Two-anchor VALUE consistency (spec §7), in the F update.

          L_Q = ( Q_{z^A1}(y^A1,a) - Q_{z^A2}(y^A2,a) )^2 ,  Q = <F, z>

        on a detached action shared across anchors, so gradient flows only
        through F (this is the update that steps F)."""
        lam_q = float(getattr(self.cfg, "anchor_q_coef", 0.0))
        if lam_q <= 0.0:
            return torch.zeros((), device=self.device), {}
        built = self._build_second_anchor(z)
        if built is None:
            return torch.zeros((), device=self.device), {}
        obs2, z2 = built
        p = self.policy
        with torch.no_grad():
            a = p._actor(obs, z, p.actor_std).sample(clip=self.cfg.stddev_clip).detach()
        q1 = (p._forward_map(obs, z, a) * z).sum(dim=-1)
        q2 = (p._forward_map(obs2, z2, a) * z2).sum(dim=-1)
        q_consist = (q1.mean(dim=0) - q2.mean(dim=0)).pow(2).mean()
        return lam_q * q_consist, {"anchor/q_consist": q_consist.detach()}

    def _actor_extra_loss(self, obs, z, dist=None, sampled_action=None):
        """Two-anchor POLICY-KL consistency (spec §7), in the actor update.

          L_pi = KL[ pi(.|y^A1, z^A1) || pi(.|y^A2, z^A2) ]

        Reuses the A1 actor dist from ``backward_actor`` (only A2 is forwarded).
        Clears the anchor context here — the actor is the LAST phase that uses
        it, so it must outlive the F update (_fb_extra_loss) which runs first.
        """
        lam_pi = float(getattr(self.cfg, "anchor_kl_coef", 0.0))
        if lam_pi <= 0.0:
            self._anchor_ctx = None
            return torch.zeros((), device=self.device), {}
        built = self._build_second_anchor(z)
        if built is None:
            self._anchor_ctx = None
            return torch.zeros((), device=self.device), {}
        obs2, z2 = built
        p = self.policy
        dist1 = dist if dist is not None else p._actor(obs, z, p.actor_std)
        dist2 = p._actor(obs2, z2, p.actor_std)
        m1, s1 = dist1.loc, dist1.scale
        m2, s2 = dist2.loc, dist2.scale
        var1, var2 = s1.pow(2) + 1e-8, s2.pow(2) + 1e-8
        kl = (torch.log(s2 / s1.clamp_min(1e-8)) + (var1 + (m1 - m2).pow(2)) / (2 * var2) - 0.5)
        kl = kl.sum(dim=-1).mean()
        loss = lam_pi * kl
        logs = {"anchor/kl": kl.detach()}
        # Actor is the last phase using the anchor ctx — clear it now.
        self._anchor_ctx = None
        return loss, logs
