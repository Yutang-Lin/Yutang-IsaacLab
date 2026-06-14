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

    def _q_fb_split(self, Fs, z):
        """Split the implicit Q = <F, z> into the LOCAL (z_local) and SPATIAL
        (z_spatial) block contributions and log each (mean over the ensemble
        + batch). Q_fb = Q_fb_local + Q_fb_spatial by construction."""
        dl = self.policy.z_local_dim
        with torch.no_grad():
            q_local = (Fs[..., :dl] * z[..., :dl]).sum(dim=-1)       # [num_par, B]
            q_spatial = (Fs[..., dl:] * z[..., dl:]).sum(dim=-1)     # [num_par, B]
            return {
                "Q_fb_local": q_local.mean().detach(),
                "Q_fb_spatial": q_spatial.mean().detach(),
            }

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
            # p(anchor = current pose g_t); remainder -> random anchor around g_t.
            "alpha_gt": float(getattr(c, "anchor_alpha_gt", 0.34)),
            "rand_xy": float(getattr(c, "anchor_random_xy_range", 10.0)),
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
        ``[clamp(px,±R)/R, clamp(py,±R)/R, cos(theta), sin(theta)]`` where
        (px,py,theta) = A^-1 g (g expressed in anchor frame).

        The xy is divided by the clamp R so all four dims live in [-1, 1] —
        a fixed, stationary, analytically-normalized encoding. This replaces
        BatchNorm on anchored_pose (whose running stats were fit on the narrow
        rollout distribution but applied to the broad relabel distribution — a
        train/eval mismatch that destabilized training). Must stay byte-identical
        to the env's _obs_anchored_pose and the expert buffer's _anchored_pose_at.
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
        px = px.clamp(-clamp, clamp) / clamp
        py = py.clamp(-clamp, clamp) / clamp
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
        """Consistent + augmented anchoring on TOP of original BFM.

        We do NOT change BFM's resampling / FB loss / z-mixing. The only thing
        added is a per-row SE(2) anchor A_i, applied CONSISTENTLY so the FB loss
        is identical-but-augmented (anchor-equivariant): for each row i, the
        actor/critic/F obs (A_i^-1 g_t), the FB-diagonal goal B(A_i^-1 g_{t+1}),
        AND the spatial block of z all live in the SAME frame A_i.

        Original BFM z (``mixed_z``, from sample_mixed_z on next_obs) is reused
        verbatim, EXCEPT its spatial block must be re-expressed in A_i: the
        goal-encoded branch of sample_mixed_z used a shuffled next_obs whose
        ``anchored_pose`` was the rollout frame, not A_i. We recompute that
        spatial block as B_spatial(A_i^-1 g_j) where j is the same shuffle. The
        local block (anchor-invariant) is left exactly as BFM produced it.
        """
        cfg = self._anchor_cfg()
        dev = self.device
        B = train_z.shape[0]
        key = cfg["key"]
        p = self.policy

        extras = train_batch.get("extras", None)
        if extras is None or "root_xy" not in extras:
            return train_obs, train_next_obs, train_next_obs, train_z

        # World SE(2) poses for this transition (g_t) and its next (g_{t+1}).
        gt_xy = extras["root_xy"].to(dev).float()
        gt_yaw = extras["root_yaw"].to(dev).float().view(-1)
        nx = train_batch["next"].get("extras", {})
        gtn_xy = nx.get("root_xy", extras["root_xy"]).to(dev).float()
        gtn_yaw = nx.get("root_yaw", extras["root_yaw"]).to(dev).float().view(-1)

        # ---- Per-row anchor A_i ~ p_A (augmentation): current pose / random.
        a_xy, a_yaw = self._sample_anchor(gt_xy, gt_yaw, dev)

        def _anchored_obs(obs_dict, g_xy, g_yaw):
            ap = self._encode_anchored_pose(g_xy, g_yaw, a_xy, a_yaw, cfg["clamp"])
            ap = self._normalize_key(key, ap)
            d = dict(obs_dict); d[key] = ap
            return d

        # Obs F/actor/critic see, and the FB-diagonal goal — all in frame A_i.
        train_obs = _anchored_obs(train_obs, gt_xy, gt_yaw)
        train_next_obs = _anchored_obs(train_next_obs, gtn_xy, gtn_yaw)
        fb_goal = train_next_obs   # FB diagonal = actual next state, in A_i.

        # ---- z: original-BFM mix (random / goal-encoded / expert), but with
        # the SPATIAL block built under A_i so z is frame-consistent with the
        # obs and the FB diagonal. See _build_anchored_z.
        train_z = self._build_anchored_z(train_obs, train_next_obs, gtn_xy,
                                         gtn_yaw, a_xy, a_yaw, expert_z, cfg)
        return train_obs, train_next_obs, fb_goal, train_z

    # ------------------------------------------------------------------ #
    def _build_anchored_z(self, train_obs, train_next_obs, gtn_xy, gtn_yaw,
                          a_xy, a_yaw, expert_z, cfg):
        """Original-BFM ``sample_mixed_z`` mix, frame-consistent under A_i.

        Replicates BFM exactly — per row, z is one of:
          * random        (prob 1 - p_goal - p_expert): uniform per-block sphere
          * goal-encoded   (prob train_goal_ratio): B(shuffled next_obs)
          * expert-encoded (prob expert_asm_ratio):  expert_z

        The ONLY anchoring change: the goal-encoded branch's SPATIAL block is
        B_spatial(A_i^-1 g_j) where j is the shuffle — i.e. the shuffled goal
        state's pose expressed in ROW i's anchor A_i (so z_i lives in the same
        frame as F(obs_i) and the diagonal B(next_obs_i)). The local block is
        anchor-invariant and uses B_local(shuffled next_obs) verbatim, as BFM.
        Random and expert z keep their own spatial blocks (random sphere /
        expert's own frame) — unchanged from BFM, never overwritten.
        """
        p = self.policy
        dev = self.device
        B = next(iter(train_next_obs.values())).shape[0]
        key = cfg["key"]
        # random z (per-block sphere) — BFM's base draw.
        z = p.sample_z(B, device=dev)
        # goal-encoded z: shuffle next_obs (BFM's perm), but rewrite the spatial
        # key to the shuffled pose expressed in THIS row's anchor A_i.
        perm = torch.randperm(B, device=dev)
        shuffled = {k: v[perm] for k, v in train_next_obs.items()}
        with torch.no_grad():
            ap = self._encode_anchored_pose(gtn_xy[perm], gtn_yaw[perm],
                                            a_xy, a_yaw, cfg["clamp"])
            shuffled[key] = self._normalize_key(key, ap)
            z_goal = p.project_z(p._backward_map(shuffled))
        p_goal = float(getattr(self.cfg, "train_goal_ratio", 0.2))
        p_expert = float(getattr(self.cfg, "expert_asm_ratio", 0.6))
        probs = torch.tensor([p_goal, p_expert, max(0.0, 1.0 - p_goal - p_expert)],
                             device=dev, dtype=torch.float32)
        mix = torch.multinomial(probs, B, replacement=True).view(-1, 1)
        z = torch.where(mix == 0, z_goal, z)
        if expert_z is not None and expert_z.shape[0] > 0:
            idx = torch.randint(0, expert_z.shape[0], (B,), device=dev)
            z = torch.where(mix == 1, expert_z[idx], z)
        return z

    def _sample_anchor(self, gt_xy, gt_yaw, dev):
        """Per-row anchor A_i ~ p_A: alpha at the current pose g_t (so
        A_i^-1 g_t ~ 0), the rest random around g_t for translation/rotation
        augmentation. (No g_h branch — there is no separate goal state; z's
        spatial goal is the row's own next pose, per original-BFM mixing.)"""
        B = gt_xy.shape[0]
        a_xy = gt_xy.clone()
        a_yaw = gt_yaw.clone()
        alpha = float(self._anchor_cfg()["alpha_gt"])
        is_rand = torch.rand(B, device=dev) >= alpha
        r = float(self._anchor_cfg()["rand_xy"])
        rand_xy = gt_xy + (torch.rand(B, 2, device=dev) * 2 - 1) * r
        rand_yaw = (torch.rand(B, device=dev) * 2 - 1) * math.pi
        a_xy = torch.where(is_rand.unsqueeze(-1), rand_xy, a_xy)
        a_yaw = torch.where(is_rand, rand_yaw, a_yaw)
        return a_xy, a_yaw

    # NOTE: no anchor-consistency losses. Anchor invariance emerges from
    # standard FB TD over anchor-relabeled transitions (z = B(goal) is
    # equivariant under the anchor; the Bellman backup aligns Q=<F,z> and the
    # policy across anchors). The base ``_fb_extra_loss`` / ``_actor_extra_loss``
    # no-op seams are inherited unchanged. This keeps the anchored variant pure
    # standard FB (local and spatial treated identically up to the A^-1 pose
    # transform) and avoids the extra F/actor forward passes that destabilised
    # training under DDP.
