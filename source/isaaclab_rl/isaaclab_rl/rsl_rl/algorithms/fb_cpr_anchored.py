# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anchored FB-CPR-Aux algorithm (Global-through-Anchoring) for BFM-One.

SINGLE-B formulation: the global pose ``A^-1 g`` is just an extra input to the
ONE backward map B (via ``backward_input_keys`` including ``anchored_pose``), so
a single z encodes the joint local+global goal. There is NO two-head B, no z
split, no spatial discriminator — everything is exactly original BFM
(``sample_mixed_z`` / FB loss / single local CPR discriminator).

The ONLY thing this subclass adds is **consistent + augmented anchoring**, done
purely as an obs relabel: the replay stores the raw world SE(2) pose
``(root_xy, root_yaw)`` per transition; each update we sample a per-row anchor
``A ~ p_A`` and overwrite the ``anchored_pose`` key of ``train_obs`` and
``train_next_obs`` with ``enc(A^-1 g)``. Because B / F / actor / critic all read
``anchored_pose``, and the FB diagonal uses the (anchored) next obs, the whole
FB objective is anchor-equivariant — identical to BFM, augmented over anchors.
z is then built by the base ``sample_mixed_z`` on the anchored next obs (B(next)
already carries the anchored pose), so no z surgery is needed here.
"""

from __future__ import annotations

import math

import torch

from .fb_cpr import FBCprAux
from ..storage.fb_cpr_storage import FBCprExpertBuffer


class AnchoredFBCprAux(FBCprAux):
    """FB-CPR-Aux + Global-through-Anchoring (single B; anchoring = obs relabel)."""

    # ------------------------------------------------------------------ #
    def _anchor_cfg(self):
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
    def _signed_log_unit(v, R, s0=1.0):
        """Signed-log range compression: sign(v)*log1p(|v|/s0)/log1p(R/s0).
        Near-linear for |v|<<s0, ->±1 at |v|=R. ``R`` is the full-scale range
        (m). MUST stay byte-identical across env/storage/algo/play encoders."""
        denom = math.log1p(R / s0)
        return torch.sign(v) * torch.log1p(v.abs() / s0) / denom

    @staticmethod
    def _encode_anchored_pose(g_xy, g_yaw, a_xy, a_yaw, clamp):
        """A^-1 g -> [signed_log(px,R), signed_log(py,R), cosθ, sinθ] (all in
        (-1,1)). ``clamp`` is the full-scale range R (m). Byte-identical to the
        env's _obs_anchored_pose and the expert buffer's _anchored_pose_at."""
        if g_yaw.dim() == 2:
            g_yaw = g_yaw.squeeze(-1)
        if a_yaw.dim() == 2:
            a_yaw = a_yaw.squeeze(-1)
        d = g_xy - a_xy
        ca, sa = torch.cos(-a_yaw), torch.sin(-a_yaw)
        px = AnchoredFBCprAux._signed_log_unit(ca * d[:, 0] - sa * d[:, 1], clamp)
        py = AnchoredFBCprAux._signed_log_unit(sa * d[:, 0] + ca * d[:, 1], clamp)
        theta = g_yaw - a_yaw
        return torch.stack([px, py, torch.cos(theta), torch.sin(theta)], dim=-1)

    def _normalize_key(self, key, val):
        """Run the obs-normalizer sub-module for ``key`` (eval mode, no stat
        update). anchored_pose is already analytic [-1,1]; BN is benign."""
        from ..modules.fb_cpr_policy import eval_mode
        norms = self.policy._obs_normalizer._normalizers
        if key not in norms:
            return val
        with torch.no_grad(), eval_mode(self.policy._obs_normalizer):
            return norms[key](val)

    def _sample_anchor(self, gt_xy, gt_yaw, dev):
        """Per-row anchor A ~ p_A: prob ``alpha`` at the current pose g_t (so
        A^-1 g_t ~ 0), else random around g_t (±range xy, ±π yaw)."""
        cfg = self._anchor_cfg()
        B = gt_xy.shape[0]
        a_xy = gt_xy.clone()
        a_yaw = gt_yaw.clone()
        is_rand = torch.rand(B, device=dev) >= cfg["alpha_gt"]
        r = cfg["rand_xy"]
        rand_xy = gt_xy + (torch.rand(B, 2, device=dev) * 2 - 1) * r
        rand_yaw = (torch.rand(B, device=dev) * 2 - 1) * math.pi
        a_xy = torch.where(is_rand.unsqueeze(-1), rand_xy, a_xy)
        a_yaw = torch.where(is_rand, rand_yaw, a_yaw)
        return a_xy, a_yaw

    # ------------------------------------------------------------------ #
    # Pre-normalize anchor-frame body-pose reframe (runs FIRST, on raw priv)
    # ------------------------------------------------------------------ #
    def _anchor_priv_pre_normalize(self, train_batch, expert_batch,
                                   train_obs, train_next_obs,
                                   expert_obs, expert_next_obs):
        """Sample the per-row anchor A_i, stash it for the anchored_pose preamble
        + z re-anchor seams, and (if ``anchor_frame_body``) reframe the RAW
        train + expert ``privileged_state`` body POS/ROT6D into A_i — BEFORE the
        normalizer, so BN stats track the anchor frame. Train priv arrives in
        heading frame (env emits heading-frame priv); expert priv likewise (the
        buffer keeps heading frame). Both reframe to A_i via the canonical root
        pose (A_i^-1 g_root), giving ONE consistent distribution to B/disc.
        """
        cfg = self._anchor_cfg()
        dev = self.device
        # Reset per-update raw-priv stashes (the goal/expert-z frame-consistency
        # seams read these; stale values across updates would corrupt the reframe).
        self._raw_train_next_priv = None
        self._raw_expert_next_priv = None
        extras = train_batch.get("extras", None)
        if extras is None or "root_xy" not in extras:
            self._row_anchor_xy = None  # signal: no anchoring this update
            return train_obs, train_next_obs, expert_obs, expert_next_obs

        gt_xy = extras["root_xy"].to(dev).float()
        gt_yaw = extras["root_yaw"].to(dev).float().view(-1)
        nx = train_batch["next"].get("extras", {})
        gtn_xy = nx.get("root_xy", extras["root_xy"]).to(dev).float()
        gtn_yaw = nx.get("root_yaw", extras["root_yaw"]).to(dev).float().view(-1)

        # One per-row anchor A_i, reused by the anchored_pose preamble below.
        a_xy, a_yaw = self._sample_anchor(gt_xy, gt_yaw, dev)
        self._row_anchor_xy = a_xy
        self._row_anchor_yaw = a_yaw
        self._train_next_canon = torch.cat([gtn_xy, gtn_yaw.view(-1, 1)], dim=-1)
        # Stash the train current/next canonical root poses (for the train priv
        # reframe — env priv is heading-frame so cr = A_i^-1 g_root).
        self._train_cur_world = (gt_xy, gt_yaw)
        self._train_nxt_world = (gtn_xy, gtn_yaw)

        if not bool(getattr(self.cfg, "anchor_frame_body", False)):
            return train_obs, train_next_obs, expert_obs, expert_next_obs

        K = self._priv_K(train_obs.get("privileged_state"))
        if K is None:
            return train_obs, train_next_obs, expert_obs, expert_next_obs
        R = cfg["clamp"]

        def _cr(g_xy, g_yaw):
            """Root pose (cr_xy metres, dθ) expressed in anchor A_i."""
            d = g_xy - a_xy
            ca, sa = torch.cos(-a_yaw), torch.sin(-a_yaw)
            cr = torch.stack([ca * d[:, 0] - sa * d[:, 1],
                              sa * d[:, 0] + ca * d[:, 1]], dim=-1)
            dth = torch.atan2(torch.sin(g_yaw - a_yaw), torch.cos(g_yaw - a_yaw))
            return cr, dth

        def _reframe(obs_dict, g_xy, g_yaw):
            cr, dth = _cr(g_xy, g_yaw)
            d = dict(obs_dict)
            d["privileged_state"] = FBCprExpertBuffer.reframe_priv_body(
                obs_dict["privileged_state"], cr, dth, K, R, True)
            return d

        # Stash the RAW (heading-frame, pre-reframe, pre-normalize) train-next
        # priv body block + K/R so the goal-z re-anchor seam can rebuild the body
        # block in row i's anchor A_i. reframe_priv_body's signed-log compression
        # is NON-LINEAR, so re-anchoring an already-reframed priv is impossible —
        # the seam MUST start from raw heading-frame priv. (Bug fix: goal-z was
        # feeding B a frame-INCONSISTENT (body in A_perm[i], anchored_pose in A_i)
        # pair; eval builds a consistent pair, so the actor trained off-manifold.)
        self._raw_train_next_priv = train_next_obs.get("privileged_state")
        self._anchor_priv_K = K
        self._anchor_priv_R = R

        train_obs = _reframe(train_obs, gt_xy, gt_yaw)
        train_next_obs = _reframe(train_next_obs, gtn_xy, gtn_yaw)

        # Expert: reframe per-row to a FRESH A_e drawn from the same p_A around
        # the expert's own (canonical) root pose, so BOTH the body POS/ROT6D and
        # the anchored_pose obs are expressed in ONE consistent random anchor.
        # This makes expert_z = B(expert) drawn under a random p_A anchor, just
        # like the policy disc-z (B(train_next_obs)) — so the disc cannot
        # shortcut on the anchor-correlated z distribution. The expert canonical
        # root pose rides at expert_batch['canon_pose']/['next']['canon_pose'].
        key = cfg["key"]
        ecp = expert_batch.get("canon_pose", None)
        ecp_n = expert_batch.get("next", {}).get("canon_pose", None)
        if ecp is not None:
            ecp = ecp.to(dev).float()
            e_xy, e_yaw = ecp[:, :2], ecp[:, 2]
            # Stash RAW heading-frame expert-next priv (pre-reframe) so the
            # expert-z re-anchor seam can rebuild the body block under row i's
            # anchor A_i (same frame-consistency fix as the goal-z path).
            self._raw_expert_next_priv = expert_next_obs.get("privileged_state")
            # ONE random anchor A_e per expert row, shared by cur+next frames.
            ea_xy, ea_yaw = self._sample_anchor(e_xy, e_yaw, dev)

            def _ecr(g_xy, g_yaw):
                d = g_xy - ea_xy
                ca, sa = torch.cos(-ea_yaw), torch.sin(-ea_yaw)
                cr = torch.stack([ca * d[:, 0] - sa * d[:, 1],
                                  sa * d[:, 0] + ca * d[:, 1]], dim=-1)
                dth = torch.atan2(torch.sin(g_yaw - ea_yaw), torch.cos(g_yaw - ea_yaw))
                return cr, dth

            def _reframe_expert(obs_dict, g_xy, g_yaw):
                cr, dth = _ecr(g_xy, g_yaw)
                d = dict(obs_dict)
                d["privileged_state"] = FBCprExpertBuffer.reframe_priv_body(
                    obs_dict["privileged_state"], cr, dth, K, R, True)
                # Re-anchor the anchored_pose obs to the SAME A_e (the buffer
                # emitted it under a different draw). Write it RAW — this runs
                # BEFORE the normalizer pass in update(), which then normalizes
                # priv AND anchored_pose exactly once (same as priv). Do NOT
                # pre-normalize here or it would be double-normalized.
                if key in obs_dict:
                    d[key] = self._encode_anchored_pose(g_xy, g_yaw, ea_xy, ea_yaw, R)
                return d

            expert_obs = _reframe_expert(expert_obs, e_xy, e_yaw)
            if ecp_n is not None:
                ecp_n = ecp_n.to(dev).float()
                expert_next_obs = _reframe_expert(expert_next_obs, ecp_n[:, :2], ecp_n[:, 2])

        return train_obs, train_next_obs, expert_obs, expert_next_obs

    def _priv_K(self, priv: torch.Tensor | None = None):
        """Number of keypoints K in the priv body block. Derived from the priv
        DIMENSION (robust — no cfg dependency, which is fragile: cfg keys not
        declared on the algo-cfg class are silently dropped by the runner's
        hasattr filter). Layout with root_height_obs=True:
          dim = 1 + (K-1)*3 + K*6 + K*3 + K*3 = 15K - 2  ->  K = (dim + 2) / 15.
        Falls back to cfg.expert_keypoint_names if priv is None. Cached."""
        K = getattr(self, "_priv_K_cache", "unset")
        if K != "unset":
            return K
        K = None
        if priv is not None:
            dim = int(priv.shape[-1])
            if (dim + 2) % 15 == 0:
                K = (dim + 2) // 15
        if K is None:
            names = getattr(self.cfg, "expert_keypoint_names", None)
            K = len(names) if names else None
        self._priv_K_cache = K
        return K

    def _disc_train_z(self, train_next_obs, train_z):
        """Policy disc-z under a RANDOM p_A anchor: re-encode B(train_next_obs).
        By the time this runs, the preamble has anchored train_next_obs
        (anchored_pose + priv body) under the per-row A_i ~ p_A, so this z is
        drawn from the same random-anchor distribution as expert_z = B(expert
        reframed under A_e ~ p_A). The disc therefore sees i.i.d. random-anchor
        z on both sides and must judge MOTION STYLE, not the anchor. (Falls back
        to the rollout z if no per-row anchor was sampled this update.)"""
        if getattr(self, "_row_anchor_xy", None) is None:
            return train_z
        with torch.no_grad():
            # train_next_obs is ALREADY normalized (preamble ran post-normalizer),
            # so use the raw module ``_backward_map`` directly — matching
            # sample_mixed_z / encode_expert, which do NOT re-normalize.
            return self.policy.project_z(self.policy._backward_map(train_next_obs))

    # ------------------------------------------------------------------ #
    # The anchoring preamble (overrides FBCprAux no-op seam, runs BEFORE z)
    # ------------------------------------------------------------------ #
    def _anchor_obs_preamble(self, train_batch, train_obs, train_next_obs):
        """Overwrite the ``anchored_pose`` obs of train_obs / train_next_obs with
        ``enc(A^-1 g)`` under the per-row anchor A_i sampled in
        ``_anchor_priv_pre_normalize``. Runs BEFORE z is built so the base
        sample_mixed_z (B(train_next_obs)) and the FB loss all see the SAME
        anchored frame -> the whole objective is anchor-equivariant.
        """
        cfg = self._anchor_cfg()
        key = cfg["key"]
        a_xy = getattr(self, "_row_anchor_xy", None)
        if a_xy is None:
            return train_obs, train_next_obs  # no world pose stored -> identity
        a_yaw = self._row_anchor_yaw
        gt_xy, gt_yaw = self._train_cur_world
        gtn_xy, gtn_yaw = self._train_nxt_world

        def _anchored(obs_dict, g_xy, g_yaw):
            ap = self._encode_anchored_pose(g_xy, g_yaw, a_xy, a_yaw, cfg["clamp"])
            ap = self._normalize_key(key, ap)
            d = dict(obs_dict); d[key] = ap
            return d

        return _anchored(train_obs, gt_xy, gt_yaw), _anchored(train_next_obs, gtn_xy, gtn_yaw)

    # ------------------------------------------------------------------ #
    # Cross-row z re-anchoring (fix: z's anchor must match obs's anchor)
    # ------------------------------------------------------------------ #
    def _reanchor_goal_z(self, shuffled, perm):
        """Re-anchor the shuffled goal's spatial channel to the DESTINATION
        row's anchor A_i. The goal-encoded z is z[i] = B(shuffled_body[i],
        anchored_pose = enc(A_i^-1 g_canon[perm[i]])). Body keys stay from
        perm[i] (the body goal); only anchored_pose is re-framed to row i, so
        z[i] and obs_i share frame A_i (matching deployment). Needs the
        shuffled goal's CANONICAL next-pose, recovered by inverting the anchored
        next-obs that the preamble already wrote (anchor there was A_{perm[i]}).
        """
        cfg = self._anchor_cfg()
        key = cfg["key"]
        a_xy = getattr(self, "_row_anchor_xy", None)
        a_yaw = getattr(self, "_row_anchor_yaw", None)
        gcanon = getattr(self, "_train_next_canon", None)
        if a_xy is None or gcanon is None or not isinstance(shuffled, dict):
            return shuffled  # nothing to re-anchor (non-anchored / missing pose)
        R = cfg["clamp"]
        dev = a_xy.device
        # g_canon[perm[i]] : canonical next-pose of the shuffled goal row.
        g = gcanon[perm]
        g_xy, g_yaw = g[:, :2], g[:, 2]
        # cr = goal expressed in row i's anchor A_i (the displacement the actor
        # must walk); dth = goal heading in A_i. This is exactly what both
        # anchored_pose AND the priv body reframe consume, so deriving them from
        # ONE (cr, dth) guarantees the frame-coherent (body, pose) pair the eval
        # path (_build_goal_z) uses.
        dd = g_xy - a_xy
        ca, sa = torch.cos(-a_yaw), torch.sin(-a_yaw)
        cr = torch.stack([ca * dd[:, 0] - sa * dd[:, 1],
                          sa * dd[:, 0] + ca * dd[:, 1]], dim=-1)
        dth = torch.atan2(torch.sin(g_yaw - a_yaw), torch.cos(g_yaw - a_yaw))

        # Bug-2 fix (travel-gated displaced practice): the real cr above is
        # bounded by how far the policy travels from spawn, so a drifting policy
        # never trains the actor on FAR goals. For a fraction of goal-z rows,
        # replace cr/dth with a FRESH wide random displacement (uniform ±range
        # xy, ±π yaw) so the actor gets far-reaching practice independent of
        # current rollout coverage. Off by default (prob 0). Applied to BOTH
        # channels so the (body, pose) pair stays coherent.
        # Only displace when the priv body can follow (anchor_frame_body); else
        # anchored_pose and the (un-reframable) body block would disagree.
        disp_prob = float(getattr(self.cfg, "goal_z_displace_prob", 0.0))
        if disp_prob > 0.0 and bool(getattr(self.cfg, "anchor_frame_body", False)) \
                and getattr(self, "_raw_train_next_priv", None) is not None:
            disp_range = float(getattr(self.cfg, "goal_z_displace_xy_range",
                                       getattr(self.cfg, "rollout_anchor_xy_range", 0.0)))
            B = cr.shape[0]
            use_rand = torch.rand(B, device=dev) < disp_prob
            rand_cr = (torch.rand(B, 2, device=dev) * 2 - 1) * disp_range
            rand_dth = (torch.rand(B, device=dev) * 2 - 1) * math.pi
            cr = torch.where(use_rand.unsqueeze(-1), rand_cr, cr)
            dth = torch.where(use_rand, rand_dth, dth)

        # anchored_pose = [signed_log(cr_x,R), signed_log(cr_y,R), cos dth, sin dth].
        ap = torch.stack([self._signed_log_unit(cr[:, 0], R),
                          self._signed_log_unit(cr[:, 1], R),
                          torch.cos(dth), torch.sin(dth)], dim=-1)
        ap = self._normalize_key(key, ap)
        d = dict(shuffled); d[key] = ap
        # Frame-consistency fix (Bug 1): re-express the goal row's priv BODY block
        # in the SAME anchor frame (cr, dth). Rebuild from the stashed RAW
        # heading-frame priv — reframe_priv_body's signed-log is non-linear, so
        # re-anchoring already-reframed priv is impossible.
        raw_priv = getattr(self, "_raw_train_next_priv", None)
        K = getattr(self, "_anchor_priv_K", None)
        if (bool(getattr(self.cfg, "anchor_frame_body", False))
                and raw_priv is not None and K is not None
                and "privileged_state" in shuffled):
            raw_goal_priv = raw_priv[perm]
            priv_i = FBCprExpertBuffer.reframe_priv_body(raw_goal_priv, cr, dth, K, R, True)
            d["privileged_state"] = self._normalize_key("privileged_state", priv_i)
        return d

    def _reanchor_expert_z(self, expert_encodings, idx):
        """Re-encode the picked expert window under the destination row's anchor
        A_i. Expert z is a T-window mean of B, so we rebuild anchored_pose_t =
        enc(A_i^-1 g_expert_canon_t) per frame, re-run B over the window's body
        obs, T-window-mean, project. Falls back to the precomputed
        expert_encodings[idx] if the canonical poses / window refs are absent.
        """
        cfg = self._anchor_cfg()
        key = cfg["key"]
        a_xy = getattr(self, "_row_anchor_xy", None)
        a_yaw = getattr(self, "_row_anchor_yaw", None)
        next_obs = getattr(self, "_expert_next_obs_ref", None)
        canon = getattr(self, "_expert_canon_pose", None)
        seq = getattr(self, "_expert_seq_length", None)
        T_per_seq = getattr(self, "_expert_T_per_seq", None)
        if (a_xy is None or next_obs is None or canon is None or seq is None
                or not isinstance(next_obs, dict) or key not in next_obs):
            return expert_encodings[idx]

        B = a_xy.shape[0]
        N_seq = canon.shape[0] // seq
        # Map each destination row i -> a sampled expert SUB-SEQUENCE. idx is
        # per-frame [batch]; collapse to the sub-sequence id (// seq).
        seq_of_row = (idx // seq).clamp_(max=N_seq - 1)  # [B]
        # Per-(row, frame-in-window) canonical expert pose, anchored under A_i.
        # canon: [N_seq*seq, 3]; gather the chosen sub-seq's frames for each row.
        frame_base = seq_of_row * seq                          # [B]
        fr = torch.arange(seq, device=a_xy.device).view(1, seq)  # [1,seq]
        gather_idx = (frame_base.view(B, 1) + fr).reshape(-1)    # [B*seq]
        g = canon[gather_idx]                                    # [B*seq, 3]
        # Repeat A_i across the window so each frame is anchored to row i.
        ax = a_xy.repeat_interleave(seq, dim=0)
        ay = a_yaw.repeat_interleave(seq, dim=0)
        ap = self._encode_anchored_pose(g[:, :2], g[:, 2], ax, ay, cfg["clamp"])
        ap = self._normalize_key(key, ap)
        # Rebuild the body obs for the chosen sub-seqs, frame-aligned.
        body = {}
        for k, v in next_obs.items():
            if k == key:
                continue
            vv = v.view(N_seq, seq, *v.shape[1:])[seq_of_row]    # [B, seq, ...]
            body[k] = vv.reshape(B * seq, *v.shape[1:])
        body[key] = ap
        # Frame-consistency fix: the priv body in next_obs was reframed under the
        # buffer's A_e, but anchored_pose above is under row i's A_i. Re-express
        # the priv BODY block under A_i too, rebuilt from RAW heading-frame expert
        # priv (signed-log reframe is non-linear -> cannot re-anchor reframed
        # priv). cr = A_i^-1 g_expert_canon_t per (row, frame).
        raw_e_priv = getattr(self, "_raw_expert_next_priv", None)
        K = getattr(self, "_anchor_priv_K", None)
        R = getattr(self, "_anchor_priv_R", None)
        if (bool(getattr(self.cfg, "anchor_frame_body", False))
                and raw_e_priv is not None and K is not None
                and "privileged_state" in body):
            raw_win = raw_e_priv.view(N_seq, seq, -1)[seq_of_row].reshape(B * seq, -1)
            dd = g[:, :2] - ax
            ca, sa = torch.cos(-ay), torch.sin(-ay)
            cr = torch.stack([ca * dd[:, 0] - sa * dd[:, 1],
                              sa * dd[:, 0] + ca * dd[:, 1]], dim=-1)
            dth = torch.atan2(torch.sin(g[:, 2] - ay), torch.cos(g[:, 2] - ay))
            priv_i = FBCprExpertBuffer.reframe_priv_body(raw_win, cr, dth, K, R, True)
            # next_obs body keys are normalized -> normalize the reframed priv too.
            body["privileged_state"] = self._normalize_key("privileged_state", priv_i)
        Bz = self.policy._backward_map(body).view(B, seq, -1)    # [B, seq, d]
        # T-window mean (per chosen sub-seq's T), matching encode_expert.
        if T_per_seq is not None:
            T = T_per_seq[seq_of_row].clamp(min=1, max=seq)      # [B]
            cumz = torch.cat([torch.zeros(B, 1, Bz.shape[-1], device=Bz.device),
                              torch.cumsum(Bz, dim=1)], dim=1)
            ar = torch.arange(B, device=Bz.device)
            z_sum = cumz[ar, T]
            z_re = z_sum / T.float().unsqueeze(-1)
        else:
            z_re = Bz.mean(dim=1)
        return self.policy.project_z(z_re)
