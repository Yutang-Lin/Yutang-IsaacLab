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
    def _encode_anchored_pose(g_xy, g_yaw, a_xy, a_yaw, clamp):
        """A^-1 g -> [clamp(px,±R)/R, clamp(py,±R)/R, cosθ, sinθ] (all in [-1,1]).
        Byte-identical to the env's _obs_anchored_pose and the expert buffer's
        _anchored_pose_at."""
        if g_yaw.dim() == 2:
            g_yaw = g_yaw.squeeze(-1)
        if a_yaw.dim() == 2:
            a_yaw = a_yaw.squeeze(-1)
        d = g_xy - a_xy
        ca, sa = torch.cos(-a_yaw), torch.sin(-a_yaw)
        px = (ca * d[:, 0] - sa * d[:, 1]).clamp(-clamp, clamp) / clamp
        py = (sa * d[:, 0] + ca * d[:, 1]).clamp(-clamp, clamp) / clamp
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
    # The anchoring preamble (overrides FBCprAux no-op seam, runs BEFORE z)
    # ------------------------------------------------------------------ #
    def _anchor_obs_preamble(self, train_batch, train_obs, train_next_obs):
        """Overwrite the ``anchored_pose`` obs of train_obs / train_next_obs with
        ``enc(A^-1 g)`` under a per-row anchor A ~ p_A. Runs BEFORE z is built so
        the base sample_mixed_z (B(train_next_obs)) and the FB loss all see the
        SAME anchored frame -> the whole objective is anchor-equivariant. Single
        B reads anchored_pose, so no z surgery is needed.
        """
        cfg = self._anchor_cfg()
        dev = self.device
        key = cfg["key"]
        extras = train_batch.get("extras", None)
        if extras is None or "root_xy" not in extras:
            return train_obs, train_next_obs  # no world pose stored -> identity

        gt_xy = extras["root_xy"].to(dev).float()
        gt_yaw = extras["root_yaw"].to(dev).float().view(-1)
        nx = train_batch["next"].get("extras", {})
        gtn_xy = nx.get("root_xy", extras["root_xy"]).to(dev).float()
        gtn_yaw = nx.get("root_yaw", extras["root_yaw"]).to(dev).float().view(-1)

        a_xy, a_yaw = self._sample_anchor(gt_xy, gt_yaw, dev)

        def _anchored(obs_dict, g_xy, g_yaw):
            ap = self._encode_anchored_pose(g_xy, g_yaw, a_xy, a_yaw, cfg["clamp"])
            ap = self._normalize_key(key, ap)
            d = dict(obs_dict); d[key] = ap
            return d

        return _anchored(train_obs, gt_xy, gt_yaw), _anchored(train_next_obs, gtn_xy, gtn_yaw)
