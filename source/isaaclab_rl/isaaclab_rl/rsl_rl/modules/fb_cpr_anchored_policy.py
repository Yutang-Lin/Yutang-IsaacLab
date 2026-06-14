# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anchored FB-CPR policy (Global-through-Anchoring) for BFM-One-Anchored.

Extends :class:`FBCprAuxPolicy` with a TWO-HEAD backward map so a single
fixed latent ``z = [z_local | z_spatial]`` encodes BOTH a local body goal and
a global SE(2) spatial goal:

  * ``B_local``   : local body state (``state``/``privileged_state``) -> R^256
  * ``B_spatial`` : anchored pose ``A^-1 g`` (``anchored_pose``, 4-D) -> R^128

Each head is L2-projected to its OWN sphere (radius ``sqrt(256)`` and
``sqrt(128)``), then concatenated. Because ``256 + 128 = 384 = z_dim``, the
aggregate ``z`` still lies on the ``sqrt(z_dim)`` sphere, so the forward map
``F(y, a, z) -> R^384``, the implicit ``Q = <F, z>`` and all FB machinery are
structurally unchanged — only B's output structure, ``project_z`` and
``sample_z`` need per-block handling.

The anchored pose ``A^-1 g`` is supplied to F / actor / critic / aux_critic as
a regular obs key (``anchored_pose``, encoded ``[clamp(px,±R), clamp(py,±R),
cos θ, sin θ]``) — it is NOT routed into ``B_local`` (which stays a pure local
encoder) but IS the sole input to ``B_spatial``.
"""

from __future__ import annotations

import copy
import math
import typing as tp

import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fb_cpr_policy import (
    BackwardMap,
    Discriminator,
    FBCprAuxPolicy,
    FBCprNetworkCfg,
)


class TwoHeadBackwardMap(nn.Module):
    """B = [Normalize_l(B_local(x)) | Normalize_s(B_spatial(A^-1 g))].

    Two independent encoders, each projected to its own sphere radius
    (``sqrt(local_dim)`` / ``sqrt(spatial_dim)``) so the concatenation lies on
    the ``sqrt(local_dim + spatial_dim)`` sphere.
    """

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        local_dim: int,
        spatial_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 1,
        local_input_keys: str | tp.Sequence[str] | None = None,
        spatial_input_keys: str | tp.Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.local_dim = int(local_dim)
        self.spatial_dim = int(spatial_dim)
        # Each sub-head is UNNORMALIZED (norm=False); we apply the per-block
        # sphere projection in forward so the radii are independent.
        self.local = BackwardMap(
            obs_space, z_dim=local_dim, hidden_dim=hidden_dim,
            hidden_layers=hidden_layers, norm=False, input_keys=local_input_keys,
        )
        self.spatial = BackwardMap(
            obs_space, z_dim=spatial_dim, hidden_dim=hidden_dim,
            hidden_layers=hidden_layers, norm=False, input_keys=spatial_input_keys,
        )
        self._r_local = math.sqrt(local_dim)
        self._r_spatial = math.sqrt(spatial_dim)
        self._spatial_keys = list(self.spatial.input_filter.keys) if hasattr(
            self.spatial.input_filter, "keys") else [
            getattr(self.spatial.input_filter, "key", "anchored_pose")]

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        zl = self._r_local * F.normalize(self.local(x), dim=-1)
        # B_spatial needs the anchored-pose key. When it's absent (e.g. expert
        # obs, which carry no spatial goal), zero-fill it — the spatial z block
        # for those samples is overwritten downstream (random / B(goal)) so the
        # value here is immaterial, we just need a valid forward pass.
        if isinstance(x, dict) and any(k not in x for k in self._spatial_keys):
            ref = next(iter(x.values()))
            x = dict(x)
            for k in self._spatial_keys:
                if k not in x:
                    dim = self.spatial.input_filter.output_space.shape[0]
                    x[k] = torch.zeros(ref.shape[0], dim, device=ref.device, dtype=ref.dtype)
        zs = self._r_spatial * F.normalize(self.spatial(x), dim=-1)
        return torch.cat([zl, zs], dim=-1)


class AnchoredFBCprPolicy(FBCprAuxPolicy):
    """FB-CPR-Aux policy with a two-head (local ⊕ spatial) backward map."""

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        action_dim: int,
        cfg: FBCprNetworkCfg,
    ) -> None:
        super().__init__(obs_space, action_dim=action_dim, cfg=cfg)
        self.z_local_dim: int = int(getattr(cfg, "z_local_dim", 256))
        self.z_spatial_dim: int = int(getattr(cfg, "z_spatial_dim", 128))
        assert self.z_local_dim + self.z_spatial_dim == self.z_dim, (
            f"z_local_dim ({self.z_local_dim}) + z_spatial_dim "
            f"({self.z_spatial_dim}) must equal z_dim ({self.z_dim})."
        )
        self._r_local = math.sqrt(self.z_local_dim)
        self._r_spatial = math.sqrt(self.z_spatial_dim)

        # Replace the single-head B with the two-head version. B_local reuses
        # the original backward_input_keys (local body state); B_spatial reads
        # the anchored-pose key.
        spatial_keys = getattr(cfg, "spatial_input_keys", ("anchored_pose",))
        self._backward_map = TwoHeadBackwardMap(
            obs_space,
            local_dim=self.z_local_dim,
            spatial_dim=self.z_spatial_dim,
            hidden_dim=cfg.backward_hidden_dim,
            hidden_layers=cfg.backward_hidden_layers,
            local_input_keys=cfg.backward_input_keys,
            spatial_input_keys=spatial_keys,
        )
        # Targets are (re)built in _prepare_for_train via deepcopy, so the
        # two-head B is picked up automatically there.

        # --- Split CPR: TWO discriminators on the SAME local body obs, each
        # conditioned on a DIFFERENT z block. The base built ``_discriminator``
        # with z_dim=384 (full z); rebuild it for z_local only, and add a
        # spatial discriminator conditioned on z_spatial. Both read the local
        # discriminator obs keys (state, privileged_state) — the spatial disc
        # judges "is this local motion plausible for the given spatial goal z?".
        disc_keys = cfg.discriminator_input_keys
        self._discriminator = Discriminator(
            obs_space,
            z_dim=self.z_local_dim,
            hidden_dim=cfg.discriminator_hidden_dim,
            hidden_layers=cfg.discriminator_hidden_layers,
            input_keys=disc_keys,
            zero_obs_tail_dims=getattr(cfg, "discriminator_zero_obs_tail_dims", 0),
        )
        self._discriminator_spatial = Discriminator(
            obs_space,
            z_dim=self.z_spatial_dim,
            hidden_dim=cfg.discriminator_hidden_dim,
            hidden_layers=cfg.discriminator_hidden_layers,
            input_keys=disc_keys,
            zero_obs_tail_dims=getattr(cfg, "discriminator_zero_obs_tail_dims", 0),
        )

    # ---- per-block latent projection / sampling ----

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        """Project the local and spatial blocks to their OWN spheres."""
        if self.soft_fb:
            # Soft-FB squashing is not supported for the anchored variant.
            return super().project_z(z)
        if not self.norm_z:
            return z
        zl, zs = z[..., : self.z_local_dim], z[..., self.z_local_dim:]
        # Clamp the norm before normalizing. A windowed mean of per-frame z's
        # (each on its sphere) can cancel to ≈0 — especially for the small,
        # untrained spatial block — and F.normalize's tiny default eps then
        # blows up to nan. clamp(min=1e-6) keeps the direction well-defined.
        zl = self._r_local * zl / zl.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        zs = self._r_spatial * zs / zs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.cat([zl, zs], dim=-1)

    def sample_z(self, batch_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """Sample z with each block uniform on its own sphere."""
        zl = F.normalize(torch.randn(batch_size, self.z_local_dim, device=device), dim=-1)
        zs = F.normalize(torch.randn(batch_size, self.z_spatial_dim, device=device), dim=-1)
        if self.norm_z:
            zl = zl * self._r_local
            zs = zs * self._r_spatial
        return torch.cat([zl, zs], dim=-1)

    def sample_z_spatial(self, batch_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """Sample ONLY the spatial block (uniform on its sphere). Used to pair a
        random spatial goal with an expert/encoded local z."""
        zs = F.normalize(torch.randn(batch_size, self.z_spatial_dim, device=device), dim=-1)
        if self.norm_z:
            zs = zs * self._r_spatial
        return zs

    # ---- z block accessors (for the split CPR discriminators) ----

    def z_local(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., : self.z_local_dim]

    def z_spatial(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., self.z_local_dim:]

    def backward_spatial(self, obs) -> torch.Tensor:
        """Run ONLY the spatial backward head (B_spatial) + its sphere
        projection, returning the [B, z_spatial_dim] block. Avoids the wasted
        B_local forward when only the spatial goal latent is needed."""
        bm = self._backward_map
        zs = bm.spatial(obs)
        return bm._r_spatial * zs / zs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
