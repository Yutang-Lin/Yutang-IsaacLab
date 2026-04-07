# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CVAE-BFM: Foundation model CVAE decoder with per-frame tokens and pad masking."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab_rl.rsl_rl.networks.transformer import TransformerEncoder


class CVAEBFMDecoder(nn.Module):
    """Transformer action decoder with per-frame future tokens.

    Token layout: [proprio(0), latent(1), frame_0(2), ..., frame_{F-1}(F+1)]

    Each frame token is projected from: keypoint_data(K*D) + delta_t(1).
    Masked frames are excluded from attention via a dynamic pad mask.
    Action is decoded from the proprio token output.

    The latent token (z_t from CVAE prior/posterior) provides stochastic
    conditioning that captures mode information beyond what the sparse
    frames provide.
    """

    def __init__(
        self,
        proprio_dim: int,
        latent_dim: int,
        num_keypoints: int,
        dims_per_keypoint: int,
        max_frames: int,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        num_actions: int,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.max_frames = max_frames
        frame_input_dim = num_keypoints * dims_per_keypoint + 1  # K*D + delta_t

        # Token projections
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_input_dim, d_model),
            activation,
            nn.Linear(d_model, d_model),
        )

        # Learned embeddings for proprio and latent tokens
        self.proprio_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.latent_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        # Sinusoidal time embedding for frame tokens (computed from delta_t)
        # Precompute frequency bands: exp(-i * log(10000) / (d/2))
        half_d = d_model // 2
        freq = torch.exp(-torch.arange(half_d, dtype=torch.float32) * (math.log(10000.0) / half_d))
        self.register_buffer("_sin_freq", freq)  # [d/2]

        # Transformer
        self.transformer = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            is_causal=False,
            activation=activation,
            enable_sdpa=False,
        )

        # Action head from proprio token
        self.action_head = nn.Linear(d_model, num_actions)

    def _sinusoidal_embed(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal positional embedding from continuous time values.

        Args:
            t: [B, F] time in seconds.

        Returns:
            embed: [B, F, d_model]
        """
        # t: [B, F] → [B, F, 1] * [d/2] → [B, F, d/2]
        angles = t.unsqueeze(-1) * self._sin_freq  # [B, F, d/2]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B, F, d]

    def forward(
        self,
        o_t: torch.Tensor,
        z_t: torch.Tensor,
        frames_flat: torch.Tensor,
        delta_t: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            o_t: [B, proprio_dim] current proprio.
            z_t: [B, latent_dim] CVAE latent sample.
            frames_flat: [B, F, K*D] selected future keypoint data.
            delta_t: [B, F] time offset per frame (seconds).
            frame_mask: [B, F] bool, True=active, False=pad.

        Returns:
            action: [B, num_actions]
        """
        B, F = frames_flat.shape[:2]

        tok_proprio = self.proprio_proj(o_t) + self.proprio_embed  # [B, d]
        tok_latent = self.latent_proj(z_t) + self.latent_embed  # [B, d]

        # Frame tokens: concat keypoint data + delta_t, project
        frame_input = torch.cat([frames_flat, delta_t.unsqueeze(-1)], dim=-1)  # [B, F, K*D+1]
        tok_frames = self.frame_proj(frame_input)  # [B, F, d]
        # Sinusoidal time embedding from delta_t
        tok_frames = tok_frames + self._sinusoidal_embed(delta_t)  # [B, F, d]

        # Assemble: [B, F+2, d]
        tokens = torch.cat([
            tok_proprio.unsqueeze(1),
            tok_latent.unsqueeze(1),
            tok_frames,
        ], dim=1)

        # Build attention mask: [B, F+2, F+2]
        # proprio and latent always attend to each other and all active frames
        # masked frames can't be attended to or attend to anything
        total = F + 2
        attn_mask = torch.ones(B, total, total, dtype=torch.bool, device=o_t.device)

        # Mask out inactive frame columns (no one can attend to them)
        # and inactive frame rows (they can't attend to anything)
        frame_active = frame_mask  # [B, F]
        # Columns 2..F+1: set to False where frame is inactive
        attn_mask[:, :, 2:] &= frame_active.unsqueeze(1)  # [B, total, F]
        # Rows 2..F+1: set to False where frame is inactive
        attn_mask[:, 2:, :] &= frame_active.unsqueeze(2)  # [B, F, total]

        out = self.transformer(tokens, attn_mask=attn_mask)

        return self.action_head(out[:, 0])  # proprio token → action
