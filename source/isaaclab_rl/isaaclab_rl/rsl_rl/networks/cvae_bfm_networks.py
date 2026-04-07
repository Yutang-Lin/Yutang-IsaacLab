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


class CVAEBFMPrior(nn.Module):
    """Transformer-based prior for sparse frame conditions.

    Token layout: [history(0), frame_0(1), ..., frame_{F-1}(F)]

    Masked frames are pad-masked in attention. The history token aggregates
    information from available frames and outputs mu_prior, logvar_prior.
    1-layer lightweight transformer.
    """

    def __init__(
        self,
        h_dim: int,
        num_keypoints: int,
        dims_per_keypoint: int,
        latent_dim: int,
        d_model: int,
        num_heads: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        frame_input_dim = num_keypoints * dims_per_keypoint + 1  # K*D + delta_t

        self.history_proj = nn.Linear(h_dim, d_model)
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_input_dim, d_model),
            activation,
            nn.Linear(d_model, d_model),
        )

        self.history_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        # Sinusoidal time embedding
        half_d = d_model // 2
        freq = torch.exp(-torch.arange(half_d, dtype=torch.float32) * (math.log(10000.0) / half_d))
        self.register_buffer("_sin_freq", freq)

        self.transformer = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=1,
            dropout=dropout,
            is_causal=False,
            activation=activation,
            enable_sdpa=False,
        )

        # Output: mu and logvar from history token
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

    def _sinusoidal_embed(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.unsqueeze(-1) * self._sin_freq
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def forward(
        self,
        h_t: torch.Tensor,
        frames_flat: torch.Tensor,
        delta_t: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prior distribution from history + sparse frames.

        Args:
            h_t: [B, h_dim] L2-normalized history encoding.
            frames_flat: [B, F, K*D] keypoint data per frame (masked keypoints zeroed).
            delta_t: [B, F] time offset per frame.
            frame_mask: [B, F] bool, True=active.

        Returns:
            mu_prior: [B, latent_dim]
            logvar_prior: [B, latent_dim]
        """
        B, F = frames_flat.shape[:2]

        tok_history = self.history_proj(h_t) + self.history_embed  # [B, d]

        frame_input = torch.cat([frames_flat, delta_t.unsqueeze(-1)], dim=-1)
        tok_frames = self.frame_proj(frame_input) + self._sinusoidal_embed(delta_t)  # [B, F, d]

        tokens = torch.cat([tok_history.unsqueeze(1), tok_frames], dim=1)  # [B, F+1, d]

        # Attention mask: history attends to all active frames, masked frames excluded
        total = F + 1
        attn_mask = torch.ones(B, total, total, dtype=torch.bool, device=h_t.device)
        attn_mask[:, :, 1:] &= frame_mask.unsqueeze(1)   # columns
        attn_mask[:, 1:, :] &= frame_mask.unsqueeze(2)   # rows

        out = self.transformer(tokens, attn_mask=attn_mask)

        h_out = out[:, 0]  # history token output
        return self.mu_head(h_out), self.logvar_head(h_out)


class CVAEBFMDecoder(nn.Module):
    """Transformer action decoder with prior, posterior, and per-frame tokens.

    Token layout: [proprio(0), prior(1), posterior(2), frame_0(3), ..., frame_{F-1}(F+2)]

    - proprio: current proprioceptive state
    - prior: deterministic history encoding (h_prior from prior MLP)
    - posterior: correction token (c_t sampled from posterior, zero at inference)
    - frames: future keypoint targets with delta_t and pad masking

    Action is decoded from the proprio token output.
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
        self.prior_proj = nn.Linear(latent_dim, d_model)
        self.posterior_proj = nn.Linear(latent_dim, d_model)
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_input_dim, d_model),
            activation,
            nn.Linear(d_model, d_model),
        )

        # Learned embeddings
        self.proprio_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.prior_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.posterior_embed = nn.Parameter(torch.randn(d_model) * 0.02)

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
        h_prior: torch.Tensor,
        c_t: torch.Tensor,
        frames_flat: torch.Tensor,
        delta_t: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            o_t: [B, proprio_dim] current proprio.
            h_prior: [B, latent_dim] deterministic prior encoding.
            c_t: [B, latent_dim] posterior correction (zero at inference).
            frames_flat: [B, F, K*D] selected future keypoint data.
            delta_t: [B, F] time offset per frame (seconds).
            frame_mask: [B, F] bool, True=active, False=pad.

        Returns:
            action: [B, num_actions]
        """
        B, F = frames_flat.shape[:2]

        tok_proprio = self.proprio_proj(o_t) + self.proprio_embed
        tok_prior = self.prior_proj(h_prior) + self.prior_embed
        tok_posterior = self.posterior_proj(c_t) + self.posterior_embed

        # Frame tokens
        frame_input = torch.cat([frames_flat, delta_t.unsqueeze(-1)], dim=-1)
        tok_frames = self.frame_proj(frame_input)
        tok_frames = tok_frames + self._sinusoidal_embed(delta_t)

        # Assemble: [B, F+3, d]
        tokens = torch.cat([
            tok_proprio.unsqueeze(1),
            tok_prior.unsqueeze(1),
            tok_posterior.unsqueeze(1),
            tok_frames,
        ], dim=1)

        # Attention mask: proprio, prior, posterior always attend to each other
        # Masked frames excluded from attention
        total = F + 3
        attn_mask = torch.ones(B, total, total, dtype=torch.bool, device=o_t.device)
        frame_active = frame_mask
        attn_mask[:, :, 3:] &= frame_active.unsqueeze(1)
        attn_mask[:, 3:, :] &= frame_active.unsqueeze(2)

        out = self.transformer(tokens, attn_mask=attn_mask)

        return self.action_head(out[:, 0])  # proprio token → action
