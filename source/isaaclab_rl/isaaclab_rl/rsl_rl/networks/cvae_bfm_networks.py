# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CVAE-BFM networks: shared frame encoder, transformer posterior, and decoder."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab_rl.rsl_rl.networks.transformer import TransformerEncoder


class BFMFrameEncoder(nn.Module):
    """Shared frame token encoder: projects (keypoint_data, delta_t) → d_model.

    Used by both the posterior and the decoder so frame features are aligned.
    Includes sinusoidal time embedding from delta_t.
    """

    def __init__(self, num_keypoints: int, dims_per_keypoint: int, d_model: int,
                 activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")
        frame_input_dim = num_keypoints * dims_per_keypoint + 1  # K*D + delta_t
        self.proj = nn.Sequential(
            nn.Linear(frame_input_dim, d_model), activation, nn.Linear(d_model, d_model),
        )
        half_d = d_model // 2
        freq = torch.exp(-torch.arange(half_d, dtype=torch.float32) * (math.log(10000.0) / half_d))
        self.register_buffer("_sin_freq", freq)

    def forward(self, frames_flat: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """Encode frame tokens.

        Args:
            frames_flat: [B, F, K*D]
            delta_t: [B, F]

        Returns:
            tok_frames: [B, F, d_model]
        """
        frame_input = torch.cat([frames_flat, delta_t.unsqueeze(-1)], dim=-1)
        tok = self.proj(frame_input)
        # Sinusoidal time embedding
        angles = delta_t.unsqueeze(-1) * self._sin_freq
        tok = tok + torch.cat([angles.sin(), angles.cos()], dim=-1)
        return tok


def _build_frame_attn_mask(B: int, F: int, frame_mask: torch.Tensor,
                           n_prefix: int, device: torch.device) -> torch.Tensor:
    """Build attention mask with n_prefix always-active tokens + F frame tokens.

    Args:
        B: batch size
        F: number of frame slots
        frame_mask: [B, F] bool
        n_prefix: number of fixed tokens before frames (e.g., 1 for posterior, 3 for decoder)
        device: torch device

    Returns:
        attn_mask: [B, n_prefix+F, n_prefix+F]
    """
    total = n_prefix + F
    attn_mask = torch.ones(B, total, total, dtype=torch.bool, device=device)
    attn_mask[:, :, n_prefix:] &= frame_mask.unsqueeze(1)
    attn_mask[:, n_prefix:, :] &= frame_mask.unsqueeze(2)
    return attn_mask


class CVAEBFMPosterior(nn.Module):
    """Transformer posterior: clean keybody + shared frame tokens → (mu, logvar).

    Token layout: [keybody(0), frame_0(1), ..., frame_{F-1}(F)]
    1-layer transformer. Masked frames are pad-masked.
    """

    def __init__(self, keybody_dim: int, latent_dim: int, d_model: int,
                 frame_encoder: BFMFrameEncoder,
                 num_heads: int = 4, hidden_dim: int = 512,
                 dropout: float = 0.0, activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.frame_encoder = frame_encoder  # shared with decoder

        self.keybody_proj = nn.Linear(keybody_dim, d_model)
        self.keybody_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        self.transformer = TransformerEncoder(
            d_model=d_model, num_heads=num_heads, hidden_dim=hidden_dim,
            num_layers=1, dropout=dropout, is_causal=False,
            activation=activation, enable_sdpa=False,
        )
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

    def forward(self, r_t, frames_flat, delta_t, frame_mask):
        """
        Args:
            r_t: [B, keybody_dim]
            frames_flat: [B, F, K*D] (masked keypoints zeroed)
            delta_t: [B, F]
            frame_mask: [B, F] bool

        Returns:
            mu, logvar: [B, latent_dim] each
        """
        B, F = frames_flat.shape[:2]
        tok_kb = self.keybody_proj(r_t) + self.keybody_embed
        tok_frames = self.frame_encoder(frames_flat, delta_t)

        tokens = torch.cat([tok_kb.unsqueeze(1), tok_frames], dim=1)
        attn_mask = _build_frame_attn_mask(B, F, frame_mask, n_prefix=1, device=r_t.device)

        out = self.transformer(tokens, attn_mask=attn_mask)
        kb_out = out[:, 0]
        return self.mu_head(kb_out), self.logvar_head(kb_out)


class CVAEBFMDecoder(nn.Module):
    """Transformer action decoder with prior, posterior, and shared frame tokens.

    Token layout: [proprio(0), prior(1), posterior(2), frame_0(3), ..., frame_{F-1}(F+2)]

    Frame tokens use the same BFMFrameEncoder as the posterior for feature alignment.
    Action is decoded from the proprio token output.
    """

    def __init__(self, proprio_dim: int, latent_dim: int, max_frames: int,
                 d_model: int, frame_encoder: BFMFrameEncoder,
                 num_heads: int = 4, hidden_dim: int = 512, num_layers: int = 2,
                 num_actions: int = 29, dropout: float = 0.0,
                 activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.frame_encoder = frame_encoder  # shared with posterior

        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.prior_proj = nn.Linear(latent_dim, d_model)
        self.posterior_proj = nn.Linear(latent_dim, d_model)

        self.proprio_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.prior_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.posterior_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        self.transformer = TransformerEncoder(
            d_model=d_model, num_heads=num_heads, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, is_causal=False,
            activation=activation, enable_sdpa=False,
        )
        self.action_head = nn.Linear(d_model, num_actions)

    def forward(self, o_t, h_prior, c_t, frames_flat, delta_t, frame_mask):
        """
        Args:
            o_t: [B, proprio_dim]
            h_prior: [B, latent_dim]
            c_t: [B, latent_dim] (zero at inference)
            frames_flat: [B, F, K*D]
            delta_t: [B, F]
            frame_mask: [B, F] bool

        Returns:
            action: [B, num_actions]
        """
        B, F = frames_flat.shape[:2]

        tok_proprio = self.proprio_proj(o_t) + self.proprio_embed
        tok_prior = self.prior_proj(h_prior) + self.prior_embed
        tok_posterior = self.posterior_proj(c_t) + self.posterior_embed
        tok_frames = self.frame_encoder(frames_flat, delta_t)

        tokens = torch.cat([
            tok_proprio.unsqueeze(1),
            tok_prior.unsqueeze(1),
            tok_posterior.unsqueeze(1),
            tok_frames,
        ], dim=1)

        attn_mask = _build_frame_attn_mask(B, F, frame_mask, n_prefix=3, device=o_t.device)
        out = self.transformer(tokens, attn_mask=attn_mask)

        return self.action_head(out[:, 0])
