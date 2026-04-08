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
    # Diagonal self-loop for frame tokens: masked frames attend to themselves
    diag_idx = torch.arange(n_prefix, total, device=device)
    attn_mask[:, diag_idx, diag_idx] = True
    return attn_mask


class CVAEBFMPosterior(nn.Module):
    """Cross-attention posterior: keybody queries against shared frame tokens.

    Keybody token cross-attends to frame tokens (with pad mask).
    Frames don't self-attend or pass through FFN — they're just a KV bank.
    """

    def __init__(self, keybody_dim: int, latent_dim: int, d_model: int,
                 frame_encoder: BFMFrameEncoder,
                 num_heads: int = 4, hidden_dim: int = 512,
                 dropout: float = 0.0, activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.frame_encoder = frame_encoder  # shared with decoder

        self.keybody_proj = nn.Linear(keybody_dim, d_model)
        self.keybody_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        # Cross-attention: Q from keybody, KV from frames
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN on keybody only
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim), activation, nn.Linear(hidden_dim, d_model),
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

        tok_kb = self.keybody_proj(r_t) + self.keybody_embed  # [B, d]
        tok_frames = self.frame_encoder(frames_flat, delta_t)  # [B, F, d]

        # Cross-attention: keybody (Q) attends to frames (KV)
        q = self.q_proj(tok_kb).unsqueeze(1)  # [B, 1, d]
        k = self.k_proj(tok_frames)  # [B, F, d]
        v = self.v_proj(tok_frames)  # [B, F, d]

        # Reshape for multi-head: [B, H, L, hd]
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, F, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, F, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention with frame mask: [B, 1, F] → [B, H, 1, F]
        attn_mask = frame_mask[:, None, None, :].expand(-1, self.num_heads, 1, -1)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.masked_fill(~attn_mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, 1, self.d_model)
        out = self.out_proj(out).squeeze(1)  # [B, d]

        # Residual + norm + FFN
        kb_out = self.norm1(tok_kb + out)
        kb_out = self.norm2(kb_out + self.ffn(kb_out))

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

    def forward(self, o_t, h_prior, c_t, frames_flat, delta_t, frame_mask, pre_encoded=False):
        """
        Args:
            o_t: [B, proprio_dim] or [B, d_model] if pre_encoded
            h_prior: [B, latent_dim] or [B, d_model] if pre_encoded
            c_t: [B, latent_dim] (zero at inference)
            frames_flat: [B, F, K*D]
            delta_t: [B, F]
            frame_mask: [B, F] bool
            pre_encoded: if True, o_t and h_prior are already in d_model space
                (from prior transformer output), skip projection, add embed only.

        Returns:
            action: [B, num_actions]
        """
        B, F = frames_flat.shape[:2]

        if pre_encoded:
            tok_proprio = o_t + self.proprio_embed
            tok_prior = h_prior + self.prior_embed
        else:
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
