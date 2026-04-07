# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CoDiT-MF: CoDiT-Track with MeanFlow denoising and contrastive features.

Extends the base CoDiT transformer with:
  - MeanFlow velocity prediction head (replaces direct denoising)
  - JVP-friendly denoise_only() path for self-consistency training
  - Future feature extraction for contrastive regularization
"""

from __future__ import annotations

import torch
import torch.nn as nn

from isaaclab_rl.rsl_rl.networks.transformer import TransformerEncoder


class CoDiTMFTransformer(nn.Module):
    """CoDiT Transformer with MeanFlow velocity prediction.

    Same T+2 token architecture as CoDiTTransformer, but:
    - Future tokens are conditioned on both t and r: input = [y_flat, t, r]
    - Velocity head predicts average velocity u(y_t, r, t) instead of clean y
    - Forward returns future features for contrastive loss
    - denoise_only() provides JVP-compatible path through (y, t, r) → u
    """

    def __init__(
        self,
        proprio_dim: int,
        history_dim: int,
        num_keypoints: int,
        dims_per_keypoint: int,
        num_future_frames: int,
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

        self.num_future_frames = num_future_frames
        self.num_keypoints = num_keypoints
        self.dims_per_keypoint = dims_per_keypoint
        future_raw_dim = num_keypoints * dims_per_keypoint  # K*D per frame
        future_token_input_dim = future_raw_dim + 2 * num_keypoints  # K*D + t(K) + r(K)
        future_output_dim = future_raw_dim  # velocity K*D per frame

        # --- Token projections ---
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.history_proj = nn.Linear(history_dim, d_model)

        self.future_proj = nn.Sequential(
            nn.Linear(future_token_input_dim, d_model),
            activation,
            nn.Linear(d_model, d_model),
        )

        # --- Learned embeddings ---
        self.proprio_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.history_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.future_index_embed = nn.Embedding(num_future_frames, d_model)

        # --- Transformer encoder ---
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

        # --- Output heads ---
        self.base_head = nn.Linear(d_model, num_actions)
        self.cond_head = nn.Linear(d_model, num_actions)
        self.velocity_head = nn.Linear(d_model, future_output_dim)

        # --- Attention mask ---
        self._build_attention_mask(num_future_frames)

    def _build_attention_mask(self, T: int):
        total = T + 2
        mask = torch.ones(total, total, dtype=torch.bool)
        mask[1, 2:] = False
        self.register_buffer("attn_mask", mask.unsqueeze(0))

    def _build_future_tokens(self, y_corrupted_flat, t, r):
        """Build future token embeddings from (y, t, r)."""
        future_input = torch.cat([y_corrupted_flat, t, r], dim=-1)  # [B, T, K*D + 2K]
        tok_future = self.future_proj(future_input)  # [B, T, d_model]
        indices = torch.arange(tok_future.shape[1], device=tok_future.device)
        tok_future = tok_future + self.future_index_embed(indices)
        return tok_future

    def forward(
        self,
        o_t: torch.Tensor,
        h_t: torch.Tensor,
        y_corrupted_flat: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            o_t: [B, proprio_dim]
            h_t: [B, history_dim]
            y_corrupted_flat: [B, T, K*D]
            t: [B, T, K] corruption time
            r: [B, T, K] interval start time

        Returns:
            a_base: [B, num_actions]
            a_cond: [B, num_actions]
            u: [B, T, K*D] predicted average velocity
            future_features: [B, T, d_model] L2-normalized future token features
        """
        tok_proprio = self.proprio_proj(o_t) + self.proprio_embed
        tok_history = self.history_proj(h_t) + self.history_embed
        tok_future = self._build_future_tokens(y_corrupted_flat, t, r)

        tokens = torch.cat([
            tok_proprio.unsqueeze(1),
            tok_history.unsqueeze(1),
            tok_future,
        ], dim=1)

        out = self.transformer(tokens, attn_mask=self.attn_mask)

        a_base = self.base_head(out[:, 1])
        a_cond = self.cond_head(out[:, 0])
        future_features = out[:, 2:]  # [B, T, d_model]
        u = self.velocity_head(future_features)

        # L2-normalize features for contrastive loss
        future_features_norm = torch.nn.functional.normalize(future_features, dim=-1)

        return a_base, a_cond, u, future_features_norm

    def denoise_only(
        self,
        y_corrupted_flat: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        tok_proprio: torch.Tensor,
        tok_history: torch.Tensor,
    ) -> torch.Tensor:
        """JVP-friendly path: (y, t, r) → u. proprio/history tokens are constants.

        Used for computing du/dt via torch.autograd.functional.jvp.
        """
        tok_future = self._build_future_tokens(y_corrupted_flat, t, r)

        tokens = torch.cat([
            tok_proprio.unsqueeze(1),
            tok_history.unsqueeze(1),
            tok_future,
        ], dim=1)

        out = self.transformer(tokens, attn_mask=self.attn_mask)
        return self.velocity_head(out[:, 2:])
