# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CoDiT-MF: CoDiT-Track with MeanFlow denoising and contrastive features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab_rl.rsl_rl.networks.transformer import TransformerEncoder


class CoDiTMFTransformer(nn.Module):
    """CoDiT Transformer with MeanFlow velocity prediction.

    T+2 token architecture with:
    - Future tokens conditioned on (y_flat, t, r)
    - Velocity head predicts average velocity u(y_t, r, t)
    - denoise_only() for JVP, forward() for full outputs
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
        future_raw_dim = num_keypoints * dims_per_keypoint
        future_token_input_dim = future_raw_dim + 2 * num_keypoints  # K*D + t(K) + r(K)

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
        self.velocity_head = nn.Linear(d_model, future_raw_dim)

        # --- Attention mask ---
        self._build_attention_mask(num_future_frames)

    def _build_attention_mask(self, T: int):
        total = T + 2
        mask = torch.ones(total, total, dtype=torch.bool)
        mask[1, 2:] = False
        self.register_buffer("attn_mask", mask.unsqueeze(0))

    def _build_future_tokens(self, y_corrupted_flat, t, r):
        future_input = torch.cat([y_corrupted_flat, t, r], dim=-1)
        tok_future = self.future_proj(future_input)
        indices = torch.arange(tok_future.shape[1], device=tok_future.device)
        tok_future = tok_future + self.future_index_embed(indices)
        return tok_future

    def _run_transformer(self, tok_proprio, tok_history, tok_future):
        tokens = torch.cat([
            tok_proprio.unsqueeze(1),
            tok_history.unsqueeze(1),
            tok_future,
        ], dim=1)
        return self.transformer(tokens, attn_mask=self.attn_mask)

    def forward(self, o_t, h_t, y_corrupted_flat, t, r):
        """Full forward: returns (a_base, a_cond, u, features_norm)."""
        tok_proprio = self.proprio_proj(o_t) + self.proprio_embed
        tok_history = self.history_proj(h_t) + self.history_embed
        tok_future = self._build_future_tokens(y_corrupted_flat, t, r)
        out = self._run_transformer(tok_proprio, tok_history, tok_future)

        a_base = self.base_head(out[:, 1])
        a_cond = self.cond_head(out[:, 0])
        future_features = out[:, 2:]
        u = self.velocity_head(future_features)
        features_norm = F.normalize(future_features, dim=-1)
        return a_base, a_cond, u, features_norm

    def denoise_only(self, y_corrupted_flat, t, r, tok_proprio, tok_history):
        """JVP-friendly: (y, t, r) → u only. tok_proprio/tok_history are constants."""
        tok_future = self._build_future_tokens(y_corrupted_flat, t, r)
        out = self._run_transformer(tok_proprio, tok_history, tok_future)
        return self.velocity_head(out[:, 2:])
