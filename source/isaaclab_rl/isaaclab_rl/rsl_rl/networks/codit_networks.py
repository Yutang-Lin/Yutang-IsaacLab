# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CoDiT-Track: Condition-Denoising Distillation for Multi-Modal Humanoid Tracking.

Network modules for the CoDiT-Track distillation policy:
  - FutureCorruptor: two-level stochastic corruption of future keypoint conditions
  - CoDiTTransformer: T+2 token transformer with masked attention and dual action heads
"""

from __future__ import annotations

import torch
import torch.nn as nn

from isaaclab_rl.rsl_rl.networks.transformer import TransformerEncoder
from isaaclab_rl.rsl_rl.networks.cvae_tracker_networks import _build_mlp


class FutureCorruptor(nn.Module):
    """Two-level stochastic corruption for future keypoint conditions.

    For each future frame k (of T total frames), corruption is applied at two levels:

    (A) Keypoint-wise: each of the K keypoints gets an independent noise scale
        σ_j ~ Uniform(σ_kp_lo, σ_kp_hi) for j = 1..K

    (B) Frame-wise: one additional noise scale shared across all keypoints in the frame
        σ_frame ~ Uniform(σ_fr_lo, σ_fr_hi)

    The corrupted value for keypoint j in frame k is:
        y_corrupted[k,j] = y_clean[k,j] + ε_kp * σ_j + ε_frame * σ_frame

    The corruption-state vector tau[k] ∈ R^{K+1} records the sampled sigmas:
        tau[k] = [σ_1, ..., σ_K, σ_frame]

    This vector is fed to the transformer so it knows how much corruption was applied.
    No learnable parameters — corruption is purely stochastic.
    """

    def __init__(
        self,
        num_keypoints: int = 6,
        dims_per_keypoint: int = 9,
        sigma_keypoint_range: tuple[float, float] = (0.0, 0.5),
        sigma_frame_range: tuple[float, float] = (0.0, 0.3),
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.dims_per_keypoint = dims_per_keypoint
        self.sigma_kp_lo, self.sigma_kp_hi = sigma_keypoint_range
        self.sigma_fr_lo, self.sigma_fr_hi = sigma_frame_range

    def corrupt(self, y_clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply two-level stochastic corruption to clean future conditions.

        Args:
            y_clean: [B, T, K, D] clean future keypoints.

        Returns:
            y_corrupted: [B, T, K, D] corrupted future keypoints.
            tau: [B, T, K+1] corruption-state vector (sampled sigmas per frame).
        """
        B, T, K, D = y_clean.shape
        device = y_clean.device

        # Sample per-keypoint noise scales: [B, T, K]
        sigma_kp = torch.empty(B, T, K, device=device).uniform_(self.sigma_kp_lo, self.sigma_kp_hi)

        # Sample per-frame noise scales: [B, T, 1]
        sigma_frame = torch.empty(B, T, 1, device=device).uniform_(self.sigma_fr_lo, self.sigma_fr_hi)

        # Keypoint-level noise: [B, T, K, D] * [B, T, K, 1]
        eps_kp = torch.randn(B, T, K, D, device=device) * sigma_kp.unsqueeze(-1)

        # Frame-level noise: [B, T, 1, D] * [B, T, 1, 1]
        eps_frame = torch.randn(B, T, 1, D, device=device) * sigma_frame.unsqueeze(-1)

        y_corrupted = y_clean + eps_kp + eps_frame

        # Corruption-state vector: [B, T, K+1]
        tau = torch.cat([sigma_kp, sigma_frame], dim=-1)

        return y_corrupted, tau

    def no_corrupt(self, y_clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return clean conditions with zero corruption state (for inference / rollout).

        Args:
            y_clean: [B, T, K, D] clean future keypoints.

        Returns:
            y_clean: [B, T, K, D] unchanged.
            tau: [B, T, K+1] all-zeros corruption-state vector.
        """
        B, T, K, _ = y_clean.shape
        tau = torch.zeros(B, T, K + 1, device=y_clean.device)
        return y_clean, tau


class CoDiTTransformer(nn.Module):
    """Condition-Denoising Distillation Transformer.

    Processes T+2 tokens through a transformer encoder:
      - Token 0: proprio (current proprioceptive state o_t)
      - Token 1: history (compressed history encoding h_t)
      - Tokens 2..T+1: future condition tokens (corrupted keypoints + corruption state)

    Attention mask enforces that the history token CANNOT attend to future tokens,
    so it encodes only condition-invariant dynamics/balance/phase information.
    Future tokens CAN attend to history, gaining temporal context.

    Output heads:
      - Base action (from history token): condition-invariant shared control
      - Conditional action (from proprio token): future-condition-dependent residual
      - Future denoising (from each future token): reconstruct clean future keypoints
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
        future_raw_dim = num_keypoints * dims_per_keypoint  # K*D per frame
        future_token_input_dim = future_raw_dim + num_keypoints + 1  # K*D + tau(K+1)
        future_output_dim = future_raw_dim  # denoised K*D per frame

        # --- Token projections ---
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.history_proj = nn.Linear(history_dim, d_model)

        # Shared MLP for future tokens (concat of corrupted keypoints + tau)
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
        # T+2 tokens is small, no need for flash attention
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
        self.base_head = nn.Linear(d_model, num_actions)      # history token → base action
        self.cond_head = nn.Linear(d_model, num_actions)      # proprio token → conditional action
        self.future_head = nn.Linear(d_model, future_output_dim)  # future tokens → denoised prediction

        # --- Attention mask ---
        # history (idx 1) cannot attend to future tokens (idx 2..T+1)
        # everything else is fully connected
        self._build_attention_mask(num_future_frames)

    def _build_attention_mask(self, T: int):
        """Build static attention mask: history blocked from attending to future.

        Token layout: [proprio(0), history(1), future_0(2), ..., future_{T-1}(T+1)]

        mask[i, j] = True means token i CAN attend to token j.
        Only restriction: history (row 1) cannot see future (cols 2..T+1).
        """
        total = T + 2
        mask = torch.ones(total, total, dtype=torch.bool)
        mask[1, 2:] = False  # history cannot attend to future tokens
        # Register as [1, 1, T+2, T+2] for broadcasting over batch and heads
        self.register_buffer("attn_mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        o_t: torch.Tensor,
        h_t: torch.Tensor,
        y_corrupted_flat: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the CoDiT transformer.

        Args:
            o_t: [B, proprio_dim] current proprioceptive observation.
            h_t: [B, history_dim] compressed history encoding.
            y_corrupted_flat: [B, T, K*D] corrupted future keypoints (flattened per frame).
            tau: [B, T, K+1] corruption-state vector per frame.

        Returns:
            a_base: [B, num_actions] base action from history token.
            a_cond: [B, num_actions] conditional action from proprio token.
            y_hat: [B, T, K*D] denoised future predictions per frame.
        """
        B, T = y_corrupted_flat.shape[:2]

        # Build tokens
        tok_proprio = self.proprio_proj(o_t) + self.proprio_embed  # [B, d_model]
        tok_history = self.history_proj(h_t) + self.history_embed  # [B, d_model]

        # Future tokens: concat corrupted keypoints with corruption state, project
        future_input = torch.cat([y_corrupted_flat, tau], dim=-1)  # [B, T, K*D + K+1]
        tok_future = self.future_proj(future_input)  # [B, T, d_model]
        # Add learnable index embeddings per future step
        indices = torch.arange(T, device=o_t.device)
        tok_future = tok_future + self.future_index_embed(indices)  # broadcast over B

        # Assemble sequence: [B, T+2, d_model]
        tokens = torch.cat([
            tok_proprio.unsqueeze(1),   # [B, 1, d_model]
            tok_history.unsqueeze(1),   # [B, 1, d_model]
            tok_future,                 # [B, T, d_model]
        ], dim=1)

        # Apply transformer with attention mask
        out = self.transformer(tokens, attn_mask=self.attn_mask)  # [B, T+2, d_model]

        # Extract outputs from respective token positions
        a_base = self.base_head(out[:, 1])   # history token → base action
        a_cond = self.cond_head(out[:, 0])   # proprio token → conditional action
        y_hat = self.future_head(out[:, 2:])  # future tokens → denoised predictions

        return a_base, a_cond, y_hat
