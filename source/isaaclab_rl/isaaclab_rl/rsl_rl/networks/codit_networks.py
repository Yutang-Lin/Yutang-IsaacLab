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
    """Two-level flow-matching corruption for future keypoint conditions.

    For each future frame k (of T total frames), corruption is applied at two levels:

    (A) Keypoint-wise: each of the K keypoints gets an independent time
        t_j ~ Uniform(t_kp_lo, t_kp_hi) for j = 1..K

    (B) Frame-wise: one additional time shared across all keypoints in the frame
        t_frame ~ Uniform(t_fr_lo, t_fr_hi)

    Combined time per keypoint: t = clamp(t_kp + t_frame, max=1).
    Linear interpolation (flow matching):
        y_corrupted[k,j] = (1 - t) * y_clean[k,j] + t * ε,  ε ~ N(0, 1)

    t=0 → clean data, t=1 → pure Gaussian noise.

    The corruption-state vector tau[k] ∈ R^{K+1} records the sampled times:
        tau[k] = [t_1, ..., t_K, t_frame]

    This vector is fed to the transformer so it knows how much corruption was applied.
    No learnable parameters — corruption is purely stochastic.
    """

    def __init__(
        self,
        num_keypoints: int = 6,
        dims_per_keypoint: int = 9,
        t_keypoint_range: tuple[float, float] = (0.0, 0.5),
        t_frame_range: tuple[float, float] = (0.0, 0.3),
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.dims_per_keypoint = dims_per_keypoint
        self.t_kp_lo, self.t_kp_hi = t_keypoint_range
        self.t_fr_lo, self.t_fr_hi = t_frame_range

    def corrupt(self, y_clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply two-level flow-matching corruption to clean future conditions.

        Args:
            y_clean: [B, T, K, D] clean future keypoints.

        Returns:
            y_corrupted: [B, T, K, D] corrupted future keypoints.
            tau: [B, T, K+1] corruption-state vector (sampled times per frame).
        """
        B, T, K, D = y_clean.shape
        device = y_clean.device

        # Sample per-keypoint times: [B, T, K]
        t_kp = torch.empty(B, T, K, device=device).uniform_(self.t_kp_lo, self.t_kp_hi)

        # Sample per-frame times: [B, T, 1]
        t_frame = torch.empty(B, T, 1, device=device).uniform_(self.t_fr_lo, self.t_fr_hi)

        # Combined time per keypoint, clamped to [0, 1]
        t = (t_kp + t_frame).clamp(max=1.0)  # [B, T, K]

        # Flow-matching linear interpolation: y_t = (1 - t) * y_clean + t * ε
        eps = torch.randn(B, T, K, D, device=device)
        y_corrupted = (1.0 - t).unsqueeze(-1) * y_clean + t.unsqueeze(-1) * eps

        # Corruption-state vector: [B, T, K+1]
        tau = torch.cat([t_kp, t_frame], dim=-1)

        return y_corrupted, tau

    def corrupt_rollout(
        self,
        y_clean: torch.Tensor,
        t_combined: torch.Tensor,
        tau_fixed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rollout corruption with per-episode fixed noise levels.

        The combined t per keypoint and the tau vector are precomputed at episode
        reset and held constant. Only the Gaussian ε is fresh each step.

        Args:
            y_clean: [B, T, K, D] clean future keypoints.
            t_combined: [B, T, K] precomputed combined t = clamp(t_kp + t_frame, max=1).
            tau_fixed: [B, T, K+1] precomputed corruption-state vector.

        Returns:
            y_corrupted: [B, T, K, D] corrupted future keypoints.
            tau_fixed: [B, T, K+1] unchanged corruption-state vector.
        """
        B, T, K, D = y_clean.shape
        eps = torch.randn(B, T, K, D, device=y_clean.device)
        y_corrupted = (1.0 - t_combined).unsqueeze(-1) * y_clean + t_combined.unsqueeze(-1) * eps
        return y_corrupted, tau_fixed

    def no_corrupt(self, y_clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return clean conditions with zero corruption state (for pure inference).

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
        # Register as [1, T+2, T+2] — MultiHeadAttention adds the head dim via unsqueeze(-3)
        self.register_buffer("attn_mask", mask.unsqueeze(0))

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
