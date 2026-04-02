# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: nn.Module) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(activation)
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class CVAEPrior(nn.Module):
    """Prior network: p(z_t | h_t, y_t) -> N(mu_prior, sigma_prior)."""

    def __init__(self, h_dim: int, cond_dim: int, latent_dim: int,
                 hidden_dims: list[int], activation: nn.Module):
        super().__init__()
        self.mlp = _build_mlp(h_dim + cond_dim, hidden_dims, 2 * latent_dim, activation)
        self.latent_dim = latent_dim

    def forward(self, h_t: torch.Tensor, y_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([h_t, y_t], dim=-1)
        out = self.mlp(x)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar


class CVAEPosterior(nn.Module):
    """Posterior (low-rank correction) network: q(c_raw | r_t) -> N(mu_raw, sigma_raw) in R^corr_rank.

    The correction is lifted to full latent space via learned matrix W.
    """

    def __init__(self, keybody_dim: int, corr_rank: int, latent_dim: int,
                 hidden_dims: list[int], activation: nn.Module):
        super().__init__()
        self.mlp = _build_mlp(keybody_dim, hidden_dims, 2 * corr_rank, activation)
        self.lift = nn.Linear(corr_rank, latent_dim, bias=False)
        self.corr_rank = corr_rank
        self.latent_dim = latent_dim

    def forward(self, r_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu_raw, logvar_raw) in low-rank space."""
        out = self.mlp(r_t)
        mu_raw, logvar_raw = out.chunk(2, dim=-1)
        return mu_raw, logvar_raw

    def sample_and_lift(self, mu_raw: torch.Tensor, logvar_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample c_raw from N(mu_raw, sigma_raw) and lift to full latent space.

        Returns:
            c_t: correction in full latent space [batch, latent_dim]
            c_raw: sample in low-rank space [batch, corr_rank] (for KL computation)
        """
        std_raw = torch.exp(0.5 * logvar_raw)
        c_raw = mu_raw + std_raw * torch.randn_like(std_raw)
        c_t = self.lift(c_raw)
        return c_t, c_raw


class CVAEActionDecoder(nn.Module):
    """Transformer-based action decoder: pi(o_t, y_t, z_t) -> a_t.

    Projects o_t, y_t, z_t into separate tokens, runs self-attention across the 3 tokens,
    then reads out actions from an MLP head on the concatenated token outputs.
    """

    def __init__(self, proprio_dim: int, cond_dim: int, latent_dim: int,
                 num_actions: int, hidden_dims: list[int], activation: nn.Module,
                 tf_d_model: int = 256, tf_num_heads: int = 4, tf_num_layers: int = 2,
                 tf_hidden_dim: int = 512, tf_dropout: float = 0.0, tf_activation: nn.Module | None = None):
        super().__init__()
        from .transformer import TransformerEncoder

        if tf_activation is None:
            tf_activation = nn.GELU(approximate="tanh")

        # project each input into a d_model token
        self.proprio_proj = nn.Linear(proprio_dim, tf_d_model)
        self.cond_proj = nn.Linear(cond_dim, tf_d_model)
        self.latent_proj = nn.Linear(latent_dim, tf_d_model)

        # learned token-type embeddings (3 tokens: proprio, condition, latent)
        self.token_embeddings = nn.Parameter(torch.randn(3, tf_d_model))

        # transformer encoder (self-attention across the 3 tokens)
        self.transformer = TransformerEncoder(
            d_model=tf_d_model,
            num_heads=tf_num_heads,
            hidden_dim=tf_hidden_dim,
            num_layers=tf_num_layers,
            dropout=tf_dropout,
            is_causal=False,
            activation=tf_activation,
            enable_sdpa=False,  # standard attention; triton JVP kernel fails with only 3 tokens
        )

        # MLP head: read from all 3 token outputs
        self.head = _build_mlp(3 * tf_d_model, hidden_dims, num_actions, activation)

    def forward(self, o_t: torch.Tensor, y_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        # project to tokens: [batch, d_model] each
        tok_o = self.proprio_proj(o_t) + self.token_embeddings[0]
        tok_y = self.cond_proj(y_t) + self.token_embeddings[1]
        tok_z = self.latent_proj(z_t) + self.token_embeddings[2]

        # stack into sequence: [batch, 3, d_model]
        tokens = torch.stack([tok_o, tok_y, tok_z], dim=1)

        # self-attention across 3 tokens
        out = self.transformer(tokens)  # [batch, 3, d_model]

        # concat all token outputs and decode
        return self.head(out.flatten(start_dim=1))  # [batch, num_actions]
