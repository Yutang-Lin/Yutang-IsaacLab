# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VQ-VAE BFM networks: vector-quantized posterior, codebook prior, and shared frame encoder."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab_rl.rsl_rl.networks.transformer import TransformerEncoder
from isaaclab_rl.rsl_rl.networks.cvae_bfm_networks import BFMFrameEncoder, _build_frame_attn_mask


class VQCodebook(nn.Module):
    """Vector quantization codebook with EMA updates and dead code reset.

    Maintains a codebook of `num_codes` vectors of dimension `latent_dim`.
    Uses EMA updates for codebook vectors during training.
    Dead codes (unused for `dead_code_threshold` steps) are reset to
    randomly sampled encoder outputs.
    """

    def __init__(self, num_codes: int, latent_dim: int, ema_decay: float = 0.99,
                 dead_code_threshold: int = 100):
        super().__init__()
        self.num_codes = num_codes
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        self.embedding = nn.Embedding(num_codes, latent_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_codes, 1.0 / num_codes)

        # EMA state
        self.register_buffer("_ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("_ema_embed_sum", self.embedding.weight.clone())
        # Dead code tracking: steps since each code was last used
        self.register_buffer("_steps_since_used", torch.zeros(num_codes, dtype=torch.long))

    def quantize(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize continuous embeddings to nearest codebook vectors.

        Args:
            z_e: [B, latent_dim] continuous posterior embedding.

        Returns:
            e_q: [B, latent_dim] quantized embedding (straight-through gradient).
            indices: [B] codebook indices.
            commit_loss: scalar commitment loss.
        """
        # Distances: [B, num_codes]
        dists = (z_e.unsqueeze(1) - self.embedding.weight.unsqueeze(0)).pow(2).sum(-1)
        indices = dists.argmin(dim=-1)  # [B]
        e_q = self.embedding(indices)  # [B, latent_dim]

        # Commitment loss
        commit_loss = F.mse_loss(z_e, e_q.detach())

        # EMA codebook update + dead code reset (training only)
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices, self.num_codes).float()  # [B, num_codes]
                usage = onehot.sum(0)  # [num_codes]

                # EMA update
                self._ema_cluster_size = self._ema_cluster_size * self.ema_decay + usage * (1 - self.ema_decay)
                embed_sum = onehot.T @ z_e  # [num_codes, latent_dim]
                self._ema_embed_sum = self._ema_embed_sum * self.ema_decay + embed_sum * (1 - self.ema_decay)

                # Laplace smoothing
                n = self._ema_cluster_size.sum()
                cluster_size = (self._ema_cluster_size + 1e-5) / (n + self.num_codes * 1e-5) * n
                self.embedding.weight.data = self._ema_embed_sum / cluster_size.unsqueeze(1)

                # Dead code tracking and reset
                used = usage > 0
                self._steps_since_used[used] = 0
                self._steps_since_used[~used] += 1

                dead = self._steps_since_used >= self.dead_code_threshold
                n_dead = dead.sum().item()
                if n_dead > 0:
                    # Reset dead codes to randomly sampled encoder outputs
                    rand_idx = torch.randint(0, z_e.shape[0], (n_dead,), device=z_e.device)
                    self.embedding.weight.data[dead] = z_e[rand_idx].detach()
                    self._ema_embed_sum[dead] = z_e[rand_idx].detach()
                    self._ema_cluster_size[dead] = 1.0
                    self._steps_since_used[dead] = 0

        # Straight-through
        e_q = z_e + (e_q - z_e).detach()

        return e_q, indices, commit_loss


class VQBFMPosterior(nn.Module):
    """Cross-attention posterior with vector quantization.

    Same architecture as CVAEBFMPosterior but outputs a continuous embedding
    that gets quantized through the codebook. No mu/logvar — direct embedding.
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
        self.frame_encoder = frame_encoder  # shared

        self.keybody_proj = nn.Linear(keybody_dim, d_model)
        self.keybody_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        # Cross-attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim), activation, nn.Linear(hidden_dim, d_model),
        )

        # Project to continuous embedding (pre-quantization)
        self.embed_head = nn.Linear(d_model, latent_dim)

    def forward(self, r_t, frames_flat, delta_t, frame_mask):
        """
        Args:
            r_t: [B, keybody_dim]
            frames_flat: [B, F, K*D]
            delta_t: [B, F]
            frame_mask: [B, F] bool

        Returns:
            z_e: [B, latent_dim] continuous embedding (before quantization)
        """
        B, F = frames_flat.shape[:2]

        tok_kb = self.keybody_proj(r_t) + self.keybody_embed
        tok_frames = self.frame_encoder(frames_flat, delta_t)

        q = self.q_proj(tok_kb).unsqueeze(1)
        k = self.k_proj(tok_frames)
        v = self.v_proj(tok_frames)

        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, F, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, F, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = frame_mask[:, None, None, :].expand(-1, self.num_heads, 1, -1)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.masked_fill(~attn_mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, 1, self.d_model)
        out = self.out_proj(out).squeeze(1)

        kb_out = self.norm1(tok_kb + out)
        kb_out = self.norm2(kb_out + self.ffn(kb_out))

        return self.embed_head(kb_out)  # [B, latent_dim]


class VQBFMPrior(nn.Module):
    """Transformer prior that predicts codebook index from history + masked frames.

    Token layout: [history(0), frame_0(1), ..., frame_{F-1}(F)]
    1-layer transformer. Outputs logits over codebook entries from history token.
    """

    def __init__(self, h_dim: int, num_codes: int, d_model: int,
                 frame_encoder: BFMFrameEncoder,
                 num_heads: int = 4, hidden_dim: int = 512,
                 dropout: float = 0.0, activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.frame_encoder = frame_encoder  # shared

        self.history_proj = nn.Linear(h_dim, d_model)
        self.history_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        self.transformer = TransformerEncoder(
            d_model=d_model, num_heads=num_heads, hidden_dim=hidden_dim,
            num_layers=1, dropout=dropout, is_causal=False,
            activation=activation, enable_sdpa=False,
        )

        # Predict codebook index
        self.logit_head = nn.Linear(d_model, num_codes)

    def forward(self, h_prior, frames_flat, delta_t, frame_mask):
        """
        Args:
            h_prior: [B, h_dim] deterministic prior encoding from history MLP.
            frames_flat: [B, F, K*D]
            delta_t: [B, F]
            frame_mask: [B, F] bool

        Returns:
            logits: [B, num_codes] unnormalized log-probabilities over codebook.
        """
        B, F = frames_flat.shape[:2]

        tok_h = self.history_proj(h_prior) + self.history_embed
        tok_frames = self.frame_encoder(frames_flat, delta_t)

        tokens = torch.cat([tok_h.unsqueeze(1), tok_frames], dim=1)
        attn_mask = _build_frame_attn_mask(B, F, frame_mask, n_prefix=1, device=h_prior.device)

        out = self.transformer(tokens, attn_mask=attn_mask)
        return self.logit_head(out[:, 0])  # [B, num_codes]
