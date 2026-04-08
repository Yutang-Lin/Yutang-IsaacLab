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


class _CrossAttnLayer(nn.Module):
    """Single cross-attention + FFN layer. Query attends to context (KV)."""

    def __init__(self, d_model, num_heads, hidden_dim, activation):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, hidden_dim), activation, nn.Linear(hidden_dim, d_model))

    def forward(self, q_tok, kv, kv_mask=None):
        """
        Args:
            q_tok: [B, d] single query token
            kv: [B, S, d] context tokens
            kv_mask: [B, S] bool (True=valid) or None
        """
        B, S, _ = kv.shape
        H, hd = self.num_heads, self.head_dim

        q = self.q_proj(q_tok).view(B, 1, H, hd).transpose(1, 2)
        k = self.k_proj(kv).view(B, S, H, hd).transpose(1, 2)
        v = self.v_proj(kv).view(B, S, H, hd).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        if kv_mask is not None:
            mask = kv_mask[:, None, None, :].expand(-1, H, 1, -1)
            attn = attn.masked_fill(~mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, self.d_model)
        out = self.out_proj(out)

        q_tok = self.norm1(q_tok + out)
        q_tok = self.norm2(q_tok + self.ffn(q_tok))
        return q_tok


class VQBFMPosterior(nn.Module):
    """2-layer cross-attention posterior. Keybody queries, frames are KV.

    Outputs continuous embedding z_e for codebook quantization.
    """

    def __init__(self, keybody_dim: int, latent_dim: int, d_model: int,
                 frame_encoder: BFMFrameEncoder,
                 num_heads: int = 4, hidden_dim: int = 512,
                 num_layers: int = 2, activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.frame_encoder = frame_encoder
        self.keybody_proj = nn.Linear(keybody_dim, d_model)
        self.keybody_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        self.layers = nn.ModuleList([
            _CrossAttnLayer(d_model, num_heads, hidden_dim, activation)
            for _ in range(num_layers)
        ])
        self.embed_head = nn.Linear(d_model, latent_dim)

    def forward(self, r_t, frames_flat, delta_t, frame_mask):
        """
        Returns:
            z_e: [B, latent_dim] continuous embedding (before quantization)
        """
        tok_kb = self.keybody_proj(r_t) + self.keybody_embed
        tok_frames = self.frame_encoder(frames_flat, delta_t)

        q = tok_kb
        for layer in self.layers:
            q = layer(q, tok_frames, frame_mask)

        return self.embed_head(q)


class _SelfCrossAttnLayer(nn.Module):
    """Self-attention among input tokens, then cross-attention to context KV."""

    def __init__(self, d_model, num_heads, hidden_dim, activation):
        super().__init__()
        self.self_attn = _CrossAttnLayer(d_model, num_heads, hidden_dim, activation)
        # Cross-attention to external KV
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.xq_proj = nn.Linear(d_model, d_model)
        self.xk_proj = nn.Linear(d_model, d_model)
        self.xv_proj = nn.Linear(d_model, d_model)
        self.xout_proj = nn.Linear(d_model, d_model)
        self.xnorm = nn.LayerNorm(d_model)
        self.xnorm2 = nn.LayerNorm(d_model)
        self.xffn = nn.Sequential(nn.Linear(d_model, hidden_dim), activation, nn.Linear(hidden_dim, d_model))

    def forward(self, tokens, kv, kv_mask=None):
        """
        Args:
            tokens: [B, N, d] input tokens (self-attend among these)
            kv: [B, S, d] external context (cross-attend to these)
            kv_mask: [B, S] bool or None

        Returns:
            tokens: [B, N, d] updated tokens
        """
        B, N, d = tokens.shape
        S = kv.shape[1]
        H, hd = self.num_heads, self.head_dim

        # Self-attention: each token attends to all input tokens
        # Use _CrossAttnLayer per-token (but we need batched multi-token self-attn)
        # Actually, let's do it directly for N tokens
        # Self-attn Q,K,V all from tokens
        sq = self.self_attn.q_proj(tokens).view(B, N, H, hd).transpose(1, 2)
        sk = self.self_attn.k_proj(tokens).view(B, N, H, hd).transpose(1, 2)
        sv = self.self_attn.v_proj(tokens).view(B, N, H, hd).transpose(1, 2)
        sa = (sq @ sk.transpose(-2, -1)) * (hd ** -0.5)
        sa = sa.softmax(dim=-1)
        s_out = (sa @ sv).transpose(1, 2).reshape(B, N, d)
        s_out = self.self_attn.out_proj(s_out)
        tokens = self.self_attn.norm1(tokens + s_out)
        tokens = self.self_attn.norm2(tokens + self.self_attn.ffn(tokens))

        # Cross-attention: each token attends to external KV
        xq = self.xq_proj(tokens).view(B, N, H, hd).transpose(1, 2)  # [B, H, N, hd]
        xk = self.xk_proj(kv).view(B, S, H, hd).transpose(1, 2)
        xv = self.xv_proj(kv).view(B, S, H, hd).transpose(1, 2)
        xa = (xq @ xk.transpose(-2, -1)) * (hd ** -0.5)
        if kv_mask is not None:
            mask = kv_mask[:, None, None, :].expand(-1, H, N, -1)
            xa = xa.masked_fill(~mask, float('-inf'))
        xa = xa.softmax(dim=-1)
        x_out = (xa @ xv).transpose(1, 2).reshape(B, N, d)
        x_out = self.xout_proj(x_out)
        tokens = self.xnorm(tokens + x_out)
        tokens = self.xnorm2(tokens + self.xffn(tokens))

        return tokens


class VQBFMPrior(nn.Module):
    """Auto-regressive prior with self-attention + cross-attention to frames.

    Input tokens: [prev_e_q, o_t, h_prior] — self-attend among these.
    Cross-attention KV: frame tokens (pad-masked).
    Output: logits from o_t token position.
    """

    def __init__(self, proprio_dim: int, h_dim: int, latent_dim: int,
                 num_codes: int, d_model: int,
                 frame_encoder: BFMFrameEncoder,
                 num_heads: int = 4, hidden_dim: int = 512,
                 num_layers: int = 2, activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.frame_encoder = frame_encoder
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.proprio_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.history_proj = nn.Linear(h_dim, d_model)
        self.history_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.prev_code_proj = nn.Linear(latent_dim, d_model)
        self.prev_code_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        self.layers = nn.ModuleList([
            _SelfCrossAttnLayer(d_model, num_heads, hidden_dim, activation)
            for _ in range(num_layers)
        ])
        self.logit_head = nn.Linear(d_model, num_codes)

    def forward(self, o_t, h_prior, prev_e_q, frames_flat, delta_t, frame_mask):
        """
        Args:
            o_t: [B, proprio_dim]
            h_prior: [B, h_dim]
            prev_e_q: [B, latent_dim] (zero at episode start / training)
            frames_flat: [B, F, K*D]
            delta_t: [B, F]
            frame_mask: [B, F] bool

        Returns:
            logits: [B, num_codes]
        """
        tok_prev = self.prev_code_proj(prev_e_q) + self.prev_code_embed
        tok_o = self.proprio_proj(o_t) + self.proprio_embed
        tok_h = self.history_proj(h_prior) + self.history_embed

        # Input tokens: [prev_e_q, o_t, h_prior]
        tokens = torch.stack([tok_prev, tok_o, tok_h], dim=1)  # [B, 3, d]

        # Cross-attention KV: frame tokens (pad-masked)
        tok_frames = self.frame_encoder(frames_flat, delta_t)  # [B, F, d]

        for layer in self.layers:
            tokens = layer(tokens, tok_frames, frame_mask)

        # Output from o_t token (index 1)
        return self.logit_head(tokens[:, 1])
