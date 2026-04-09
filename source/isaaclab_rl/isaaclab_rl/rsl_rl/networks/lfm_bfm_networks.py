# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LFM-BFM: Latent Flow Matching networks.

Flow matching in L2-normalized latent space instead of action space.
Posterior maps to unit sphere, flow model generates latent codes,
decoder produces actions deterministically.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab_rl.rsl_rl.networks.transformer import TransformerEncoder
from isaaclab_rl.rsl_rl.networks.cvae_bfm_networks import BFMFrameEncoder, _build_frame_attn_mask


class LFMPosterior(nn.Module):
    """Cross-attention posterior: keybody (Q) × encoded [h_prior, o_t] (KV) → latent.

    By cross-attending to encoder outputs instead of raw frames, the posterior's
    latent space stays grounded in the same representations the flow decoder sees
    at rollout, reducing train-rollout distribution shift.
    """

    def __init__(self, keybody_dim: int, latent_dim: int, d_model: int,
                 num_heads: int = 4, hidden_dim: int = 512,
                 num_layers: int = 2, activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.keybody_proj = nn.Linear(keybody_dim, d_model)
        self.keybody_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        # Reuse _CrossAttnLayer pattern
        from isaaclab_rl.rsl_rl.networks.vqvae_bfm_networks import _CrossAttnLayer
        self.layers = nn.ModuleList([
            _CrossAttnLayer(d_model, num_heads, hidden_dim, activation)
            for _ in range(num_layers)
        ])
        self.embed_head = nn.Linear(d_model, latent_dim)

    def forward(self, r_t, context_hp_ot):
        """
        Args:
            r_t: [B, keybody_dim] full body state
            context_hp_ot: [B, 2, d_model] encoded h_prior and o_t from encoder

        Returns:
            z_t: [B, latent_dim] unbounded, regularized via loss to stay in [-1, 1]
        """
        tok_kb = self.keybody_proj(r_t) + self.keybody_embed

        # KV mask: both tokens always valid
        kv_mask = None

        q = tok_kb
        for layer in self.layers:
            q = layer(q, context_hp_ot, kv_mask)

        z_t = self.embed_head(q)
        return z_t


class LatentFlowDecoder(nn.Module):
    """AdaLN-Zero cross-attention decoder for latent flow matching.

    Noised latent z cross-attends to context, predicts velocity in latent space.
    Same architecture as FlowBFMDecoder but operates on latent_dim instead of num_actions.
    """

    def __init__(self, latent_dim: int, d_model: int,
                 num_heads: int = 4, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.0,
                 activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Latent token projection (no t — t goes through AdaLN)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model), activation, nn.Linear(d_model, d_model),
        )

        # Time embedding: takes (t, r) pair
        self.time_embed = nn.Sequential(
            nn.Linear(2, d_model), activation, nn.Linear(d_model, d_model),
        )

        # Cross-attention layers with AdaLN-Zero
        self.cross_attn_layers = nn.ModuleList()
        self.adaln_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_attn_layers.append(nn.ModuleDict({
                'q_proj': nn.Linear(d_model, d_model),
                'k_proj': nn.Linear(d_model, d_model),
                'v_proj': nn.Linear(d_model, d_model),
                'out_proj': nn.Linear(d_model, d_model),
                'norm1': nn.LayerNorm(d_model, elementwise_affine=False),
                'norm2': nn.LayerNorm(d_model, elementwise_affine=False),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, hidden_dim), activation, nn.Linear(hidden_dim, d_model),
                ),
            }))
            adaln = nn.Linear(d_model, 6 * d_model)
            nn.init.zeros_(adaln.weight)
            nn.init.zeros_(adaln.bias)
            self.adaln_layers.append(adaln)

        # Final norm + velocity head
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaln_final = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.adaln_final.weight)
        nn.init.zeros_(self.adaln_final.bias)
        self.velocity_head = nn.Linear(d_model, latent_dim)

    def _adaln_modulate(self, x, norm, scale, shift):
        return scale * norm(x) + shift

    def _cross_attn_adaln(self, layer, adaln, q_tok, context, ctx_mask, t_embed):
        B, S, _ = context.shape
        H, hd = self.num_heads, self.head_dim
        s1, sh1, g1, s2, sh2, g2 = adaln(t_embed).chunk(6, dim=-1)

        q_normed = self._adaln_modulate(q_tok, layer['norm1'], 1 + s1, sh1)
        q = layer['q_proj'](q_normed).view(B, 1, H, hd).transpose(1, 2)
        k = layer['k_proj'](context).view(B, S, H, hd).transpose(1, 2)
        v = layer['v_proj'](context).view(B, S, H, hd).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        mask = ctx_mask[:, None, None, :].expand(-1, H, 1, -1)
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, self.d_model)
        out = layer['out_proj'](out)

        q_tok = q_tok + g1 * out
        ffn_in = self._adaln_modulate(q_tok, layer['norm2'], 1 + s2, sh2)
        q_tok = q_tok + g2 * layer['ffn'](ffn_in)
        return q_tok

    def _cross_attn_cached_adaln(self, layer, adaln, q_tok, k_cached, v_cached, ctx_mask, t_embed):
        B, H, S, hd = k_cached.shape
        s1, sh1, g1, s2, sh2, g2 = adaln(t_embed).chunk(6, dim=-1)

        q_normed = self._adaln_modulate(q_tok, layer['norm1'], 1 + s1, sh1)
        q = layer['q_proj'](q_normed).view(B, 1, H, hd).transpose(1, 2)

        attn = (q @ k_cached.transpose(-2, -1)) * (hd ** -0.5)
        mask = ctx_mask[:, None, None, :].expand(-1, H, 1, -1)
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ v_cached).transpose(1, 2).reshape(B, self.d_model)
        out = layer['out_proj'](out)

        q_tok = q_tok + g1 * out
        ffn_in = self._adaln_modulate(q_tok, layer['norm2'], 1 + s2, sh2)
        q_tok = q_tok + g2 * layer['ffn'](ffn_in)
        return q_tok

    def forward(self, z_noised, t, context, ctx_mask, r=None):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if r is None:
            r = t  # default: r=t (standard flow matching)
        elif r.dim() == 1:
            r = r.unsqueeze(-1)
        tok = self.latent_proj(z_noised)
        t_embed = self.time_embed(torch.cat([t, r], dim=-1))

        for layer, adaln in zip(self.cross_attn_layers, self.adaln_layers):
            tok = self._cross_attn_adaln(layer, adaln, tok, context, ctx_mask, t_embed)

        sf, shf = self.adaln_final(t_embed).chunk(2, dim=-1)
        tok = self._adaln_modulate(tok, self.final_norm, 1 + sf, shf)
        return self.velocity_head(tok)

    def build_kv_cache(self, context, ctx_mask):
        B, S, _ = context.shape
        H, hd = self.num_heads, self.head_dim
        cache = []
        for layer in self.cross_attn_layers:
            k = layer['k_proj'](context).view(B, S, H, hd).transpose(1, 2)
            v = layer['v_proj'](context).view(B, S, H, hd).transpose(1, 2)
            cache.append((k, v))
        return cache, ctx_mask

    def forward_cached(self, z_noised, t, kv_cache, ctx_mask, r=None):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if r is None:
            r = t
        elif r.dim() == 1:
            r = r.unsqueeze(-1)
        tok = self.latent_proj(z_noised)
        t_embed = self.time_embed(torch.cat([t, r], dim=-1))

        for (layer, adaln), (k_cached, v_cached) in zip(
                zip(self.cross_attn_layers, self.adaln_layers), kv_cache):
            tok = self._cross_attn_cached_adaln(layer, adaln, tok, k_cached, v_cached, ctx_mask, t_embed)

        sf, shf = self.adaln_final(t_embed).chunk(2, dim=-1)
        tok = self._adaln_modulate(tok, self.final_norm, 1 + sf, shf)
        return self.velocity_head(tok)


class LFMActionDecoder(nn.Module):
    """1-layer self-attention decoder: encoded context + z_t → action.

    Reuses encoder output tokens directly (no re-encoding).
    Token layout: [z_t(0), h_prior_enc(1), o_t_enc(2), frame_0(3), ..., frame_{F-1}(F+2)]
    Action from o_t_enc position (index 2).
    """

    def __init__(self, latent_dim: int, d_model: int,
                 num_heads: int = 4, hidden_dim: int = 512,
                 num_layers: int = 1, num_actions: int = 29,
                 dropout: float = 0.0, activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.z_proj = nn.Linear(latent_dim, d_model)
        self.z_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        self.transformer = TransformerEncoder(
            d_model=d_model, num_heads=num_heads, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, is_causal=False,
            activation=activation, enable_sdpa=False,
        )
        self.action_head = nn.Linear(d_model, num_actions)

    def forward(self, z_t, context, ctx_mask):
        """
        Args:
            z_t: [B, latent_dim]
            context: [B, 2+F, d_model] encoded [h_prior, o_t, frames]
            ctx_mask: [B, 2+F] bool

        Returns:
            action: [B, num_actions]
        """
        B = z_t.shape[0]
        nf = context.shape[1] - 2

        tok_z = self.z_proj(z_t) + self.z_embed
        tokens = torch.cat([tok_z.unsqueeze(1), context], dim=1)  # [B, 3+F, d]

        # Attention mask: z_t + h_prior + o_t always valid, frames pad-masked
        total = 3 + nf
        attn_mask = torch.ones(B, total, total, dtype=torch.bool, device=z_t.device)
        frame_mask = ctx_mask[:, 2:]
        attn_mask[:, :, 3:] &= frame_mask.unsqueeze(1)
        attn_mask[:, 3:, :] &= frame_mask.unsqueeze(2)
        diag_idx = torch.arange(3, total, device=z_t.device)
        attn_mask[:, diag_idx, diag_idx] = True

        out = self.transformer(tokens, attn_mask=attn_mask)
        return self.action_head(out[:, 2])  # o_t_enc position
