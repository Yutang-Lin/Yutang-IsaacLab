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
    """Cross-attention posterior: keybody (Q) × encoded context (KV) → latent.

    Cross-attends to full encoder output [h_prior, o_t, frames] so the
    posterior sees the same representations as the flow decoder and action decoder.
    """

    def __init__(self, keybody_dim: int, latent_dim: int, d_model: int,
                 num_heads: int = 4, hidden_dim: int = 512,
                 num_layers: int = 2, activation: nn.Module | None = None,
                 use_proj_norm: bool = False):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        if use_proj_norm:
            self.keybody_proj = nn.Sequential(
                nn.Linear(keybody_dim, d_model), nn.LayerNorm(d_model),
            )
        else:
            self.keybody_proj = nn.Linear(keybody_dim, d_model)
        self.keybody_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        # Reuse _CrossAttnLayer pattern
        from isaaclab_rl.rsl_rl.networks.vqvae_bfm_networks import _CrossAttnLayer
        self.layers = nn.ModuleList([
            _CrossAttnLayer(d_model, num_heads, hidden_dim, activation)
            for _ in range(num_layers)
        ])
        if use_proj_norm:
            self.embed_head = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim),
            )
        else:
            self.embed_head = nn.Linear(d_model, latent_dim)

    def forward(self, r_t, context, ctx_mask):
        """
        Args:
            r_t: [B, keybody_dim] full body state
            context: [B, 2+F, d_model] encoded [h_prior, o_t, frames]
            ctx_mask: [B, 2+F] bool (True=valid)

        Returns:
            z_t: [B, latent_dim] unbounded, regularized via loss to stay in [-1, 1]
        """
        tok_kb = self.keybody_proj(r_t) + self.keybody_embed

        q = tok_kb
        for layer in self.layers:
            q = layer(q, context, ctx_mask)

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
                 activation: nn.Module | None = None,
                 use_proj_norm: bool = False,
                 use_prev_z: bool = False):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_prev_z = use_prev_z

        # Latent token projection (no t — t goes through AdaLN)
        layers = [nn.Linear(latent_dim, d_model), activation, nn.Linear(d_model, d_model)]
        if use_proj_norm:
            layers.append(nn.LayerNorm(d_model))
        self.latent_proj = nn.Sequential(*layers)

        # z_prev projection: previous step's z → context token for cross-attention
        if use_prev_z:
            self.prev_z_proj = nn.Sequential(
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

    def _append_prev_z(self, context, ctx_mask, z_prev, z_noised):
        """Append z_prev as an extra context token for cross-attention."""
        if not self.use_prev_z:
            return context, ctx_mask
        if z_prev is None:
            z_prev = torch.zeros_like(z_noised)
        tok_prev = self.prev_z_proj(z_prev).unsqueeze(1)  # [B, 1, d_model]
        context = torch.cat([context, tok_prev], dim=1)    # [B, S+1, d_model]
        prev_mask = torch.ones(ctx_mask.shape[0], 1, dtype=torch.bool, device=ctx_mask.device)
        ctx_mask = torch.cat([ctx_mask, prev_mask], dim=1) # [B, S+1]
        return context, ctx_mask

    def forward(self, z_noised, t, context, ctx_mask, r=None, z_prev=None):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if r is None:
            r = t  # default: r=t (standard flow matching)
        elif r.dim() == 1:
            r = r.unsqueeze(-1)
        context, ctx_mask = self._append_prev_z(context, ctx_mask, z_prev, z_noised)
        tok = self.latent_proj(z_noised)
        t_embed = self.time_embed(torch.cat([t, r], dim=-1))

        for layer, adaln in zip(self.cross_attn_layers, self.adaln_layers):
            tok = self._cross_attn_adaln(layer, adaln, tok, context, ctx_mask, t_embed)

        sf, shf = self.adaln_final(t_embed).chunk(2, dim=-1)
        tok = self._adaln_modulate(tok, self.final_norm, 1 + sf, shf)
        return self.velocity_head(tok)

    def build_kv_cache(self, context, ctx_mask, z_prev=None, z_noised_like=None):
        """Build KV cache. If use_prev_z, appends z_prev token to context first."""
        if self.use_prev_z:
            if z_prev is None:
                z_prev = torch.zeros(context.shape[0], context.shape[2], device=context.device)
                if z_noised_like is not None:
                    z_prev = torch.zeros_like(z_noised_like)
            tok_prev = self.prev_z_proj(z_prev).unsqueeze(1)
            context = torch.cat([context, tok_prev], dim=1)
            prev_mask = torch.ones(ctx_mask.shape[0], 1, dtype=torch.bool, device=ctx_mask.device)
            ctx_mask = torch.cat([ctx_mask, prev_mask], dim=1)
        B, S, _ = context.shape
        H, hd = self.num_heads, self.head_dim
        cache = []
        for layer in self.cross_attn_layers:
            k = layer['k_proj'](context).view(B, S, H, hd).transpose(1, 2)
            v = layer['v_proj'](context).view(B, S, H, hd).transpose(1, 2)
            cache.append((k, v))
        return cache, ctx_mask

    def forward_cached(self, z_noised, t, kv_cache, ctx_mask, r=None, z_prev=None):
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


class LFMReconDecoder(nn.Module):
    """Reconstruction decoder: z_t -> posterior condition (1 future frame)."""

    def __init__(self, latent_dim: int, output_dim: int,
                 hidden_dims: list[int] | None = None,
                 activation: nn.Module | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        if activation is None:
            activation = nn.GELU(approximate="tanh")
        from isaaclab_rl.rsl_rl.networks.cvae_tracker_networks import _build_mlp
        self.mlp = _build_mlp(latent_dim, hidden_dims, output_dim, activation)

    def forward(self, z_t):
        return self.mlp(z_t)


class LFMActionDecoder(nn.Module):
    """Action decoder: [z_t, o_t_enc] → action from z_t position."""

    def __init__(self, latent_dim: int, d_model: int,
                 num_heads: int = 4, hidden_dim: int = 512,
                 num_layers: int = 1, num_actions: int = 29,
                 dropout: float = 0.0, activation: nn.Module | None = None,
                 use_proj_norm: bool = False):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        if use_proj_norm:
            self.z_proj = nn.Sequential(nn.Linear(latent_dim, d_model), nn.LayerNorm(d_model))
        else:
            self.z_proj = nn.Linear(latent_dim, d_model)
        self.z_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        self.transformer = TransformerEncoder(
            d_model=d_model, num_heads=num_heads, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, is_causal=False,
            activation=activation, enable_sdpa=False,
        )
        self.action_head = nn.Linear(d_model, num_actions)

    def forward(self, z_t, o_t_enc):
        """
        Args:
            z_t: [B, latent_dim]
            o_t_enc: [B, d_model] encoded proprio

        Returns:
            action: [B, num_actions]
        """
        tok_z = self.z_proj(z_t) + self.z_embed
        tokens = torch.stack([tok_z, o_t_enc], dim=1)  # [B, 2, d]
        out = self.transformer(tokens)
        return self.action_head(out[:, 0])  # z_t position
