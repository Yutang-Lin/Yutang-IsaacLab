# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flow-BFM networks: flow matching on action space with cross-attention decoder."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab_rl.rsl_rl.networks.transformer import TransformerEncoder
from isaaclab_rl.rsl_rl.networks.cvae_bfm_networks import BFMFrameEncoder, _build_frame_attn_mask


class FlowBFMEncoder(nn.Module):
    """Encodes [h_prior, proprio, frame_0, ..., frame_{F-1}] into context tokens.

    Self-attention encoder. Masked frames are pad-masked.
    Output context tokens are used as KV for the flow decoder.
    """

    def __init__(self, latent_dim: int, proprio_dim: int, d_model: int,
                 frame_encoder: BFMFrameEncoder,
                 num_heads: int = 4, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.0,
                 activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.frame_encoder = frame_encoder

        self.prior_proj = nn.Linear(latent_dim, d_model)
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.prior_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.proprio_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        self.transformer = TransformerEncoder(
            d_model=d_model, num_heads=num_heads, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, is_causal=False,
            activation=activation, enable_sdpa=False,
        )

    def forward(self, h_prior, o_t, frames_flat, delta_t, frame_mask):
        """
        Args:
            h_prior: [B, latent_dim]
            o_t: [B, proprio_dim]
            frames_flat: [B, F, K*D]
            delta_t: [B, F]
            frame_mask: [B, F] bool

        Returns:
            context: [B, 2+F, d_model] encoded context tokens
            ctx_mask: [B, 2+F] bool (True=valid)
        """
        B, F = frames_flat.shape[:2]

        tok_prior = self.prior_proj(h_prior) + self.prior_embed
        tok_proprio = self.proprio_proj(o_t) + self.proprio_embed
        tok_frames = self.frame_encoder(frames_flat, delta_t)

        tokens = torch.cat([
            tok_prior.unsqueeze(1),
            tok_proprio.unsqueeze(1),
            tok_frames,
        ], dim=1)  # [B, 2+F, d]

        attn_mask = _build_frame_attn_mask(B, F, frame_mask, n_prefix=2, device=h_prior.device)
        context = self.transformer(tokens, attn_mask=attn_mask)

        # Context validity mask: first 2 always valid, frames follow frame_mask
        ctx_mask = torch.ones(B, 2 + F, dtype=torch.bool, device=h_prior.device)
        ctx_mask[:, 2:] = frame_mask

        return context, ctx_mask


class FlowBFMDecoder(nn.Module):
    """Cross-attention flow decoder: noised action token attends to context.

    Single action token with (a_t, t) input cross-attends to encoder context,
    then predicts action-space velocity v(a_t, t).

    Supports KV caching for efficient multi-step ODE integration at rollout.
    """

    def __init__(self, num_actions: int, d_model: int,
                 num_heads: int = 4, hidden_dim: int = 512,
                 num_layers: int = 1, dropout: float = 0.0,
                 activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_layers = num_layers

        # Action token: project (a_t, t) to d_model
        self.action_proj = nn.Sequential(
            nn.Linear(num_actions + 1, d_model),  # +1 for t
            activation,
            nn.Linear(d_model, d_model),
        )

        # Cross-attention layers (one per layer)
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_attn_layers.append(nn.ModuleDict({
                'q_proj': nn.Linear(d_model, d_model),
                'k_proj': nn.Linear(d_model, d_model),
                'v_proj': nn.Linear(d_model, d_model),
                'out_proj': nn.Linear(d_model, d_model),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, hidden_dim), activation, nn.Linear(hidden_dim, d_model),
                ),
            }))

        self.velocity_head = nn.Linear(d_model, num_actions)

    def _cross_attn(self, layer, q_tok, k, v, ctx_mask):
        """Single cross-attention + FFN layer.

        Args:
            layer: nn.ModuleDict with projections
            q_tok: [B, d] action token
            k: [B, S, d] context keys
            v: [B, S, d] context values
            ctx_mask: [B, S] bool

        Returns:
            q_tok: [B, d] updated action token
        """
        B, S, _ = k.shape
        H, hd = self.num_heads, self.head_dim

        q = layer['q_proj'](q_tok).view(B, 1, H, hd).transpose(1, 2)  # [B, H, 1, hd]
        k_ = layer['k_proj'](k).view(B, S, H, hd).transpose(1, 2)  # [B, H, S, hd]
        v_ = layer['v_proj'](v).view(B, S, H, hd).transpose(1, 2)  # [B, H, S, hd]

        attn = (q @ k_.transpose(-2, -1)) * (hd ** -0.5)
        mask = ctx_mask[:, None, None, :].expand(-1, H, 1, -1)
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ v_).transpose(1, 2).reshape(B, self.d_model)
        out = layer['out_proj'](out)

        q_tok = layer['norm1'](q_tok + out)
        q_tok = layer['norm2'](q_tok + layer['ffn'](q_tok))
        return q_tok

    def _cross_attn_cached(self, layer, q_tok, k_cached, v_cached, ctx_mask):
        """Cross-attention using pre-computed K, V caches."""
        B, H, S, hd = k_cached.shape
        q = layer['q_proj'](q_tok).view(B, 1, H, hd).transpose(1, 2)

        attn = (q @ k_cached.transpose(-2, -1)) * (hd ** -0.5)
        mask = ctx_mask[:, None, None, :].expand(-1, H, 1, -1)
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ v_cached).transpose(1, 2).reshape(B, self.d_model)
        out = layer['out_proj'](out)

        q_tok = layer['norm1'](q_tok + out)
        q_tok = layer['norm2'](q_tok + layer['ffn'](q_tok))
        return q_tok

    def forward(self, a_t, t, context, ctx_mask):
        """Single denoising step.

        Args:
            a_t: [B, num_actions] noised action
            t: [B] or [B, 1] flow time
            context: [B, S, d] encoder context
            ctx_mask: [B, S] bool

        Returns:
            v: [B, num_actions] predicted velocity
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        tok = self.action_proj(torch.cat([a_t, t], dim=-1))  # [B, d]

        for layer in self.cross_attn_layers:
            tok = self._cross_attn(layer, tok, context, context, ctx_mask)

        return self.velocity_head(tok)

    def build_kv_cache(self, context, ctx_mask):
        """Pre-compute K, V projections for all layers.

        Returns:
            kv_cache: list of (k_cached, v_cached) per layer
                k_cached: [B, H, S, hd]
                v_cached: [B, H, S, hd]
            ctx_mask: [B, S] (passed through)
        """
        B, S, _ = context.shape
        H, hd = self.num_heads, self.head_dim
        cache = []
        for layer in self.cross_attn_layers:
            k = layer['k_proj'](context).view(B, S, H, hd).transpose(1, 2)
            v = layer['v_proj'](context).view(B, S, H, hd).transpose(1, 2)
            cache.append((k, v))
        return cache, ctx_mask

    def forward_cached(self, a_t, t, kv_cache, ctx_mask):
        """Denoising step using cached K, V (for ODE integration).

        Args:
            a_t: [B, num_actions]
            t: [B] or [B, 1]
            kv_cache: list of (k_cached, v_cached) per layer
            ctx_mask: [B, S] bool

        Returns:
            v: [B, num_actions]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        tok = self.action_proj(torch.cat([a_t, t], dim=-1))

        for layer, (k_cached, v_cached) in zip(self.cross_attn_layers, kv_cache):
            tok = self._cross_attn_cached(layer, tok, k_cached, v_cached, ctx_mask)

        return self.velocity_head(tok)
