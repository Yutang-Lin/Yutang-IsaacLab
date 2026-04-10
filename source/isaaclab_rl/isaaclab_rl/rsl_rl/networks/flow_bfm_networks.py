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
        self.input_norm = None

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
        if self.input_norm is not None:
            tok_prior = self.input_norm(tok_prior)
            tok_proprio = self.input_norm(tok_proprio)
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
    """Cross-attention flow decoder with AdaLN-Zero time conditioning.

    Action token cross-attends to encoder context. Flow time t modulates
    each sub-layer via adaptive layer norm with zero-initialized gates:
        x = x + gate * sublayer(scale * norm(x) + shift)

    At init, gates=0 → each layer is identity → stable training start.
    Supports KV caching for efficient multi-step ODE integration.
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

        # Action token projection (no t — t goes through AdaLN)
        self.action_proj = nn.Sequential(
            nn.Linear(num_actions, d_model),
            activation,
            nn.Linear(d_model, d_model),
        )

        # Time embedding: t → d_model
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            activation,
            nn.Linear(d_model, d_model),
        )

        # Per-layer: cross-attention + FFN + AdaLN-Zero params
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
            # AdaLN-Zero: time_embed → (scale1, shift1, gate1, scale2, shift2, gate2)
            adaln = nn.Linear(d_model, 6 * d_model)
            nn.init.zeros_(adaln.weight)
            nn.init.zeros_(adaln.bias)
            self.adaln_layers.append(adaln)

        # Final norm + velocity head
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaln_final = nn.Linear(d_model, 2 * d_model)  # scale, shift for final
        nn.init.zeros_(self.adaln_final.weight)
        nn.init.zeros_(self.adaln_final.bias)
        self.velocity_head = nn.Linear(d_model, num_actions)

    def _get_adaln_params(self, t_embed, layer_idx):
        """Compute AdaLN-Zero params for a layer from time embedding."""
        params = self.adaln_layers[layer_idx](t_embed)  # [B, 6*d]
        return params.chunk(6, dim=-1)  # scale1, shift1, gate1, scale2, shift2, gate2

    def _adaln_modulate(self, x, norm, scale, shift):
        """Apply adaptive layer norm: scale * norm(x) + shift."""
        return scale * norm(x) + shift

    def _cross_attn_adaln(self, layer, layer_idx, q_tok, k, v, ctx_mask, t_embed):
        """Cross-attention + FFN with AdaLN-Zero conditioning."""
        B, S, _ = k.shape
        H, hd = self.num_heads, self.head_dim
        s1, sh1, g1, s2, sh2, g2 = self._get_adaln_params(t_embed, layer_idx)

        # AdaLN on pre-attn norm
        q_normed = self._adaln_modulate(q_tok, layer['norm1'], 1 + s1, sh1)
        q = layer['q_proj'](q_normed).view(B, 1, H, hd).transpose(1, 2)
        k_ = layer['k_proj'](k).view(B, S, H, hd).transpose(1, 2)
        v_ = layer['v_proj'](v).view(B, S, H, hd).transpose(1, 2)

        attn = (q @ k_.transpose(-2, -1)) * (hd ** -0.5)
        mask = ctx_mask[:, None, None, :].expand(-1, H, 1, -1)
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ v_).transpose(1, 2).reshape(B, self.d_model)
        out = layer['out_proj'](out)

        # Gate1: zero-init → identity at start
        q_tok = q_tok + g1 * out

        # AdaLN on pre-FFN norm
        ffn_in = self._adaln_modulate(q_tok, layer['norm2'], 1 + s2, sh2)
        q_tok = q_tok + g2 * layer['ffn'](ffn_in)

        return q_tok

    def _cross_attn_cached_adaln(self, layer, layer_idx, q_tok, k_cached, v_cached, ctx_mask, t_embed):
        """Cross-attention with KV cache + AdaLN-Zero."""
        B, H, S, hd = k_cached.shape
        s1, sh1, g1, s2, sh2, g2 = self._get_adaln_params(t_embed, layer_idx)

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

    def forward(self, a_t, t, context, ctx_mask):
        """Single denoising step with AdaLN-Zero."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        tok = self.action_proj(a_t)
        t_embed = self.time_embed(t)  # [B, d]

        for i, layer in enumerate(self.cross_attn_layers):
            tok = self._cross_attn_adaln(layer, i, tok, context, context, ctx_mask, t_embed)

        # Final AdaLN + velocity head
        sf, shf = self.adaln_final(t_embed).chunk(2, dim=-1)
        tok = self._adaln_modulate(tok, self.final_norm, 1 + sf, shf)
        return self.velocity_head(tok)

    def build_kv_cache(self, context, ctx_mask):
        """Pre-compute K, V projections for all layers."""
        B, S, _ = context.shape
        H, hd = self.num_heads, self.head_dim
        cache = []
        for layer in self.cross_attn_layers:
            k = layer['k_proj'](context).view(B, S, H, hd).transpose(1, 2)
            v = layer['v_proj'](context).view(B, S, H, hd).transpose(1, 2)
            cache.append((k, v))
        return cache, ctx_mask

    def forward_cached(self, a_t, t, kv_cache, ctx_mask):
        """Denoising step using cached K, V."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        tok = self.action_proj(a_t)
        t_embed = self.time_embed(t)

        for i, (layer, (k_cached, v_cached)) in enumerate(zip(self.cross_attn_layers, kv_cache)):
            tok = self._cross_attn_cached_adaln(layer, i, tok, k_cached, v_cached, ctx_mask, t_embed)

        sf, shf = self.adaln_final(t_embed).chunk(2, dim=-1)
        tok = self._adaln_modulate(tok, self.final_norm, 1 + sf, shf)
        return self.velocity_head(tok)
