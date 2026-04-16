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
    """Self-attention + cross-attention decoder for latent flow matching.

    Latent trajectory [prev_z_1, ..., prev_z_L, z_current] forms L+1 self-attention
    tokens. These cross-attend to condition context [o_t_enc, h_enc, frames].

    Training modes:
      - With prob `noise_all_prob`: add independent noise to ALL L+1 tokens,
        denoise all (each token can have different noise level).
      - Otherwise: only z_current is noised, prev_z's are clean.

    Rollout: prev_z's always clean.

    Each token type has its own positional embedding.
    """

    def __init__(self, latent_dim: int, d_model: int,
                 num_heads: int = 4, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.0,
                 activation: nn.Module | None = None,
                 use_proj_norm: bool = False,
                 use_prev_z: bool = False,
                 max_prev_z: int = 8):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.latent_dim = latent_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_prev_z = use_prev_z
        self.max_prev_z = max_prev_z

        # Shared z projection: latent_dim → d_model (for both prev_z and z_current)
        layers = [nn.Linear(latent_dim, d_model), activation, nn.Linear(d_model, d_model)]
        if use_proj_norm:
            layers.append(nn.LayerNorm(d_model))
        self.latent_proj = nn.Sequential(*layers)

        # Positional embeddings for latent sequence [prev_0, prev_1, ..., prev_L-1, current]
        self.z_pos_embed = nn.Parameter(torch.randn(max_prev_z + 1, d_model) * 0.02)

        # Positional embeddings for cross-attn context types
        self.ctx_type_embed = nn.ParameterDict({
            'o_t': nn.Parameter(torch.randn(d_model) * 0.02),
            'h_enc': nn.Parameter(torch.randn(d_model) * 0.02),
            'frame': nn.Parameter(torch.randn(d_model) * 0.02),
        })

        # Time embedding: takes (t, r) pair
        self.time_embed = nn.Sequential(
            nn.Linear(2, d_model), activation, nn.Linear(d_model, d_model),
        )

        # Self-attention + cross-attention layers with AdaLN-Zero
        self.self_attn_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.adaln_layers = nn.ModuleList()
        for _ in range(num_layers):
            # Self-attention over latent sequence
            self.self_attn_layers.append(nn.ModuleDict({
                'q_proj': nn.Linear(d_model, d_model),
                'k_proj': nn.Linear(d_model, d_model),
                'v_proj': nn.Linear(d_model, d_model),
                'out_proj': nn.Linear(d_model, d_model),
                'norm': nn.LayerNorm(d_model, elementwise_affine=False),
            }))
            # Cross-attention to condition context
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
            # AdaLN: self-attn (3: scale, shift, gate) + cross-attn (3) + FFN (3) = 9 * d_model
            adaln = nn.Linear(d_model, 9 * d_model)
            nn.init.zeros_(adaln.weight)
            nn.init.zeros_(adaln.bias)
            self.adaln_layers.append(adaln)

        # Final norm + velocity head (per-token, but we only read the last token)
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaln_final = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.adaln_final.weight)
        nn.init.zeros_(self.adaln_final.bias)
        self.velocity_head = nn.Linear(d_model, latent_dim)

    def _adaln_modulate(self, x, norm, scale, shift):
        return scale * norm(x) + shift

    def _build_z_sequence(self, z_noised, z_prev=None, z_prev_mask=None):
        """Build the latent self-attention sequence [prev_z_1, ..., prev_z_L, z_current].

        Returns:
            z_tokens: [B, L+1, d_model]
            z_mask: [B, L+1] bool (True=valid)
        """
        B = z_noised.shape[0]
        device = z_noised.device

        # Project current z
        tok_current = self.latent_proj(z_noised).unsqueeze(1)  # [B, 1, d]

        if self.use_prev_z and z_prev is not None:
            if z_prev.dim() == 2:
                z_prev = z_prev.unsqueeze(1)
            L = z_prev.shape[1]
            tok_prev = self.latent_proj(z_prev)  # [B, L, d] — shared projection
            z_tokens = torch.cat([tok_prev, tok_current], dim=1)  # [B, L+1, d]
            # Add positional embeddings (last L+1 positions from z_pos_embed)
            pos_start = self.max_prev_z - L
            z_tokens = z_tokens + self.z_pos_embed[pos_start:self.max_prev_z + 1].unsqueeze(0)
            # Mask
            if z_prev_mask is not None:
                current_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
                z_mask = torch.cat([z_prev_mask, current_mask], dim=1)
            else:
                z_mask = torch.ones(B, L + 1, dtype=torch.bool, device=device)
        else:
            z_tokens = tok_current + self.z_pos_embed[self.max_prev_z:self.max_prev_z + 1].unsqueeze(0)
            z_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        return z_tokens, z_mask

    def _add_ctx_type_embeds(self, context):
        """Add per-type positional embeddings to context tokens.

        Context layout: [o_t_enc(1), h_enc(1), frames(F)]
        """
        context = context.clone()
        context[:, 0] = context[:, 0] + self.ctx_type_embed['o_t']
        context[:, 1] = context[:, 1] + self.ctx_type_embed['h_enc']
        context[:, 2:] = context[:, 2:] + self.ctx_type_embed['frame']
        return context

    def forward(self, z_noised, t, context, ctx_mask, r=None,
                z_prev=None, z_prev_mask=None):
        """Training forward: self-attn over [prev_z, z_noised], cross-attn to context."""
        B = z_noised.shape[0]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if r is None:
            r = t
        elif r.dim() == 1:
            r = r.unsqueeze(-1)

        z_tokens, z_mask = self._build_z_sequence(z_noised, z_prev, z_prev_mask)
        context = self._add_ctx_type_embeds(context)
        t_embed = self.time_embed(torch.cat([t, r], dim=-1))  # [B, d]

        N = z_tokens.shape[1]  # L+1
        S = context.shape[1]
        H, hd = self.num_heads, self.head_dim

        for sa_layer, ca_layer, adaln in zip(self.self_attn_layers, self.cross_attn_layers, self.adaln_layers):
            # 9 AdaLN params: self-attn (s, sh, g) + cross-attn (s, sh, g) + FFN (s, sh, g)
            sa_s, sa_sh, sa_g, ca_s, ca_sh, ca_g, ff_s, ff_sh, ff_g = adaln(t_embed).chunk(9, dim=-1)

            # Self-attention over latent sequence
            sa_normed = self._adaln_modulate(z_tokens, sa_layer['norm'], 1 + sa_s.unsqueeze(1), sa_sh.unsqueeze(1))
            q = sa_layer['q_proj'](sa_normed).view(B, N, H, hd).transpose(1, 2)
            k = sa_layer['k_proj'](sa_normed).view(B, N, H, hd).transpose(1, 2)
            v = sa_layer['v_proj'](sa_normed).view(B, N, H, hd).transpose(1, 2)
            sa_mask = z_mask[:, None, None, :].expand(-1, H, N, -1) & z_mask[:, None, :, None].expand(-1, H, -1, N)
            attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
            attn = attn.masked_fill(~sa_mask, float('-inf'))
            attn = attn.softmax(dim=-1).nan_to_num(0.0)  # masked tokens → zero attn
            sa_out = (attn @ v).transpose(1, 2).reshape(B, N, self.d_model)
            sa_out = sa_layer['out_proj'](sa_out)
            z_tokens = z_tokens + sa_g.unsqueeze(1) * sa_out

            # Cross-attention to condition context
            ca_normed = self._adaln_modulate(z_tokens, ca_layer['norm1'], 1 + ca_s.unsqueeze(1), ca_sh.unsqueeze(1))
            q = ca_layer['q_proj'](ca_normed).view(B, N, H, hd).transpose(1, 2)
            k = ca_layer['k_proj'](context).view(B, S, H, hd).transpose(1, 2)
            v = ca_layer['v_proj'](context).view(B, S, H, hd).transpose(1, 2)
            ca_attn_mask = ctx_mask[:, None, None, :].expand(-1, H, N, -1)
            attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
            attn = attn.masked_fill(~ca_attn_mask, float('-inf'))
            attn = attn.softmax(dim=-1).nan_to_num(0.0)  # all-masked → zero attn
            ca_out = (attn @ v).transpose(1, 2).reshape(B, N, self.d_model)
            ca_out = ca_layer['out_proj'](ca_out)
            z_tokens = z_tokens + ca_g.unsqueeze(1) * ca_out

            # FFN
            ffn_in = self._adaln_modulate(z_tokens, ca_layer['norm2'], 1 + ff_s.unsqueeze(1), ff_sh.unsqueeze(1))
            z_tokens = z_tokens + ff_g.unsqueeze(1) * ca_layer['ffn'](ffn_in)

        # Read the LAST token (z_current) for velocity prediction
        tok_out = z_tokens[:, -1]  # [B, d]
        sf, shf = self.adaln_final(t_embed).chunk(2, dim=-1)
        tok_out = self._adaln_modulate(tok_out, self.final_norm, 1 + sf, shf)
        return self.velocity_head(tok_out)

    def build_kv_cache(self, context, ctx_mask, z_prev=None, z_prev_mask=None, z_noised_like=None):
        """Build KV cache for cross-attention context (no prev_z in context anymore)."""
        context = self._add_ctx_type_embeds(context)
        B, S, _ = context.shape
        H, hd = self.num_heads, self.head_dim
        cache = []
        for layer in self.cross_attn_layers:
            k = layer['k_proj'](context).view(B, S, H, hd).transpose(1, 2)
            v = layer['v_proj'](context).view(B, S, H, hd).transpose(1, 2)
            cache.append((k, v))
        # Store z_prev info for forward_cached
        self._cached_z_prev = z_prev
        self._cached_z_prev_mask = z_prev_mask
        return cache, ctx_mask

    def forward_cached(self, z_noised, t, kv_cache, ctx_mask, r=None, z_prev=None):
        """Cached forward for ODE inference. Uses stored z_prev from build_kv_cache."""
        B = z_noised.shape[0]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if r is None:
            r = t
        elif r.dim() == 1:
            r = r.unsqueeze(-1)

        z_prev_use = self._cached_z_prev
        z_prev_mask_use = self._cached_z_prev_mask
        z_tokens, z_mask = self._build_z_sequence(z_noised, z_prev_use, z_prev_mask_use)
        t_embed = self.time_embed(torch.cat([t, r], dim=-1))

        N = z_tokens.shape[1]
        H, hd = self.num_heads, self.head_dim

        for sa_layer, (ca_layer, adaln), (k_cached, v_cached) in zip(
                self.self_attn_layers,
                zip(self.cross_attn_layers, self.adaln_layers),
                kv_cache):
            sa_s, sa_sh, sa_g, ca_s, ca_sh, ca_g, ff_s, ff_sh, ff_g = adaln(t_embed).chunk(9, dim=-1)

            # Self-attention
            sa_normed = self._adaln_modulate(z_tokens, sa_layer['norm'], 1 + sa_s.unsqueeze(1), sa_sh.unsqueeze(1))
            q = sa_layer['q_proj'](sa_normed).view(B, N, H, hd).transpose(1, 2)
            k = sa_layer['k_proj'](sa_normed).view(B, N, H, hd).transpose(1, 2)
            v = sa_layer['v_proj'](sa_normed).view(B, N, H, hd).transpose(1, 2)
            sa_mask = z_mask[:, None, None, :].expand(-1, H, N, -1) & z_mask[:, None, :, None].expand(-1, H, -1, N)
            attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
            attn = attn.masked_fill(~sa_mask, float('-inf'))
            attn = attn.softmax(dim=-1).nan_to_num(0.0)
            sa_out = (attn @ v).transpose(1, 2).reshape(B, N, self.d_model)
            sa_out = sa_layer['out_proj'](sa_out)
            z_tokens = z_tokens + sa_g.unsqueeze(1) * sa_out

            # Cross-attention (cached)
            ca_normed = self._adaln_modulate(z_tokens, ca_layer['norm1'], 1 + ca_s.unsqueeze(1), ca_sh.unsqueeze(1))
            q = ca_layer['q_proj'](ca_normed).view(B, N, H, hd).transpose(1, 2)
            ca_attn_mask = ctx_mask[:, None, None, :].expand(-1, H, N, -1)
            attn = (q @ k_cached.transpose(-2, -1)) * (hd ** -0.5)
            attn = attn.masked_fill(~ca_attn_mask, float('-inf'))
            attn = attn.softmax(dim=-1).nan_to_num(0.0)
            ca_out = (attn @ v_cached).transpose(1, 2).reshape(B, N, self.d_model)
            ca_out = ca_layer['out_proj'](ca_out)
            z_tokens = z_tokens + ca_g.unsqueeze(1) * ca_out

            # FFN
            ffn_in = self._adaln_modulate(z_tokens, ca_layer['norm2'], 1 + ff_s.unsqueeze(1), ff_sh.unsqueeze(1))
            z_tokens = z_tokens + ff_g.unsqueeze(1) * ca_layer['ffn'](ffn_in)

        tok_out = z_tokens[:, -1]
        sf, shf = self.adaln_final(t_embed).chunk(2, dim=-1)
        tok_out = self._adaln_modulate(tok_out, self.final_norm, 1 + sf, shf)
        return self.velocity_head(tok_out)


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
    """Action decoder: concat(z_t_proj, o_t_enc, h_enc) → MLP → action."""

    def __init__(self, latent_dim: int, d_model: int,
                 hidden_dim: int = 512, num_layers: int = 1,
                 hidden_dims: list[int] | None = None,
                 num_actions: int = 29,
                 dropout: float = 0.0, activation: nn.Module | None = None,
                 **kwargs):
        """
        Args:
            latent_dim: Latent z dimension.
            d_model: Dimension of o_t_enc and h_enc tokens.
            hidden_dim: MLP hidden size (used when hidden_dims is None).
            num_layers: Number of MLP hidden layers (used when hidden_dims is None).
            hidden_dims: Explicit list of MLP hidden dims. Overrides hidden_dim/num_layers.
            num_actions: Output action dimension.
            dropout: Dropout rate between MLP layers.
            activation: Activation function.
        """
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.z_proj = nn.Linear(latent_dim, d_model)

        # Build hidden dims list
        if hidden_dims is None:
            hidden_dims = [hidden_dim] * num_layers

        mlp_input_dim = 3 * d_model  # z_proj + o_t_enc + h_enc
        layers = []
        in_dim = mlp_input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.mlp = nn.Sequential(*layers)
        self._d_model = d_model

    def forward(self, z_t, o_t_enc, h_enc=None):
        """
        Args:
            z_t: [B, latent_dim]
            o_t_enc: [B, d_model] encoded proprio
            h_enc: [B, d_model] encoded history (optional)

        Returns:
            action: [B, num_actions]
        """
        tok_z = self.z_proj(z_t)
        if h_enc is not None:
            x = torch.cat([tok_z, o_t_enc, h_enc], dim=-1)  # [B, 3*d]
        else:
            # Pad with zeros for missing h_enc to keep MLP input dim fixed
            x = torch.cat([tok_z, o_t_enc, torch.zeros_like(o_t_enc)], dim=-1)
        return self.mlp(x)
