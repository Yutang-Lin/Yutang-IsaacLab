# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CVAE-BFM networks: MaskedMimic-style CVAE.

Shared frame encoder, transformer prior, residual posterior, MLP decoder.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab_rl.rsl_rl.networks.transformer import TransformerEncoder


class BFMFrameEncoder(nn.Module):
    """Shared frame token encoder: projects (keypoint_data, delta_t) → d_model.

    Includes sinusoidal time embedding from delta_t.
    """

    def __init__(self, num_keypoints: int, dims_per_keypoint: int, d_model: int,
                 activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")
        frame_input_dim = num_keypoints * dims_per_keypoint + 1  # K*D + delta_t
        self.proj = nn.Sequential(
            nn.Linear(frame_input_dim, d_model), activation, nn.Linear(d_model, d_model),
        )
        self.output_norm = None
        half_d = d_model // 2
        freq = torch.exp(-torch.arange(half_d, dtype=torch.float32) * (math.log(10000.0) / half_d))
        self.register_buffer("_sin_freq", freq)

    def forward(self, frames_flat: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames_flat: [B, F, K*D]
            delta_t: [B, F]
        Returns:
            tok_frames: [B, F, d_model]
        """
        frame_input = torch.cat([frames_flat, delta_t.unsqueeze(-1)], dim=-1)
        tok = self.proj(frame_input)
        angles = delta_t.unsqueeze(-1) * self._sin_freq
        tok = tok + torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.output_norm is not None:
            tok = self.output_norm(tok)
        return tok


def _build_frame_attn_mask(B: int, F: int, frame_mask: torch.Tensor,
                           n_prefix: int, device: torch.device) -> torch.Tensor:
    """Build attention mask with n_prefix always-active tokens + F frame tokens."""
    total = n_prefix + F
    attn_mask = torch.ones(B, total, total, dtype=torch.bool, device=device)
    attn_mask[:, :, n_prefix:] &= frame_mask.unsqueeze(1)
    attn_mask[:, n_prefix:, :] &= frame_mask.unsqueeze(2)
    diag_idx = torch.arange(n_prefix, total, device=device)
    attn_mask[:, diag_idx, diag_idx] = True
    return attn_mask


class CVAEBFMPriorV2(nn.Module):
    """Transformer prior: processes [h_enc, o_t_enc, frame_tokens] with masking.

    Outputs Gaussian (mu, logvar) in latent space. Frame tokens are masked
    via attention mask so the prior handles variable sparse conditioning.
    """

    def __init__(self, d_model: int, latent_dim: int,
                 num_heads: int = 4, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.0,
                 activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.latent_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.transformer = TransformerEncoder(
            d_model=d_model, num_heads=num_heads, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, is_causal=False,
            activation=activation, enable_sdpa=False,
        )
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

    def forward(self, h_enc, o_t_enc, frame_tokens, frame_mask):
        """
        Args:
            h_enc: [B, d_model] history encoding (L2-normalized)
            o_t_enc: [B, d_model] current proprio encoding
            frame_tokens: [B, F, d_model] frame tokens
            frame_mask: [B, F] bool (True=active)

        Returns:
            mu_prior, logvar_prior: [B, latent_dim]
        """
        B, F, _ = frame_tokens.shape

        # Learnable latent query token
        latent_tok = self.latent_embed.unsqueeze(0).expand(B, -1).unsqueeze(1)  # [B, 1, d]

        tokens = torch.cat([
            latent_tok,              # [B, 1, d] — position 0: latent query
            h_enc.unsqueeze(1),      # [B, 1, d] — position 1: history
            o_t_enc.unsqueeze(1),    # [B, 1, d] — position 2: proprio
            frame_tokens,            # [B, F, d] — positions 3..F+2: frames
        ], dim=1)

        attn_mask = _build_frame_attn_mask(B, F, frame_mask, n_prefix=3, device=h_enc.device)
        out = self.transformer(tokens, attn_mask=attn_mask)

        latent_out = out[:, 0]  # [B, d_model]
        return self.mu_head(latent_out), self.logvar_head(latent_out)


class CVAEBFMPosteriorV2(nn.Module):
    """Residual posterior: keybody cross-attends to frame tokens, outputs
    residual (delta_mu, delta_logvar) on top of prior.

    Final posterior: mu = mu_prior + delta_mu, logvar = delta_logvar.
    """

    def __init__(self, keybody_dim: int, latent_dim: int, d_model: int,
                 num_heads: int = 4, hidden_dim: int = 512,
                 activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.keybody_proj = nn.Linear(keybody_dim, d_model)
        self.keybody_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        # Cross-attention: Q from keybody, KV from frames
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim), activation, nn.Linear(hidden_dim, d_model),
        )

        # Residual outputs
        self.delta_mu_head = nn.Linear(d_model, latent_dim)
        self.delta_logvar_head = nn.Linear(d_model, latent_dim)

    def forward(self, r_t, frame_tokens, frame_mask):
        """
        Args:
            r_t: [B, keybody_dim] full body state (privileged)
            frame_tokens: [B, F, d_model] (already encoded by frame_encoder)
            frame_mask: [B, F] bool

        Returns:
            delta_mu, delta_logvar: [B, latent_dim] residuals to add to prior
        """
        B, F, _ = frame_tokens.shape

        tok_kb = self.keybody_proj(r_t) + self.keybody_embed

        q = self.q_proj(tok_kb).unsqueeze(1).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(frame_tokens).view(B, F, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(frame_tokens).view(B, F, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = frame_mask[:, None, None, :].expand(-1, self.num_heads, 1, -1)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.masked_fill(~attn_mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, 1, self.d_model)
        out = self.out_proj(out).squeeze(1)

        kb_out = self.norm1(tok_kb + out)
        kb_out = self.norm2(kb_out + self.ffn(kb_out))

        return self.delta_mu_head(kb_out), self.delta_logvar_head(kb_out)


class CVAEBFMDecoderV2(nn.Module):
    """MLP action decoder: concat(z_proj, o_t_enc) → action.

    History reaches the decoder only through z (the latent bottleneck),
    forcing the prior to learn a good latent representation.
    """

    def __init__(self, latent_dim: int, d_model: int,
                 hidden_dims: list[int] | None = None,
                 num_actions: int = 29,
                 activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.z_proj = nn.Linear(latent_dim, d_model)

        if hidden_dims is None:
            hidden_dims = [512, 256]

        mlp_input_dim = 2 * d_model  # z_proj + o_t_enc
        layers = []
        in_dim = mlp_input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), activation])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, o_t_enc):
        """
        Args:
            z: [B, latent_dim] sampled latent
            o_t_enc: [B, d_model] current proprio encoding

        Returns:
            action: [B, num_actions]
        """
        tok_z = self.z_proj(z)
        x = torch.cat([tok_z, o_t_enc], dim=-1)
        return self.mlp(x)


# Legacy decoder kept for VQ-VAE BFM and other trackers that use the old interface
class CVAEBFMDecoder(nn.Module):
    """Transformer action decoder (legacy interface for VQ-VAE BFM).

    Token layout: [proprio(0), prior(1), posterior(2), frame_0(3), ..., frame_{F-1}(F+2)]
    """

    def __init__(self, proprio_dim: int, latent_dim: int, max_frames: int,
                 d_model: int, frame_encoder: BFMFrameEncoder,
                 num_heads: int = 4, hidden_dim: int = 512, num_layers: int = 2,
                 num_actions: int = 29, dropout: float = 0.0,
                 activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.frame_encoder = frame_encoder

        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.prior_proj = nn.Linear(latent_dim, d_model)
        self.posterior_proj = nn.Linear(latent_dim, d_model)

        self.proprio_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.prior_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        self.posterior_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        self.transformer = TransformerEncoder(
            d_model=d_model, num_heads=num_heads, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, is_causal=False,
            activation=activation, enable_sdpa=False,
        )
        self.action_head = nn.Linear(d_model, num_actions)

    def forward(self, o_t, h_prior, c_t, frames_flat, delta_t, frame_mask, pre_encoded=False):
        B, F = frames_flat.shape[:2]
        if pre_encoded:
            tok_proprio = o_t + self.proprio_embed
            tok_prior = h_prior + self.prior_embed
        else:
            tok_proprio = self.proprio_proj(o_t) + self.proprio_embed
            tok_prior = self.prior_proj(h_prior) + self.prior_embed
        tok_posterior = self.posterior_proj(c_t) + self.posterior_embed
        tok_frames = self.frame_encoder(frames_flat, delta_t)

        tokens = torch.cat([
            tok_proprio.unsqueeze(1), tok_prior.unsqueeze(1),
            tok_posterior.unsqueeze(1), tok_frames,
        ], dim=1)

        attn_mask = _build_frame_attn_mask(B, F, frame_mask, n_prefix=3, device=o_t.device)
        out = self.transformer(tokens, attn_mask=attn_mask)
        return self.action_head(out[:, 0])


# Legacy alias
CVAEBFMPosterior = CVAEBFMPosteriorV2
