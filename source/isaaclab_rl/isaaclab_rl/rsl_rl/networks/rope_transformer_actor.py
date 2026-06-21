# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RoPE causal-transformer actor for BFM (BFM-0.5+).

A from-scratch transformer policy that replaces the residual-MLP actor. Design:

  * Each history timestep's proprio state is encoded into ONE token by a SHARED
    per-frame linear encoder (every frame has the same ``frame_dim`` features).
  * The latent ``z`` is encoded into a single token placed FIRST (position 0).
    The z token does NOT participate in RoPE (it is position-agnostic
    conditioning); the H+1 timestep tokens get rotary position embeddings by
    their time index.
  * A causal transformer runs over ``[z, f_{t-H}, ..., f_t]`` so each timestep
    token attends to z + itself + earlier timesteps only.
  * An action head is applied to EACH of the H+1 timestep tokens, producing
    H+1 actions in parallel (the current-step action is the last one). This
    supports parallel actor training (FB -Q at every position) while sharing one
    forward pass; at inference only the last action is used.

Self-contained (own attention + RoPE); does not reuse the JVP-flash
``networks/transformer.py`` (that path is built for flow/diffusion and has no
RoPE).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Rotary position embedding (RoPE)
# --------------------------------------------------------------------------- #
class RotaryEmbedding(nn.Module):
    """Standard RoPE over the head dimension. ``apply`` rotates q/k by a
    per-token integer position. A position of -1 means "no rotation" (used for
    the z token) — implemented by leaving those tokens untouched via a mask.
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim

    def _cos_sin(self, positions: torch.Tensor):
        # positions: [L] float. -> cos/sin: [L, head_dim]
        freqs = torch.outer(positions.to(self.inv_freq.dtype), self.inv_freq)  # [L, hd/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [L, hd]
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def rotate(self, x: torch.Tensor, positions: torch.Tensor, rope_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, n_heads, L, head_dim]; positions: [L]; rope_mask: [L] bool
        (True = apply rotary, False = leave token unchanged, e.g. the z token).

        NOTE: named ``rotate`` not ``apply`` — ``nn.Module.apply(fn)`` is a
        reserved recursive method (used by weight-init traversal); overriding it
        would break module init/traversal.
        """
        cos, sin = self._cos_sin(positions)            # [L, hd]
        cos = cos.view(1, 1, x.shape[-2], x.shape[-1])
        sin = sin.view(1, 1, x.shape[-2], x.shape[-1])
        rot = x * cos + self._rotate_half(x) * sin
        m = rope_mask.view(1, 1, -1, 1).to(x.dtype)
        return rot * m + x * (1.0 - m)


# --------------------------------------------------------------------------- #
# Causal self-attention with RoPE
# --------------------------------------------------------------------------- #
class RoPECausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                rope_mask: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D]; positions/rope_mask: [L]; attn_mask bool, True=allowed,
        either [L, L] (shared causal) or [B, L, L] (per-sample, e.g. when invalid
        padding frame tokens are masked out as keys). Returns [B, L, D]."""
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                       # each [B, L, h, hd]
        q = q.transpose(1, 2)                             # [B, h, L, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = self.rope.rotate(q, positions, rope_mask)
        k = self.rope.rotate(k, positions, rope_mask)
        # -> additive bias broadcast over heads: [1,1,L,L] (shared) or [B,1,L,L].
        if attn_mask.dim() == 2:
            bias = torch.zeros(L, L, device=x.device, dtype=q.dtype)
            bias = bias.masked_fill(~attn_mask, float("-inf")).view(1, 1, L, L)
        else:
            bias = torch.zeros(B, L, L, device=x.device, dtype=q.dtype)
            bias = bias.masked_fill(~attn_mask, float("-inf")).view(B, 1, L, L)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: RoPE causal attention + MLP."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPECausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model), nn.Mish(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x, positions, rope_mask, attn_mask):
        x = x + self.attn(self.ln1(x), positions, rope_mask, attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


# --------------------------------------------------------------------------- #
# The actor
# --------------------------------------------------------------------------- #
class RoPETransformerActor(nn.Module):
    """Causal-transformer actor ``pi(a | f_{t-H..t}, z)`` over H+1 frame tokens
    plus a leading (RoPE-exempt) z token. Outputs H+1 actions in parallel.

    Forward consumes a per-frame feature tensor ``frames [B, H+1, frame_dim]``
    (token order oldest->current) and ``z [B, z_dim]``. The action distribution
    is a tanh-mean TruncatedNormal-style mean with the caller-supplied std (the
    distribution wrapper is applied by the policy, matching the MLP actor).
    """

    def __init__(
        self,
        frame_dim: int,
        z_dim: int,
        action_dim: int,
        n_layers: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.frame_dim = int(frame_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.frame_enc = nn.Linear(frame_dim, d_model)   # SHARED per-frame encoder
        self.z_enc = nn.Linear(z_dim, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, action_dim)
        # NOTE: head weights are re-initialized orthogonally by the algorithm's
        # ``policy.apply(weight_init)`` after construction, so no manual scaling
        # here (a previous ``*0.01`` was dead — it was overwritten by weight_init).

    def forward(self, frames: torch.Tensor, z: torch.Tensor,
                valid: torch.Tensor | None = None,
                last_only: bool = False) -> torch.Tensor:
        """frames: [B, H+1, frame_dim] (oldest..current); z: [B, z_dim].

        ``valid`` ([B, H+1] bool, optional): marks REAL frame tokens. Invalid
        positions (zero-padded history at/after an episode reset) are excluded as
        ATTENTION KEYS so no token attends to garbage frames — fixing both the
        first-H-steps-of-every-episode regime (history zero-filled at reset) and
        post-reset recovery. The z token (pos 0) is always a valid key, so every
        causal query row keeps at least one key (no all-masked row -> no NaN).

        Returns action MEANS (pre-std) ``tanh(out)``: [B, H+1, action_dim] (one
        per timestep token), or [B, action_dim] for the current step only when
        ``last_only`` (cheaper rollout path)."""
        B, Tp1, _ = frames.shape
        ftok = self.frame_enc(frames)                    # [B, H+1, D]
        ztok = self.z_enc(z).unsqueeze(1)                # [B, 1, D]
        x = torch.cat([ztok, ftok], dim=1)               # [B, H+2, D]  (z first)
        L = Tp1 + 1
        # Positions: z at index 0 is RoPE-exempt; the H+1 frame tokens occupy
        # indices 1..H+1 and get those rotary positions (relative deltas are what
        # matter for frame-frame attention; the absolute offset is consistent
        # between train and rollout).
        positions = torch.arange(L, device=frames.device, dtype=torch.float32)
        rope_mask = torch.ones(L, dtype=torch.bool, device=frames.device)
        rope_mask[0] = False                             # z token: no rotary
        # Causal mask [L,L]: token i attends to j<=i. z (col 0) is visible to all
        # (it's position 0, so causal already lets every token attend to it).
        idx = torch.arange(L, device=frames.device)
        causal = idx.view(1, L) <= idx.view(L, 1)        # [L,L] True=allowed
        if valid is None:
            attn_mask = causal                           # shared [L,L]
        else:
            # Per-sample key validity over the H+2 tokens: z (col 0) always valid,
            # then the H+1 frame validities. A key column j is attendable only if
            # causal AND token j is valid. [B, L, L].
            key_valid = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=frames.device),
                 valid.bool()], dim=1)                   # [B, L]
            attn_mask = causal.view(1, L, L) & key_valid.view(B, 1, L)
        for blk in self.blocks:
            x = blk(x, positions, rope_mask, attn_mask)
        x = self.ln_f(x)
        # Drop the z token; action head on the timestep token(s).
        if last_only:
            a = self.action_head(x[:, -1])               # [B, action_dim]
        else:
            a = self.action_head(x[:, 1:])               # [B, H+1, action_dim]
        return torch.tanh(a)
