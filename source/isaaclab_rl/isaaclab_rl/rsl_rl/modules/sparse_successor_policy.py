# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class QueryEncoder(nn.Module):
    """Encodes a single query q = (keypoint_id, target_value, tau) into a d-dimensional vector."""

    def __init__(self, num_keypoints: int, target_dim: int, d_model: int):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.target_dim = target_dim
        self.d_model = d_model

        self.key_emb = nn.Embedding(num_keypoints, d_model)
        self.tau_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.target_mlp = nn.Sequential(
            nn.Linear(target_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, keypoint_id: torch.Tensor, target_value: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keypoint_id: [*] int tensor of keypoint indices
            target_value: [*, target_dim] float tensor
            tau: [*] float tensor (future lag)

        Returns:
            [*, d_model] encoded query vector
        """
        e_k = self.key_emb(keypoint_id)
        e_x = self.target_mlp(target_value)
        e_t = self.tau_mlp(tau.unsqueeze(-1))
        return self.out_mlp(torch.cat([e_k, e_x, e_t], dim=-1))


class ConstraintSetEncoder(nn.Module):
    """Encodes a variable-size set C = [(q_i, w_i)] into z_C by weighted averaging.

    Optionally projects the pooled latent onto a unit sphere or clamps its
    magnitude (``project_mode``) so that ``z_C`` has bounded norm across
    rollouts, expert batches, and relabeled batches. Unbounded ``z_C`` scale
    tends to destabilize both the actor (conditioning input) and the
    discriminator ((snippet, z_C) concat). BFM's native latent always lives
    on a unit sphere — this knob lets us replicate that behaviour at will.
    """

    def __init__(
        self,
        query_encoder: QueryEncoder,
        d_model: int,
        project_mode: str = "none",
        clamp_radius: float = 1.0,
    ):
        super().__init__()
        self.query_encoder = query_encoder
        self.d_model = d_model
        self.post_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        if project_mode not in ("none", "unit_sphere", "clamp_radius"):
            raise ValueError(
                f"Unknown project_mode={project_mode!r}; expected one of "
                f"'none', 'unit_sphere', 'clamp_radius'."
            )
        self.project_mode = project_mode
        self.clamp_radius = float(clamp_radius)

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        if self.project_mode == "none":
            return z
        if self.project_mode == "unit_sphere":
            # Scale to unit L2 norm. Gives ``z_C`` exact parity with BFM's
            # ``project_z``. Multiply by sqrt(d) to keep component variance ~1.
            norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return z / norm * math.sqrt(self.d_model)
        if self.project_mode == "clamp_radius":
            norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            scale = torch.minimum(
                torch.ones_like(norm),
                torch.tensor(self.clamp_radius, device=z.device) / norm,
            )
            return z * scale
        return z  # unreachable

    def forward(
        self,
        keypoint_ids: torch.Tensor,
        target_values: torch.Tensor,
        taus: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            keypoint_ids: [B, N] int
            target_values: [B, N, target_dim] float
            taus: [B, N] float
            weights: [B, N] float (importance weights per query)
            mask: [B, N] float (1 if valid, 0 if padded)

        Returns:
            z_C: [B, d_model]
        """
        B, N = keypoint_ids.shape
        flat_k = keypoint_ids.reshape(B * N)
        flat_x = target_values.reshape(B * N, target_values.shape[-1])
        flat_t = taus.reshape(B * N).float()

        flat_b = self.query_encoder(flat_k, flat_x, flat_t)
        b = flat_b.reshape(B, N, -1)

        w = weights * mask
        denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        z = (b * w.unsqueeze(-1)).sum(dim=1) / denom

        return self._project(self.post_mlp(z))


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: nn.Module, layer_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_d = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation)
        in_d = h
    layers.append(nn.Linear(in_d, output_dim))
    return nn.Sequential(*layers)


class TruncatedNormal(Normal):
    """Normal distribution with sample() truncated to [low, high].

    Mirrors BFM-Zero's TruncatedNormal (``humanoidverse/agents/nn_models.py``):
    draws noise from N(0, scale), optionally clips it to ``[-clip, clip]``,
    adds to the mean, then clamps the result into ``[low, high]`` using a
    straight-through gradient trick so backprop still reaches ``loc``.
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor,
                 low: float = -1.0, high: float = 1.0, eps: float = 1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        clamped = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        # Straight-through: gradient of x w.r.t. loc is preserved.
        return x - x.detach() + clamped.detach()

    def sample_with_noise_clip(self, clip: float | None = None) -> torch.Tensor:
        eps = torch.randn_like(self.loc) * self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        return self._clamp(self.loc + eps)


# ------------------------------------------------------------------
# BFM-style residual building blocks
# ------------------------------------------------------------------
# These mirror the primitives used by BFM-Zero's ``ResidualActor`` and
# ``ResidualForwardMap`` (``humanoidverse/agents/nn_models.py``). For
# ``num_parallel=1`` (our case — we use twin critics as separate modules,
# not ensemble-parallel linear layers) ``linear`` = ``nn.Linear`` and
# ``layernorm`` = ``nn.LayerNorm``.


class _Block(nn.Module):
    """Pre-norm block: LayerNorm → Linear → (optional Mish)."""

    def __init__(self, in_dim: int, out_dim: int, activation: bool):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim)]
        if activation:
            layers.append(nn.Mish())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _ResidualBlock(nn.Module):
    """x + (LayerNorm → Linear → Mish)(x). Preserves dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Mish())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


def _residual_embedding(in_dim: int, hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    """BFM's residual_embedding: Block(in→h) · (n-2) × ResidualBlock(h) · Block(h→h/2).

    The final output dimension is ``hidden_dim // 2``; the common actor/critic
    body then concatenates two embeddings (e.g. obs-only + obs-z) to recover
    ``hidden_dim`` before the residual trunk.
    """
    if hidden_layers < 2:
        raise ValueError("hidden_layers must be >= 2 for residual_embedding")
    seq: list[nn.Module] = [_Block(in_dim, hidden_dim, activation=True)]
    for _ in range(hidden_layers - 2):
        seq.append(_ResidualBlock(hidden_dim))
    seq.append(_Block(hidden_dim, hidden_dim // 2, activation=True))
    return nn.Sequential(*seq)


def _residual_body(hidden_dim: int, n_blocks: int, output_dim: int) -> nn.Sequential:
    """N residual blocks of width ``hidden_dim`` followed by a final Block to ``output_dim`` (no activation)."""
    seq: list[nn.Module] = [_ResidualBlock(hidden_dim) for _ in range(n_blocks)]
    seq.append(_Block(hidden_dim, output_dim, activation=False))
    return nn.Sequential(*seq)


class SuccessorActor(nn.Module):
    """Deterministic mean network with a fixed exploration stddev (BFM-style).

    Follows BFM-Zero's ``Actor``:
      - outputs ``mu = tanh(MLP(obs, z_C))`` in ``[-1, 1]``
      - samples from TruncatedNormal(mu, fixed_std) with noise clipped to
        ``[-stddev_clip, stddev_clip]``
      - the std is NOT learnable; ``act_inference`` returns the squashed mean
        directly
      - consumed by a TD3-style actor update (loss = -Q.mean()); no entropy /
        log-prob term is used by the algorithm
    """

    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        fixed_std: float = 0.2,
        stddev_clip: float = 0.3,
        action_low: float = -1.0,
        action_high: float = 1.0,
        layer_norm: bool = False,
        use_residual: bool = False,
        residual_hidden_dim: int = 1024,
        residual_hidden_layers: int = 1,
        residual_embedding_layers: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.fixed_std = float(fixed_std)
        self.stddev_clip = float(stddev_clip)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.use_residual = bool(use_residual)

        if self.use_residual:
            # BFM Actor: embed_z(obs ⊕ z) → h/2, embed_s(obs) → h/2,
            # policy = N residual blocks(h) + Block(h → action_dim, no act).
            h = int(residual_hidden_dim)
            emb_layers = int(residual_embedding_layers)
            body_blocks = int(residual_hidden_layers)
            self.embed_z = _residual_embedding(obs_dim + z_dim, h, emb_layers)
            self.embed_s = _residual_embedding(obs_dim, h, emb_layers)
            self.body = _residual_body(h, body_blocks, action_dim)
        else:
            act_fn = resolve_nn_activation(activation)
            self.net = _build_mlp(obs_dim + z_dim, hidden_dims, action_dim, act_fn, layer_norm=layer_norm)

    @property
    def action_std(self) -> torch.Tensor:
        return torch.tensor(self.fixed_std)

    def forward(self, obs: torch.Tensor, z_C: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            z_emb = self.embed_z(torch.cat([obs, z_C], dim=-1))  # [B, h/2]
            s_emb = self.embed_s(obs)                            # [B, h/2]
            return torch.tanh(self.body(torch.cat([s_emb, z_emb], dim=-1)))
        return torch.tanh(self.net(torch.cat([obs, z_C], dim=-1)))

    def _make_dist(self, obs: torch.Tensor, z_C: torch.Tensor) -> TruncatedNormal:
        mu = self.forward(obs, z_C)
        std = torch.ones_like(mu) * self.fixed_std
        return TruncatedNormal(mu, std, low=self.action_low, high=self.action_high)

    def sample(self, obs: torch.Tensor, z_C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action with clipped Gaussian noise around tanh(mean).

        Returns ``(action, dummy_log_prob)``; the second value is always zero and
        only kept so callers that still expect a tuple don't break. The
        algorithm must not use it in the loss.
        """
        dist = self._make_dist(obs, z_C)
        action = dist.sample_with_noise_clip(clip=self.stddev_clip)
        return action, torch.zeros(action.shape[0], device=action.device)

    def act_inference(self, obs: torch.Tensor, z_C: torch.Tensor) -> torch.Tensor:
        # Deterministic: just the squashed mean, clamped to the action box.
        mu = self.forward(obs, z_C)
        return torch.clamp(mu, self.action_low + 1e-6, self.action_high - 1e-6)


class SuccessorCritic(nn.Module):
    """Evaluates U(obs, priv, action, q_emb) for a set of queries simultaneously.

    Input: obs_hist [B, obs_dim], priv_state [B, priv_dim], action [B, act_dim], q_emb [B, N, d_model]
    Output: [B, N] scalar successor values per query.
    """

    def __init__(
        self,
        obs_dim: int,
        priv_dim: int,
        action_dim: int,
        query_dim: int,
        hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        layer_norm: bool = False,
        use_residual: bool = False,
        residual_hidden_dim: int = 1024,
        residual_hidden_layers: int = 1,
        residual_embedding_layers: int = 2,
    ):
        super().__init__()
        self.use_residual = bool(use_residual)
        if self.use_residual:
            # BFM ForwardMap pattern, adapted for per-query scoring:
            #   embed_q(q_emb)            → h/2   (analog of embed_z)
            #   embed_sa(obs,priv,action) → h/2   (analog of embed_sa)
            #   body = N residual blocks(h) + Block(h → 1)
            h = int(residual_hidden_dim)
            emb_layers = int(residual_embedding_layers)
            body_blocks = int(residual_hidden_layers)
            self.embed_q = _residual_embedding(query_dim, h, emb_layers)
            self.embed_sa = _residual_embedding(obs_dim + priv_dim + action_dim, h, emb_layers)
            self.body = _residual_body(h, body_blocks, output_dim=1)
        else:
            act_fn = resolve_nn_activation(activation)
            trunk_dim = hidden_dims[-1]
            self.trunk = _build_mlp(
                obs_dim + priv_dim + action_dim, hidden_dims[:-1], trunk_dim, act_fn,
                layer_norm=layer_norm,
            )
            self.head = _build_mlp(trunk_dim + query_dim, [hidden_dims[-1]], 1, act_fn, layer_norm=False)

    def forward(
        self,
        obs: torch.Tensor,
        priv: torch.Tensor,
        action: torch.Tensor,
        q_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs: [B, obs_dim]
            priv: [B, priv_dim]
            action: [B, action_dim]
            q_emb: [B, N, query_dim]

        Returns:
            [B, N] successor values
        """
        B, N, _ = q_emb.shape

        if self.use_residual:
            sa_emb = self.embed_sa(torch.cat([obs, priv, action], dim=-1))   # [B, h/2]
            sa_emb = sa_emb.unsqueeze(1).expand(B, N, -1)                    # [B, N, h/2]
            q_flat = q_emb.reshape(B * N, q_emb.shape[-1])
            q_emb_out = self.embed_q(q_flat).reshape(B, N, -1)               # [B, N, h/2]
            body_in = torch.cat([sa_emb, q_emb_out], dim=-1)                 # [B, N, h]
            body_in = body_in.reshape(B * N, -1)
            return self.body(body_in).reshape(B, N)

        trunk_in = torch.cat([obs, priv, action], dim=-1)
        h = self.trunk(trunk_in)                                             # [B, trunk_dim]
        h_expanded = h.unsqueeze(1).expand(B, N, -1)                         # [B, N, trunk_dim]
        head_in = torch.cat([h_expanded, q_emb], dim=-1)
        return self.head(head_in).squeeze(-1)


class StyleCritic(nn.Module):
    """Q_style / Q_aux(obs, priv, action, z_C) -> scalar.

    When ``use_residual`` is True, mirrors BFM's ``ForwardMap``-style
    architecture: split embedding of ``(obs, priv, action)`` and ``z_C``, then
    a residual body mapped down to a scalar.
    """

    def __init__(
        self,
        obs_dim: int,
        priv_dim: int,
        action_dim: int,
        z_dim: int,
        hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        layer_norm: bool = False,
        use_residual: bool = False,
        residual_hidden_dim: int = 1024,
        residual_hidden_layers: int = 1,
        residual_embedding_layers: int = 2,
    ):
        super().__init__()
        self.use_residual = bool(use_residual)
        if self.use_residual:
            h = int(residual_hidden_dim)
            emb_layers = int(residual_embedding_layers)
            body_blocks = int(residual_hidden_layers)
            self.embed_z = _residual_embedding(z_dim, h, emb_layers)
            self.embed_sa = _residual_embedding(obs_dim + priv_dim + action_dim, h, emb_layers)
            self.body = _residual_body(h, body_blocks, output_dim=1)
        else:
            act_fn = resolve_nn_activation(activation)
            self.net = _build_mlp(
                obs_dim + priv_dim + action_dim + z_dim, hidden_dims, 1, act_fn,
                layer_norm=layer_norm,
            )

    def forward(
        self,
        obs: torch.Tensor,
        priv: torch.Tensor,
        action: torch.Tensor,
        z_C: torch.Tensor,
    ) -> torch.Tensor:
        """Returns [B] scalar style value."""
        if self.use_residual:
            sa = self.embed_sa(torch.cat([obs, priv, action], dim=-1))  # [B, h/2]
            ze = self.embed_z(z_C)                                      # [B, h/2]
            return self.body(torch.cat([sa, ze], dim=-1)).squeeze(-1)
        return self.net(torch.cat([obs, priv, action, z_C], dim=-1)).squeeze(-1)


class RunningScalarNormalizer(nn.Module):
    """Running mean/std normalizer for a scalar stream (BFM's aux_reward_normalizer).

    Follows Welford's online algorithm. ``update(x)`` folds the batch into
    running stats; ``normalize(x)`` returns ``(x - mean) / std`` using the
    current stats without updating them. Designed to be DDP-friendly by keeping
    everything as module buffers.
    """

    def __init__(self, eps: float = 1e-4, clip: float = 10.0):
        super().__init__()
        self.register_buffer("count", torch.zeros(1))
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("M2", torch.zeros(1))
        self.eps = eps
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1).float()
        if x.numel() == 0:
            return
        batch_count = float(x.numel())
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_M2 = batch_var * batch_count

        new_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / new_count)
        new_M2 = self.M2 + batch_M2 + delta.pow(2) * (self.count * batch_count / new_count)

        self.count.copy_(new_count)
        self.mean.copy_(new_mean)
        self.M2.copy_(new_M2)

    def std(self) -> torch.Tensor:
        if self.count.item() < 2:
            return torch.ones_like(self.mean)
        return (self.M2 / self.count).clamp(min=self.eps ** 2).sqrt()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        s = self.std()
        return ((x - self.mean) / s).clamp(-self.clip, self.clip)


class StyleDiscriminator(nn.Module):
    """D(snippet, z_C) -> probability in (0, 1)."""

    def __init__(
        self,
        snippet_dim: int,
        z_dim: int,
        hidden_dims: list[int] = [512, 256],
        activation: str = "elu",
        layer_norm: bool = True,
    ):
        super().__init__()
        act_fn = resolve_nn_activation(activation)
        self.net = _build_mlp(snippet_dim + z_dim, hidden_dims, 1, act_fn, layer_norm=layer_norm)

    def forward(self, snippet: torch.Tensor, z_C: torch.Tensor) -> torch.Tensor:
        """Returns [B] probability values in (0, 1)."""
        logits = self.net(torch.cat([snippet, z_C], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)


class SparseSuccessorPolicy(nn.Module):
    """Top-level module containing all networks for sparse-constraint successor tracking.

    This module owns:
      - QueryEncoder + ConstraintSetEncoder
      - SuccessorActor
      - Twin SuccessorCritics (+ targets)
      - StyleDiscriminator
      - Twin StyleCritics (+ targets)
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        # Query / constraint params
        num_keypoints: int,
        target_dim: int = 3,
        d_model: int = 128,
        max_constraints: int = 16,
        # Actor params
        actor_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        actor_fixed_std: float = 0.2,
        actor_stddev_clip: float = 0.3,
        action_low: float = -1.0,
        action_high: float = 1.0,
        # Critic params
        critic_hidden_dims: list[int] = [512, 256, 128],
        # Discriminator params
        style_feature_dim: int | None = None,
        snippet_dim: int | None = None,
        disc_hidden_dims: list[int] = [512, 256],
        # Snippet config
        snippet_length: int = 8,
        # BFM-style residual architecture. When True the actor, successor
        # critics, style critics, and aux critics are replaced with the
        # Block / ResidualBlock design from ``humanoidverse/agents/nn_models.py``
        # (LayerNorm → Linear → Mish, with skip connections). Hidden dim
        # and block counts are controlled by ``residual_*`` below.
        use_residual_arch: bool = True,
        residual_hidden_dim: int = 1024,
        residual_hidden_layers: int = 1,
        residual_embedding_layers: int = 2,
        # z_C projection — controls the magnitude of the pooled constraint
        # latent handed to the actor and discriminator.
        project_constraint_latent: str = "unit_sphere",
        constraint_latent_clamp_radius: float = 1.0,
        # Misc
        layer_norm: bool = False,
        # Metadata (unused but accepted for compatibility)
        actor_obs_meta: dict | None = None,
        critic_obs_meta: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(f"SparseSuccessorPolicy.__init__ got unexpected arguments: {list(kwargs.keys())}")

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.num_keypoints = num_keypoints
        self.target_dim = target_dim
        self.d_model = d_model
        self.max_constraints = max_constraints
        self.snippet_length = snippet_length

        # -- Query / constraint encoding --
        self.query_encoder = QueryEncoder(num_keypoints, target_dim, d_model)
        self.constraint_encoder = ConstraintSetEncoder(
            self.query_encoder,
            d_model,
            project_mode=project_constraint_latent,
            clamp_radius=constraint_latent_clamp_radius,
        )

        # -- Actor (BFM-style: tanh mean + fixed-std TruncatedNormal) --
        # Common residual-architecture kwargs. When ``use_residual_arch`` is
        # False the sub-networks ignore these and fall back to the plain-MLP
        # path they used before.
        residual_kwargs = dict(
            use_residual=bool(use_residual_arch),
            residual_hidden_dim=int(residual_hidden_dim),
            residual_hidden_layers=int(residual_hidden_layers),
            residual_embedding_layers=int(residual_embedding_layers),
        )

        self.actor = SuccessorActor(
            obs_dim=num_actor_obs,
            z_dim=d_model,
            action_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            fixed_std=actor_fixed_std,
            stddev_clip=actor_stddev_clip,
            action_low=action_low,
            action_high=action_high,
            layer_norm=layer_norm,
            **residual_kwargs,
        )

        # -- Twin successor critics --
        self.successor_critic_1 = SuccessorCritic(
            obs_dim=num_actor_obs,
            priv_dim=num_critic_obs,
            action_dim=num_actions,
            query_dim=d_model,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            **residual_kwargs,
        )
        self.successor_critic_2 = SuccessorCritic(
            obs_dim=num_actor_obs,
            priv_dim=num_critic_obs,
            action_dim=num_actions,
            query_dim=d_model,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            **residual_kwargs,
        )

        # Determine the discriminator input dim. Priority:
        # 1) explicit snippet_dim (absolute size), 2) style_feature_dim * snippet_length,
        # 3) fall back to num_actor_obs * snippet_length (legacy).
        if snippet_dim is not None:
            _snippet_dim = snippet_dim
        elif style_feature_dim is not None:
            _snippet_dim = style_feature_dim * snippet_length
        else:
            _snippet_dim = num_actor_obs * snippet_length
        self.snippet_dim = _snippet_dim

        # -- Style discriminator --
        self.style_discriminator = StyleDiscriminator(
            snippet_dim=_snippet_dim,
            z_dim=d_model,
            hidden_dims=disc_hidden_dims,
            activation=activation,
            layer_norm=True,
        )

        # -- Twin style critics --
        self.style_critic_1 = StyleCritic(
            obs_dim=num_actor_obs,
            priv_dim=num_critic_obs,
            action_dim=num_actions,
            z_dim=d_model,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            **residual_kwargs,
        )
        self.style_critic_2 = StyleCritic(
            obs_dim=num_actor_obs,
            priv_dim=num_critic_obs,
            action_dim=num_actions,
            z_dim=d_model,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            **residual_kwargs,
        )

        # -- Twin auxiliary critics (BFM's aux_critic) --
        # Score the action against env-level shaping rewards (action_smoothness,
        # joint limits, etc.). Structurally identical to StyleCritic; kept
        # separate so actor-loss weighting can be tuned independently.
        self.aux_critic_1 = StyleCritic(
            obs_dim=num_actor_obs,
            priv_dim=num_critic_obs,
            action_dim=num_actions,
            z_dim=d_model,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            **residual_kwargs,
        )
        self.aux_critic_2 = StyleCritic(
            obs_dim=num_actor_obs,
            priv_dim=num_critic_obs,
            action_dim=num_actions,
            z_dim=d_model,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            **residual_kwargs,
        )

        # -- Target networks (no grad) --
        self.successor_critic_1_target = copy.deepcopy(self.successor_critic_1)
        self.successor_critic_2_target = copy.deepcopy(self.successor_critic_2)
        self.style_critic_1_target = copy.deepcopy(self.style_critic_1)
        self.style_critic_2_target = copy.deepcopy(self.style_critic_2)
        self.aux_critic_1_target = copy.deepcopy(self.aux_critic_1)
        self.aux_critic_2_target = copy.deepcopy(self.aux_critic_2)
        for p in self.successor_critic_1_target.parameters():
            p.requires_grad = False
        for p in self.successor_critic_2_target.parameters():
            p.requires_grad = False
        for p in self.style_critic_1_target.parameters():
            p.requires_grad = False
        for p in self.style_critic_2_target.parameters():
            p.requires_grad = False
        for p in self.aux_critic_1_target.parameters():
            p.requires_grad = False
        for p in self.aux_critic_2_target.parameters():
            p.requires_grad = False

        # Running normalizer for the scalar env reward fed to the aux critic.
        # Mirrors BFM's ``_aux_reward_normalizer``.
        self.aux_reward_normalizer = RunningScalarNormalizer()

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_constraint_set(
        self,
        keypoint_ids: torch.Tensor,
        target_values: torch.Tensor,
        taus: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a padded constraint set into z_C [B, d_model]."""
        return self.constraint_encoder(keypoint_ids, target_values, taus, weights, mask)

    def encode_single_queries(
        self,
        keypoint_ids: torch.Tensor,
        target_values: torch.Tensor,
        taus: torch.Tensor,
    ) -> torch.Tensor:
        """Encode individual queries for critic input. Returns [B, N, d_model]."""
        B, N = keypoint_ids.shape
        flat_k = keypoint_ids.reshape(B * N)
        flat_x = target_values.reshape(B * N, target_values.shape[-1])
        flat_t = taus.reshape(B * N).float()
        flat_q = self.query_encoder(flat_k, flat_x, flat_t)
        return flat_q.reshape(B, N, -1)

    # ------------------------------------------------------------------
    # Compatibility interface (called by BaseRunner)
    # ------------------------------------------------------------------

    @property
    def action_mean(self):
        return torch.zeros(1)

    @property
    def action_std(self):
        return self.actor.action_std.mean()

    def reset(self, dones=None):
        pass

    def act(self, observations, **kwargs):
        """Default act using a zero z_C (for compatibility). Real act goes through algorithm."""
        z_C = torch.zeros(observations.shape[0], self.d_model, device=observations.device)
        actions, _ = self.actor.sample(observations, z_C)
        return actions

    def act_inference(self, observations, **kwargs):
        z_C = torch.zeros(observations.shape[0], self.d_model, device=observations.device)
        return self.actor.act_inference(observations, z_C)

    def evaluate(self, critic_observations, **kwargs):
        return torch.zeros(critic_observations.shape[0], 1, device=critic_observations.device)

    def load_state_dict(self, state_dict, strict=True):
        # Drop any legacy learnable-noise-std parameters from older checkpoints;
        # the current actor uses a fixed exploration stddev.
        for legacy_key in ("actor.std", "actor.log_std"):
            if legacy_key in state_dict:
                del state_dict[legacy_key]
                strict = False
        super().load_state_dict(state_dict, strict=strict)
        return True
