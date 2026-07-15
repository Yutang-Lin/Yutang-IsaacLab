# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural network layer port of BFM-Zero's FB-CPR-Aux algorithm.

Consolidates the model-layer glue from BFM-Zero's ``nn_models.py``, ``nn_filters.py``,
``normalizers.py``, ``fb/model.py``, ``fb_cpr/model.py`` and ``fb_cpr_aux/model.py``
into a single self-contained module following the Isaac Lab / rsl_rl conventions.

The main entry points are:
  * :class:`FBCprNetworkCfg` -- dataclass config (see BFM's ``train_bfm_zero()`` defaults)
  * :class:`FBCprAuxPolicy`  -- composite module with forward / backward / actor /
    critic / aux-critic / discriminator networks and their target copies
"""

from __future__ import annotations

import copy
import math
import numbers
import typing as tp
from contextlib import contextmanager
from dataclasses import field

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
from torch.distributions.utils import _standard_normal

from isaaclab.utils import configclass


##########################
# Initialization utils
##########################


def parallel_orthogonal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Orthogonal init for parallel ensemble weight tensors ``(n_parallel, rows, cols)``."""
    if tensor.ndimension() == 2:
        return nn.init.orthogonal_(tensor, gain=gain)
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 3 or more dimensions are supported")
    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() // n_parallel // rows
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()
        q, r = torch.linalg.qr(flat_tensor)
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph
        if rows < cols:
            q.t_()
        qs.append(q)

    qs = torch.stack(qs, dim=0)
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor


def weight_init(m: nn.Module) -> None:
    """Orthogonal init for Linear / DenseParallel, reset-on-touch for everything else."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, DenseParallel):
        gain = nn.init.calculate_gain("relu")
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


##########################
# Target-update utils
##########################


def _soft_update_params(net_params: tp.Any, target_net_params: tp.Any, tau: float) -> None:
    torch._foreach_mul_(target_net_params, 1 - tau)
    torch._foreach_add_(target_net_params, net_params, alpha=tau)


def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float) -> None:
    tau = float(min(max(tau, 0.0), 1.0))
    net_params = tuple(x.data for x in net.parameters())
    target_net_params = tuple(x.data for x in target_net.parameters())
    _soft_update_params(net_params, target_net_params, tau)


class eval_mode:
    """Context manager that sets the given modules to ``eval()`` inside the block."""

    def __init__(self, *models: nn.Module) -> None:
        self.models = models
        self.prev_states: list[bool] = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


##########################
# Helper modules
##########################


class DenseParallel(nn.Module):
    """Parallel linear layer for ensemble-in-batch.

    Stores ``n_parallel`` independent weight matrices and applies them via
    ``torch.baddbmm`` so the forward produces ``(n_parallel, batch, out_features)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_parallel: int,
        bias: bool = True,
        device=None,
        dtype=None,
        reset_params: bool = True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        if n_parallel is None or (n_parallel == 1):
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.weight = nn.Parameter(torch.empty((n_parallel, in_features, out_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty((n_parallel, 1, out_features), **factory_kwargs))
            else:
                self.register_parameter("bias", None)
            if self.bias is None:
                raise NotImplementedError
        if reset_params:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.n_parallel is None or (self.n_parallel == 1):
            return F.linear(input, self.weight, self.bias)
        return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, n_parallel={}, bias={}".format(
            self.in_features, self.out_features, self.n_parallel, self.bias is not None
        )


class ParallelLayerNorm(nn.Module):
    """LayerNorm that shares its normalization across a parallel ensemble dim."""

    def __init__(
        self,
        normalized_shape,
        n_parallel: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [normalized_shape]
        assert len(normalized_shape) == 1
        self.n_parallel = n_parallel
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            if n_parallel is None or (n_parallel == 1):
                self.weight = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
            else:
                self.weight = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        norm_input = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias
        return norm_input

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class TruncatedNormal(pyd.Normal):
    """Normal distribution with per-coord action-space clamp and straight-through clamp trick."""

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, low: float = -1.0, high: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.noise_upper_limit = high - self.loc
        self.noise_lower_limit = low - self.loc

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip: float | None = None, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:  # type: ignore[override]
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class SquashedNormal:
    """Gaussian squashed through tanh with correct log_prob (SAC-style).

    ``sample()`` returns ``tanh(u)`` where ``u ~ N(mu, std)`` and caches
    ``u`` so ``log_prob()`` can use the pre-tanh value directly, avoiding
    the numerically unstable ``atanh(tanh(u))`` round-trip that breaks
    when ``tanh(u)`` saturates to ±1 in float32.
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> None:
        self._normal = pyd.Normal(loc, scale, validate_args=False)
        self.eps = eps
        self.loc = loc
        self.scale = scale
        self._cached_u: torch.Tensor | None = None

    @property
    def mean(self) -> torch.Tensor:
        return torch.tanh(self.loc)

    def sample(self, clip: float | None = None, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        u = self._normal.rsample(sample_shape)
        if clip is not None:
            u = self.loc + (u - self.loc).clamp(-clip, clip)
        self._cached_u = u
        return torch.tanh(u)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        u = self._normal.rsample(sample_shape)
        self._cached_u = u
        return torch.tanh(u)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        if self._cached_u is not None and action.shape == self._cached_u.shape:
            u = self._cached_u
        else:
            action_c = action.clamp(-1.0 + self.eps, 1.0 - self.eps)
            u = torch.atanh(action_c)
        log_p = self._normal.log_prob(u)
        log_p = log_p - torch.log(1.0 - torch.tanh(u).pow(2) + self.eps)
        return log_p


class Norm(nn.Module):
    """Projects to the sphere of radius ``sqrt(dim)`` along the last axis."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


class EMA(nn.Module):
    """Exponential moving average reward normalizer."""

    def __init__(
        self,
        tau: float = 0.99,
        epsilon: float = 1e-8,
        shape: tuple[int, ...] = (1,),
        translate: bool = False,
        scale: bool = False,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("mean_square", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("counter", torch.LongTensor([0]))
        self.translate = translate
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.mean()
        sm = x.pow(2).mean()
        self.mean.data = self.tau * self.mean + (1 - self.tau) * m
        self.mean_square.data = self.tau * self.mean_square + (1 - self.tau) * sm
        self.counter += 1  # type: ignore[operator]
        norm = 1 - self.tau**self.counter
        ema_mean = self.mean / norm
        ema_mean_square = self.mean_square / norm
        var = torch.clamp(ema_mean_square - ema_mean**2, min=self.epsilon)

        translate_mean = ema_mean if self.translate else 0
        scale_std = torch.sqrt(var) if self.scale else 1
        return (x - translate_mean) / scale_std

    @property
    def S(self) -> torch.Tensor:
        norm = 1 - self.tau**self.counter
        ema_mean = self.mean / norm
        ema_mean_square = self.mean_square / norm
        return torch.clamp(ema_mean_square - ema_mean**2, self.epsilon)

    @property
    def M(self) -> torch.Tensor:
        norm = 1 - self.tau**self.counter
        return self.mean / norm


##########################
# Creation helpers
##########################


def linear(input_dim: int, output_dim: int, num_parallel: int = 1) -> nn.Module:
    if num_parallel > 1:
        return DenseParallel(input_dim, output_dim, n_parallel=num_parallel)
    return nn.Linear(input_dim, output_dim)


def layernorm(input_dim: int, num_parallel: int = 1) -> nn.Module:
    if num_parallel > 1:
        return ParallelLayerNorm([input_dim], n_parallel=num_parallel)
    return nn.LayerNorm(input_dim)


def simple_embedding(input_dim: int, hidden_dim: int, hidden_layers: int, num_parallel: int = 1) -> nn.Sequential:
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq: list[nn.Module] = [linear(input_dim, hidden_dim, num_parallel), layernorm(hidden_dim, num_parallel), nn.Tanh()]
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
    seq += [linear(hidden_dim, hidden_dim // 2, num_parallel), nn.ReLU()]
    return nn.Sequential(*seq)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, num_parallel: int = 1) -> None:
        super().__init__()
        ln = layernorm(dim, num_parallel)
        lin = linear(dim, dim, num_parallel)
        self.mlp = nn.Sequential(ln, lin, nn.Mish())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class Block(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: bool, num_parallel: int = 1) -> None:
        super().__init__()
        ln = layernorm(input_dim, num_parallel)
        lin = linear(input_dim, output_dim, num_parallel)
        seq = [ln, lin] + ([nn.Mish()] if activation else [])
        self.mlp = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def residual_embedding(input_dim: int, hidden_dim: int, hidden_layers: int, num_parallel: int = 1) -> nn.Sequential:
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq: list[nn.Module] = [Block(input_dim, hidden_dim, True, num_parallel)]
    for _ in range(hidden_layers - 2):
        seq += [ResidualBlock(hidden_dim, num_parallel)]
    seq += [Block(hidden_dim, hidden_dim // 2, True, num_parallel)]
    return nn.Sequential(*seq)


##########################
# Input filters
##########################


class _IdentityFilter(nn.Module):
    """Passthrough filter that simply exposes ``output_space``."""

    def __init__(self, space: gymnasium.spaces.Space) -> None:
        super().__init__()
        self.output_space = space

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return x  # type: ignore[return-value]


class DictInputFilter(nn.Module):
    """Extracts a single key from a ``gymnasium.spaces.Dict`` observation."""

    def __init__(self, space: gymnasium.spaces.Space, key: str) -> None:
        super().__init__()
        assert isinstance(space, gymnasium.spaces.Dict), "space must be a Dict space"
        assert key in space.spaces, f"key {key} not found in space of keys {list(space.spaces.keys())}"
        self.key = key
        self.output_space = space.spaces[key]

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, dict):
            x = x[self.key]
        return x


class DictInputConcatFilter(nn.Module):
    """Concatenates multiple keys from a ``gymnasium.spaces.Dict`` along the last dim."""

    def __init__(self, space: gymnasium.spaces.Space, keys: tp.Sequence[str]) -> None:
        super().__init__()
        assert isinstance(space, gymnasium.spaces.Dict), "space must be a Dict space"
        assert all(k in space.spaces for k in keys), (
            f"keys {list(keys)} not found in space of keys {list(space.spaces.keys())}"
        )
        assert all(isinstance(space[k], gymnasium.spaces.Box) for k in keys), "All keys must be Box spaces"
        assert all(len(space[k].shape) == 1 for k in keys), (
            f"All key spaces must have 1D shape, got {[space[k].shape for k in keys]}"
        )
        first_dtype = space.spaces[keys[0]].dtype
        assert all(space.spaces[k].dtype == first_dtype for k in keys), "All keys must share dtype"
        self.keys = list(keys)
        self.output_space = gymnasium.spaces.Box(
            low=np.concatenate([space.spaces[k].low for k in keys]),
            high=np.concatenate([space.spaces[k].high for k in keys]),
            dtype=first_dtype,
        )

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, dict):
            x = torch.cat([x[k] for k in self.keys], dim=-1)
        return x


def build_input_filter(
    obs_space: gymnasium.spaces.Space, keys: str | tp.Sequence[str] | None
) -> nn.Module:
    """Returns the appropriate input filter for the given obs space and keys.

    * ``keys=None`` -> :class:`_IdentityFilter` (exposes the whole space).
    * ``keys=str`` -> :class:`DictInputFilter` picks a single key.
    * ``keys=list[str]`` of length 1 -> :class:`DictInputFilter` picks that key.
    * ``keys=list[str]`` of length > 1 -> :class:`DictInputConcatFilter` concatenates them.
    """
    if keys is None:
        return _IdentityFilter(obs_space)
    if isinstance(keys, str):
        return DictInputFilter(obs_space, keys)
    keys = list(keys)
    if len(keys) == 1:
        return DictInputFilter(obs_space, keys[0])
    return DictInputConcatFilter(obs_space, keys)


##########################
# Observation normalizers
##########################


class BatchNormNormalizer(nn.Module):
    """Wraps ``nn.BatchNorm1d(affine=False)`` for 1D observation tensors."""

    def __init__(self, obs_space: gymnasium.spaces.Space, momentum: float = 0.01) -> None:
        super().__init__()
        assert len(obs_space.shape) == 1, "BatchNormNormalizer only supports 1D observation spaces"
        self._normalizer = nn.BatchNorm1d(num_features=obs_space.shape[0], affine=False, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalizer(x)


class ObsNormalizer(nn.Module):
    """Holds per-key normalizers for dict observations.

    If ``per_key_momentum_dict`` is a dict mapping ``key -> momentum``, constructs a
    :class:`BatchNormNormalizer` for each listed key. If ``allow_mismatching_keys`` is
    True, keys present in the obs dict but missing from the normalizer dict are simply
    passed through (not normalized).
    """

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        per_key_momentum_dict: dict[str, float],
        allow_mismatching_keys: bool = True,
    ) -> None:
        super().__init__()
        assert isinstance(obs_space, gymnasium.spaces.Dict), "ObsNormalizer expects a Dict obs space"
        if len(per_key_momentum_dict) == 0:
            raise ValueError("ObsNormalizer was initialized with no per-key normalizers.")
        self.allow_mismatching_keys = allow_mismatching_keys
        if not allow_mismatching_keys:
            if set(obs_space.spaces.keys()) != set(per_key_momentum_dict.keys()):
                raise ValueError(
                    f"per_key_momentum_dict keys {set(per_key_momentum_dict.keys())} do not match observation "
                    f"space keys {set(obs_space.spaces.keys())}. Set allow_mismatching_keys=True to ignore."
                )
        self._normalizers = nn.ModuleDict(
            {key: BatchNormNormalizer(obs_space[key], momentum=per_key_momentum_dict[key]) for key in per_key_momentum_dict}
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        # BFM iterates normalizer keys only — obs keys not in the normalizer
        # are DROPPED (not passed through). This matches BFM's behavior where
        # expert obs missing 'history_actor' produces a 3-key output dict.
        for key in self._normalizers.keys():
            if key not in x:
                if self.allow_mismatching_keys:
                    continue
                raise KeyError(f"Key '{key}' not found in the observation, but expected by normalizer.")
            out[key] = self._normalizers[key](x[key])
        return out


##########################
# Core networks
##########################


class BackwardMap(nn.Module):
    """BFM backward map ``B(s_+) -> z`` with optional sphere projection.

    ``model``:
      * ``"simple"`` (default) — the original plain MLP: an input LayerNorm+Tanh
        block, then ``hidden_layers-1`` Linear+ReLU layers, then the z head.
      * ``"residual"`` — a lightweight residual MLP with LayerNorm: input
        projection, then ``hidden_layers-1`` pre-LayerNorm residual blocks
        (LayerNorm -> Linear -> Mish, with a skip), then the z head. Same
        ``hidden_dim``/``hidden_layers`` budget as ``simple`` (still lightweight)
        but with skip connections + per-block LayerNorm for stabler B gradients.
    """

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        z_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 1,
        norm: bool = True,
        input_keys: str | tp.Sequence[str] | None = None,
        model: str = "simple",
    ) -> None:
        super().__init__()
        self.input_filter = build_input_filter(obs_space, input_keys)
        filtered_space = self.input_filter.output_space
        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}."
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        in_dim = filtered_space.shape[0]
        if model == "residual":
            # Lightweight residual MLP with LayerNorm. Input projection into the
            # hidden width, then (hidden_layers-1) pre-LN residual blocks, then
            # the linear z head. ResidualBlock = LayerNorm -> Linear -> Mish + skip.
            seq: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish()]
            for _ in range(hidden_layers - 1):
                seq += [ResidualBlock(hidden_dim)]
            seq += [nn.Linear(hidden_dim, z_dim)]
        elif model == "simple":
            seq = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
            for _ in range(hidden_layers - 1):
                seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            seq += [nn.Linear(hidden_dim, z_dim)]
        else:
            raise ValueError(f"BackwardMap: unknown model '{model}' (want 'simple' or 'residual').")
        if norm:
            seq += [Norm()]
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.input_filter(x)
        return self.net(x)


class ReconstructionHead(nn.Module):
    """Decode ``z`` back into a concat of per-key obs slices (e.g. end-effector
    positions). Used as a regulariser pulling ``B`` to retain spatial info.
    """

    def __init__(
        self,
        z_dim: int,
        targets: tp.Sequence[tuple[str, int, int]],
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        linear: bool = False,
        square_augment: bool = False,
        target_scale: float = 1.0,
        model: str = "simple",
    ) -> None:
        super().__init__()
        self.targets = [(str(k), int(s), int(e)) for (k, s, e) in targets]
        self.square_augment = bool(square_augment)
        # Per-feature divisor applied to the base features BEFORE the square
        # augment, so the (already BatchNorm-unit-variance) targets land in
        # ~[-1,1] and their squares in ~[0,1] — representable by the
        # sphere-bounded B via the linear W. Applied identically in
        # gather_target (recon-loss target) and gather_base_target (z_bar goal
        # features), so z_bar = W^T c_g stays unit-consistent (the scale is a
        # global factor that project_z removes for uniform target_scale).
        self.target_scale = float(target_scale)
        base_dim = sum(e - s for _, s, e in self.targets)
        assert base_dim > 0, "ReconstructionHead needs at least one target slice."
        # With square_augment the target (and output) is [features, features^2].
        self.base_dim = base_dim
        self.output_dim = base_dim * 2 if self.square_augment else base_dim
        if linear:
            # Single linear projection W: z -> R^output_dim, NO BIAS — a pure
            # linear map so the targets must genuinely lie in the SPAN of B(s)
            # (a bias would let W fit the feature mean for free, defeating the
            # span constraint). BFM-0.5 feature-coverage map.
            self.net = nn.Linear(z_dim, self.output_dim, bias=False)
        elif model == "residual":
            # Lightweight residual MLP with LayerNorm — SAME block structure as
            # the residual BackwardMap: input proj -> pre-LN residual blocks ->
            # linear output head. Decodes the full target from z = B(s).
            seq: list[nn.Module] = [nn.Linear(z_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish()]
            for _ in range(max(0, hidden_layers - 1)):
                seq += [ResidualBlock(hidden_dim)]
            seq += [nn.Linear(hidden_dim, self.output_dim)]
            self.net = nn.Sequential(*seq)
        elif model == "simple":
            seq = [nn.Linear(z_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
            for _ in range(max(0, hidden_layers - 1)):
                seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            seq += [nn.Linear(hidden_dim, self.output_dim)]
            self.net = nn.Sequential(*seq)
        else:
            raise ValueError(f"ReconstructionHead: unknown model '{model}' (want 'simple' or 'residual').")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def gather_target(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pull the concat of slices from ``obs`` matching ``self.targets``.

        Expects ``obs`` to be a dict of ``[B, dim_k]`` tensors. Raises
        ``KeyError`` if any target key is missing. When ``square_augment`` is
        set, returns ``[features, features^2]`` so the linear W must also cover
        the second moments (the tracking reward is quadratic in these features).
        """
        parts: list[torch.Tensor] = []
        for key, s, e in self.targets:
            if key not in obs:
                raise KeyError(
                    f"ReconstructionHead: obs dict missing key '{key}'. "
                    f"Available keys: {list(obs.keys())}"
                )
            parts.append(obs[key][:, s:e])
        feats = torch.cat(parts, dim=-1)
        if self.target_scale != 1.0:
            feats = feats / self.target_scale
        if self.square_augment:
            return torch.cat([feats, feats * feats], dim=-1)
        return feats

    def gather_base_target(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Like ``gather_target`` but ALWAYS returns the base ``x`` features
        (``[B, base_dim]``, no square augmentation). Used to build the analytic
        z_bar goal vector ``g`` (the squares are added analytically in c_g)."""
        parts: list[torch.Tensor] = []
        for key, s, e in self.targets:
            if key not in obs:
                raise KeyError(
                    f"ReconstructionHead: obs dict missing key '{key}'. "
                    f"Available keys: {list(obs.keys())}"
                )
            parts.append(obs[key][:, s:e])
        feats = torch.cat(parts, dim=-1)
        if self.target_scale != 1.0:
            feats = feats / self.target_scale
        return feats


class ForwardMap(nn.Module):
    """BFM forward map ``F(s, z, a) -> z`` (also reused as the critic / aux-critic).

    Optional DISCOUNT CONDITIONING (``gamma_embed_dim > 0``): the map becomes
    ``F(s, z, a, gamma)``, letting one network represent successor measures at a
    RANGE of discounts. The conditioning signal is the log effective-horizon
    ``h = -log(1 - gamma)`` (smoother than raw gamma near 1), passed through a
    small MLP ``embed_gamma`` (1 -> gamma_embed_dim -> gamma_embed_dim) and
    concatenated into BOTH embedding-branch inputs. Non-conditioned instances
    (critic / aux / entropy, and F when the feature is off) are byte-identical
    to before.
    """

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        z_dim: int,
        action_dim: int,
        hidden_dim: int = 2048,
        model: str = "residual",
        hidden_layers: int = 6,
        embedding_layers: int = 2,
        num_parallel: int = 2,
        input_keys: str | tp.Sequence[str] | None = None,
        output_dim: int | None = None,
        gamma_embed_dim: int = 0,
    ) -> None:
        super().__init__()
        self.input_filter = build_input_filter(obs_space, input_keys)
        filtered_space = self.input_filter.output_space
        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}."
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        obs_dim = filtered_space.shape[0]

        self.z_dim = z_dim
        self.num_parallel = num_parallel
        self.hidden_dim = hidden_dim
        self.model = model
        self.gamma_embed_dim = int(gamma_embed_dim)

        if model == "residual":
            embed_fn = residual_embedding
        elif model == "simple":
            embed_fn = simple_embedding
        else:
            raise ValueError(f"Unsupported forward_map model {model}")

        # Discount-conditioning embedding: h = -log(1-gamma) -> gamma_embed_dim.
        # A plain (num_parallel-agnostic) MLP; its output is expanded across the
        # parallel ensemble in forward() and concatenated into both branches.
        gdim = self.gamma_embed_dim
        if gdim > 0:
            self.embed_gamma = nn.Sequential(
                nn.Linear(1, gdim), nn.Mish(), nn.Linear(gdim, gdim), nn.Mish(),
            )
        else:
            self.embed_gamma = None

        # BFM quirk: the residual variant of ForwardMap/ResidualForwardMap
        # passes ``cfg.hidden_layers`` (not ``cfg.embedding_layers``) as the
        # embedding depth — see BFM-Zero nn_models.py:484-485. The simple
        # variant honours ``embedding_layers``. Keep parity.
        embed_depth = hidden_layers if model == "residual" else embedding_layers
        # Conditioning widens BOTH branch inputs by gamma_embed_dim (concat).
        self.embed_z = embed_fn(obs_dim + z_dim + gdim, hidden_dim, embed_depth, num_parallel)
        self.embed_sa = embed_fn(obs_dim + action_dim + gdim, hidden_dim, embed_depth, num_parallel)

        out_dim = output_dim if output_dim is not None else z_dim
        if model == "residual":
            seq: list[nn.Module] = [ResidualBlock(hidden_dim, num_parallel) for _ in range(hidden_layers)]
            seq += [Block(hidden_dim, out_dim, False, num_parallel)]
        else:
            seq = []
            for _ in range(hidden_layers):
                seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
            seq += [linear(hidden_dim, out_dim, num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        action: torch.Tensor,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs = self.input_filter(obs)
        g_emb = None
        if self.embed_gamma is not None:
            if gamma is None:
                raise ValueError("ForwardMap is gamma-conditioned but no gamma was passed.")
            # gamma: [B] or [B,1] in (0,1). h = -log(1-gamma), then MLP-embed.
            g = gamma.reshape(-1, 1).to(obs.dtype)
            h = -torch.log1p(-g.clamp(max=1 - 1e-6))
            g_emb = self.embed_gamma(h)                      # [B, gdim]
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
            if g_emb is not None:
                g_emb = g_emb.expand(self.num_parallel, -1, -1)
        z_in = torch.cat([obs, z], dim=-1) if g_emb is None else torch.cat([obs, z, g_emb], dim=-1)
        sa_in = torch.cat([obs, action], dim=-1) if g_emb is None else torch.cat([obs, action, g_emb], dim=-1)
        z_embedding = self.embed_z(z_in)
        sa_embedding = self.embed_sa(sa_in)
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class Actor(nn.Module):
    """BFM actor ``pi(a | s, z)`` returning a :class:`TruncatedNormal`."""

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        z_dim: int,
        action_dim: int,
        hidden_dim: int = 2048,
        model: str = "residual",
        hidden_layers: int = 6,
        embedding_layers: int = 2,
        input_keys: str | tp.Sequence[str] | None = None,
        learned_std: bool = False,
        min_std: float = 0.01,
        max_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_filter = build_input_filter(obs_space, input_keys)
        filtered_space = self.input_filter.output_space
        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}."
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        obs_dim = filtered_space.shape[0]
        self.model = model
        self.learned_std = learned_std
        self._log_min_std = math.log(min_std)
        self._log_max_std = math.log(max_std)

        if model == "residual":
            embed_fn = residual_embedding
        elif model == "simple":
            embed_fn = simple_embedding
        else:
            raise ValueError(f"Unsupported actor model {model}")

        # Actor has no parallel ensemble dim.
        self.embed_z = embed_fn(obs_dim + z_dim, hidden_dim, embedding_layers, 1)
        self.embed_s = embed_fn(obs_dim, hidden_dim, embedding_layers, 1)

        out_dim = action_dim * 2 if learned_std else action_dim
        if model == "residual":
            seq: list[nn.Module] = [ResidualBlock(hidden_dim) for _ in range(hidden_layers)]
            seq += [Block(hidden_dim, out_dim, False)]
        else:
            seq = []
            for _ in range(hidden_layers):
                seq += [linear(hidden_dim, hidden_dim), nn.ReLU()]
            seq += [linear(hidden_dim, out_dim)]
        self.policy = nn.Sequential(*seq)
        self._action_dim = action_dim

        # For SquashedNormal (learned_std), scale down the last layer so
        # mu_raw starts near zero (tanh(0) = 0, well inside [-1,1]) and
        # log_std_raw starts near the midpoint of the clamped range.
        # Without this, orthogonal init produces |mu_raw| ~ 1-3 at init,
        # pushing tanh to saturation and making log_prob ≈ -∞.
        if learned_std:
            last_module = seq[-1]
            with torch.no_grad():
                if isinstance(last_module, Block):
                    # Block.mlp = Sequential(LayerNorm, Linear [, Mish])
                    lin = last_module.mlp[1]
                    lin.weight.data.mul_(0.01)
                    if hasattr(lin, "bias") and lin.bias is not None:
                        lin.bias.data.zero_()
                elif isinstance(last_module, (nn.Linear, DenseParallel)):
                    last_module.weight.data.mul_(0.01)
                    if hasattr(last_module, "bias") and last_module.bias is not None:
                        last_module.bias.data.zero_()

    def forward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        std: float | torch.Tensor,
    ) -> TruncatedNormal:
        obs = self.input_filter(obs)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        s_embedding = self.embed_s(obs)
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        out = self.policy(embedding)
        if self.learned_std:
            mu_raw, log_std_raw = out.split(self._action_dim, dim=-1)
            std_tensor = log_std_raw.clamp(self._log_min_std, self._log_max_std).exp()
            # SquashedNormal: sample = tanh(u), u ~ N(mu_raw, std).
            # mu_raw is NOT tanh'd here — tanh is applied inside the distribution.
            return SquashedNormal(mu_raw, std_tensor)
        else:
            mu = torch.tanh(out)
            if torch.is_tensor(std):
                # Per-env std: accept [N] or [N,1], broadcast over action dims.
                s = std.to(mu.device, mu.dtype)
                if s.dim() == 1:
                    s = s.unsqueeze(-1)
                std_tensor = torch.ones_like(mu) * s
            else:
                std_tensor = torch.ones_like(mu) * std
            return TruncatedNormal(mu, std_tensor)


class TransformerActorWrapper(nn.Module):
    """Adapts :class:`RoPETransformerActor` to the policy's actor interface.

    Consumes the RAW (un-normalized) obs dict and owns a SINGLE shared frame
    normalizer (BatchNorm1d over the ``frame_dim`` per-frame vector) applied
    identically to every one of the H+1 frame tokens — so the shared per-frame
    encoder sees a consistent scale whether a frame is "current" or "past", and
    whether it came from the training window or the rollout history.

    Per-frame token = ``[state | last_action]`` in the canonical TRAINING order
    ``[dof_pos_dev(29), dof_vel(29), gravity(3), root_ang_vel(3), action(29)]``
    (= 93). The env's ``history_actor`` packs the same fields per past frame in
    ALPHABETICAL group order ``[actions(29), base_ang_vel(3), dof_pos(29),
    dof_vel(29), projected_gravity(3)]``; ``_assemble_frames`` reorders them to
    the training order so train/rollout frames are byte-identical.
    """

    # canonical training frame layout (dim ranges within the 93-d frame)
    _N_DOFP, _N_DOFV, _N_GRAV, _N_ANGV, _N_ACT = 29, 29, 3, 3, 29

    def __init__(self, obs_space, z_dim, action_dim, frame_dim=93,
                 history_len=9, d_model=512, n_layers=6, n_heads=8, mlp_ratio=4):
        super().__init__()
        from isaaclab_rl.rsl_rl.networks.rope_transformer_actor import RoPETransformerActor
        self.frame_dim = int(frame_dim)
        self.history_len = int(history_len)
        self.action_dim = int(action_dim)
        self.frame_norm = nn.BatchNorm1d(self.frame_dim, affine=False, momentum=0.01)
        self.net = RoPETransformerActor(
            frame_dim=self.frame_dim, z_dim=z_dim, action_dim=action_dim,
            n_layers=n_layers, d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio,
        )

    # -- frame assembly ----------------------------------------------------
    def _current_frame(self, obs: dict) -> torch.Tensor:
        """[B, 93] current-step frame = [state(64) | last_action(29)]."""
        return torch.cat([obs["state"], obs["last_action"]], dim=-1)

    def _history_frames(self, obs: dict) -> torch.Tensor:
        """[B, H, 93] past frames, reordered from the history_actor blob to the
        canonical training order [dof_pos, dof_vel, gravity, root_ang_vel, action].

        CRITICAL — history_actor is per-TERM-BLOCKED, NOT per-frame-interleaved.
        The env builds it by concatenating 5 SEPARATE lagged obs terms, each
        flattened frame-major within the term:
            [ act(H*29) | angvel(H*3) | dofpos(H*29) | dofvel(H*29) | grav(H*3) ]
        (alphabetical term order: actions, base_ang_vel, dof_pos, dof_vel,
        projected_gravity). So we must split into the 5 contiguous blocks, view
        each as [B, H, Dk], then stack along the field axis — a plain
        ``view(B, H, 93)`` would scramble fields across frames.
        """
        h = obs["history_actor"]
        B = h.shape[0]
        H = self.history_len
        # per-term block widths in alphabetical term order
        D_act, D_angv, D_dofp, D_dofv, D_grav = 29, 3, 29, 29, 3
        o = 0
        act = h[:, o:o + H * D_act].view(B, H, D_act); o += H * D_act
        angv = h[:, o:o + H * D_angv].view(B, H, D_angv); o += H * D_angv
        dofp = h[:, o:o + H * D_dofp].view(B, H, D_dofp); o += H * D_dofp
        dofv = h[:, o:o + H * D_dofv].view(B, H, D_dofv); o += H * D_dofv
        grav = h[:, o:o + H * D_grav].view(B, H, D_grav); o += H * D_grav
        # field order: [dof_pos, dof_vel, gravity, root_ang_vel, action]
        frames = torch.cat([dofp, dofv, grav, angv, act], dim=-1)  # [B, H, 93]
        # TIME ORDER: the lagged history buffer is RECENT-FIRST — _LaggedHistory
        # Wrapper returns [t-1, t-2, ..., t-H] (forward-roll inserts current at 0,
        # older frames at higher idx). But the TRAINING window (_gather_actor_window,
        # offsets [-H..0]) is OLDEST-FIRST [t-H, ..., t-1]. The RoPE actor assigns
        # position by token index, so the orders MUST match or past tokens land on
        # reversed positions at inference vs training. Flip to oldest-first.
        return frames.flip(1)

    def assemble_frames(self, obs: dict) -> torch.Tensor:
        """[B, H+1, 93] raw frames oldest..current (history then current)."""
        past = self._history_frames(obs)                       # [B, H, 93]
        cur = self._current_frame(obs).unsqueeze(1)            # [B, 1, 93]
        return torch.cat([past, cur], dim=1)                  # [B, H+1, 93]

    def _norm_frames(self, frames: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the shared frame BatchNorm to every frame identically.

        ``valid`` ([B, L] bool) marks real frames. In TRAIN mode the actor-loss
        path zero-fills cross/pre-episode positions; those zeros must NOT enter
        the BatchNorm batch statistics (else running_mean -> valid_frac*true_mean
        and running_var is inflated, biasing the SAME stats that rollout/TD-target
        consume in eval mode). So when training with a mask, update the running
        stats from the VALID rows only, then normalize the invalid rows with the
        (now-clean) running stats. Invalid rows are masked out of the loss anyway;
        normalizing them keeps them finite. In eval mode (rollout/TD target) the
        running stats are used for all rows and never updated, so no mask needed.
        """
        B, L, D = frames.shape
        flat = frames.reshape(B * L, D)
        bn = self.frame_norm
        # ALWAYS normalize with the RUNNING stats, never batch stats. A vanilla
        # BatchNorm in train mode normalizes its output with the current BATCH
        # mean/var, but at rollout / TD-target it runs in eval mode and uses
        # RUNNING stats — a train/serve normalization skew (the action the actor
        # is TRAINED to emit under batch-norm differs slightly from the action it
        # emits at rollout under running-norm). Normalizing with running stats in
        # BOTH modes makes train == rollout == TD-target, while still UPDATING the
        # running stats from the VALID frames only (the zeroed invalid positions
        # must not corrupt the statistics). NOTE: this removes a genuine but
        # BOUNDED skew; it is NOT the cause of the Q_disc/Q_aux runaway (that was
        # a [BL] vs [BL,1] broadcast bug in the actor loss — see fb_cpr.py).
        if self.training and valid is not None:
            vmask = valid.reshape(B * L).bool()
            if vmask.sum() > 1:
                with torch.no_grad():
                    v = flat[vmask]
                    batch_mean = v.mean(0)
                    batch_var = v.var(0, unbiased=False)
                    m = bn.momentum if bn.momentum is not None else 0.01
                    bn.running_mean.mul_(1.0 - m).add_(m * batch_mean)
                    bn.running_var.mul_(1.0 - m).add_(m * batch_var)
                    if bn.num_batches_tracked is not None:
                        bn.num_batches_tracked.add_(1)
        with torch.no_grad():
            rm = bn.running_mean
            rv = bn.running_var
            eps = bn.eps
        return ((flat - rm) / torch.sqrt(rv + eps)).view(B, L, D)

    # -- forward (rollout / act): last token only --------------------------
    def forward(self, obs, z, std):
        """obs = RAW obs dict. Returns a TruncatedNormal over the CURRENT-step
        action (last token).

        Derives a per-frame ``valid`` mask from the RAW frames so zero-padded
        history positions (the env zero-fills history_actor at reset; a real
        93-d frame is never exactly all-zero) are masked OUT as attention keys —
        the current token therefore never attends to garbage frames during the
        first H steps of an episode. Matches the train path's storage valid mask.
        """
        raw = self.assemble_frames(obs)                        # [B, H+1, 93] RAW
        valid = raw.abs().sum(dim=-1) > 0                      # [B, H+1] nonzero frame
        valid[:, -1] = True                                    # current always valid
        frames = self._norm_frames(raw)
        mu = self.net(frames, z, valid=valid, last_only=True)  # [B, A] current step
        return self._dist(mu, std)

    # -- training: all H+1 tokens (temporal-parallel scoring) --------------
    def forward_window(self, frames: torch.Tensor, z, std, valid: torch.Tensor | None = None):
        """frames = RAW [B, H+1, 93] window (training path). ``valid`` ([B, H+1])
        marks real frames: it masks cross/pre-episode positions out of the frame
        BatchNorm statistics AND excludes them as attention KEYS (so no token
        attends to zero-padded frames — train matches rollout).

        Returns a TruncatedNormal over ALL H+1 token actions ([B, H+1, A]) for
        temporal-parallel actor scoring. NOTE: past tokens attend to a TRUNCATED
        causal context [t-H..t-p] that does not occur at rollout (where each step
        sees a full window) — accepted trade-off for the extra gradient signal."""
        means = self.net(self._norm_frames(frames, valid), z, valid=valid)  # [B, H+1, A]
        return self._dist(means, std)

    @staticmethod
    def _dist(mu, std):
        if torch.is_tensor(std):
            s = std.to(mu.device, mu.dtype)
            while s.dim() < mu.dim():
                s = s.unsqueeze(-1)
            std_tensor = torch.ones_like(mu) * s
        else:
            std_tensor = torch.ones_like(mu) * std
        return TruncatedNormal(mu, std_tensor)


class Discriminator(nn.Module):
    """BFM CPR discriminator ``D(s, z)`` for style matching vs. expert motions."""

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        z_dim: int,
        hidden_dim: int = 1024,
        hidden_layers: int = 3,
        input_keys: str | tp.Sequence[str] | None = None,
        zero_obs_tail_dims: int = 0,
    ) -> None:
        super().__init__()
        self.input_filter = build_input_filter(obs_space, input_keys)
        filtered_space = self.input_filter.output_space
        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}."
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        obs_dim = filtered_space.shape[0]
        # Ablation: zero out the last N dims of the filtered obs at the
        # disc boundary. Does NOT shrink the network — we keep the same
        # input width and rely on the 0-valued tail to carry no info.
        # This way existing checkpoints load without shape changes and
        # the ablation is a pure data-side intervention.
        assert 0 <= zero_obs_tail_dims <= obs_dim, (
            f"zero_obs_tail_dims={zero_obs_tail_dims} must be in [0, {obs_dim}]"
        )
        self.zero_obs_tail_dims = int(zero_obs_tail_dims)
        seq: list[nn.Module] = [nn.Linear(obs_dim + z_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [nn.Linear(hidden_dim, 1)]
        self.trunk = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.compute_logits(obs, z))

    def compute_logits(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        obs = self.input_filter(obs)
        if self.zero_obs_tail_dims > 0:
            # In-place would modify the caller's tensor — use a clone + index.
            obs = obs.clone()
            obs[..., -self.zero_obs_tail_dims:] = 0.0
        x = torch.cat([z, obs], dim=1)
        return self.trunk(x)

    def compute_reward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        s = self.forward(obs, z)
        s = torch.clamp(s, eps, 1 - eps)
        return s.log() - (1 - s).log()


class ManifoldAttractor(nn.Module):
    """Unconditional state discriminator D(s) — no z conditioning, no
    transition pair. Classifies single obs as expert vs policy, leveraging
    the privileged_state in obs to capture full body state. Constrains the
    policy to stay on the expert motion manifold.
    """

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        hidden_dim: int = 1024,
        hidden_layers: int = 3,
        input_keys: str | tp.Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.input_filter = build_input_filter(obs_space, input_keys)
        filtered_space = self.input_filter.output_space
        assert isinstance(filtered_space, gymnasium.spaces.Box)
        obs_dim = filtered_space.shape[0]
        seq: list[nn.Module] = [
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
        ]
        for _ in range(hidden_layers - 1):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [nn.Linear(hidden_dim, 1)]
        self.trunk = nn.Sequential(*seq)

    def compute_logits(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.trunk(self.input_filter(obs))

    def forward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.sigmoid(self.compute_logits(obs))

    def compute_reward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        eps: float = 1e-7,
    ) -> torch.Tensor:
        s = self.forward(obs)
        s = torch.clamp(s, eps, 1 - eps)
        return s.log() - (1 - s).log()


##########################
# Configs
##########################


@configclass
class FBCprNetworkCfg:
    """Hyperparameters for :class:`FBCprAuxPolicy`.

    Defaults mirror BFM-Zero's production ``train_bfm_zero()`` overrides (see
    ``BFM-Zero/humanoidverse/train.py`` lines 594-718).
    """

    # Latent dimension and projection
    z_dim: int = 256
    """Dimension of the forward-backward latent ``z``."""

    norm_z: bool = True
    """If True, project ``z`` to the sphere of radius ``sqrt(z_dim)``."""

    # --- Anchored variant (Global-through-Anchoring) only ---
    # Split of z into [local | spatial] blocks for the two-head backward map.
    # Must sum to z_dim. Ignored by the standard single-head policy.
    z_local_dim: int = 256
    """Local-body block dimension of z (AnchoredFBCprPolicy)."""
    z_spatial_dim: int = 0
    """Spatial (anchored-pose) block dimension of z (AnchoredFBCprPolicy).
    0 = disabled (standard single-head behaviour)."""
    spatial_input_keys: tp.Sequence[str] = ("anchored_pose",)
    """Obs keys feeding B_spatial (the anchored SE(2) pose A^-1 g)."""

    # Backward map (B)
    backward_hidden_dim: int = 256
    backward_hidden_layers: int = 1
    backward_norm: bool = True
    backward_model: str = "simple"  # {"simple", "residual"} — residual = LN residual MLP
    backward_input_keys: tp.Sequence[str] = ("state", "privileged_state")

    # Forward map (F) / critics share this architecture
    forward_hidden_dim: int = 2048
    forward_model: str = "residual"  # one of {"residual", "simple"}
    forward_hidden_layers: int = 6
    forward_embedding_layers: int = 2
    forward_num_parallel: int = 2
    forward_input_keys: tp.Sequence[str] = (
        "state",
        "privileged_state",
        "last_action",
        "history_actor",
    )
    # Discount conditioning of F: 0 = off (plain F(s,z,a)); >0 makes F(s,z,a,gamma)
    # with a gamma_embed_dim-wide MLP embedding of h=-log(1-gamma). Only the main
    # forward map is conditioned (critics/aux/entropy stay plain).
    forward_gamma_embed_dim: int = 0

    # Actor
    actor_hidden_dim: int = 2048
    actor_model: str = "residual"
    actor_hidden_layers: int = 6
    actor_embedding_layers: int = 2
    actor_std: float = 0.05
    actor_input_keys: tp.Sequence[str] = ("state", "last_action", "history_actor")
    # Actor architecture: "mlp" (default residual MLP, Actor) or "transformer"
    # (RoPE causal transformer over per-timestep tokens, RoPETransformerActor).
    # The transformer actor tokenizes each of the H+1 frames (current + H past)
    # via a shared per-frame linear encoder, prepends a RoPE-exempt z token, runs
    # a causal transformer, and emits H+1 actions in parallel. Each frame =
    # [state (joint_state_dev+gravity+root_ang_vel) | last_action] = frame_dim.
    actor_arch: str = "mlp"
    actor_tf_d_model: int = 512
    actor_tf_layers: int = 6
    actor_tf_heads: int = 8
    actor_tf_mlp_ratio: int = 4
    actor_history_len: int = 9          # H (number of PAST frames; window = H+1)
    actor_frame_dim: int = 93           # per-frame token feature dim (state 64 + last_action 29)

    # Critic (twin Q for discriminator reward); re-uses ForwardMap with
    # ``output_dim = 1`` (scalar Q) or ``output_dim = critic_n_quantiles``
    # when ``critic_distributional=True`` (quantile-regression critic).
    critic_hidden_dim: int = 2048
    critic_model: str = "residual"
    critic_hidden_layers: int = 6
    critic_embedding_layers: int = 2
    critic_num_parallel: int = 2
    critic_input_keys: tp.Sequence[str] = (
        "state",
        "privileged_state",
        "last_action",
        "history_actor",
    )
    # QR distributional critic knobs. When ``critic_distributional=True`` the
    # critic head emits ``critic_n_quantiles`` outputs in place of the single
    # scalar Q. The algorithm trains it with quantile-Huber loss (Dabney et
    # al. 2018) and the actor consumes ``Q = mean(quantiles)`` so the outer
    # update logic is unchanged.
    critic_distributional: bool = False
    critic_n_quantiles: int = 51
    critic_huber_kappa: float = 1.0

    # Aux critic (twin Q for aux env reward). Same contract as critic above.
    aux_critic_hidden_dim: int = 2048
    aux_critic_model: str = "residual"
    aux_critic_hidden_layers: int = 6
    aux_critic_embedding_layers: int = 2
    aux_critic_num_parallel: int = 2
    aux_critic_input_keys: tp.Sequence[str] = (
        "state",
        "privileged_state",
        "last_action",
        "history_actor",
    )
    aux_critic_distributional: bool = False
    aux_critic_n_quantiles: int = 51
    aux_critic_huber_kappa: float = 1.0

    # Discriminator
    discriminator_hidden_dim: int = 1024
    discriminator_hidden_layers: int = 3
    discriminator_input_keys: tp.Sequence[str] = ("state", "privileged_state")
    # Ablation knob: zero out the last N dims of the disc's concat-filtered
    # obs BEFORE it's passed into the trunk. For BFM-Zero's obs layout,
    # ``privileged_state`` is the last key in ``discriminator_input_keys``
    # and its final 93 dims are ``local_body_ang_vel`` (31 keypoints × 3).
    # Setting this to 93 masks that block so the disc cannot separate
    # policy vs expert on sim-vs-mocap ω distribution gap (end-effector
    # ω is especially noisy in PhysX vs spline-smoothed LAFAN). Base
    # ``root_ang_vel`` (inside ``state``) is unaffected.
    discriminator_zero_obs_tail_dims: int = 0

    # Obs normalizer (BatchNorm1d with affine=False) momentums per key
    obs_normalizer_momentum: dict[str, float] = field(
        default_factory=lambda: {
            "state": 0.01,
            "privileged_state": 0.01,
            "last_action": 0.01,
            "history_actor": 0.01,
        }
    )
    obs_normalizer_allow_mismatching_keys: bool = True

    # Aux reward normalizer (EMA, BFM defaults: translate=False, scale=True)
    aux_reward_normalizer_translate: bool = False
    aux_reward_normalizer_scale: bool = True

    # Tracking-inference sequence length (BFM seq_length=8 in production)
    seq_length: int = 8

    # --------------------------------------------------------------- #
    # Reconstruction head: end-effector positions from B(goal) = z
    # --------------------------------------------------------------- #
    # When ``recon_targets`` is non-empty, ``FBCprAuxPolicy`` builds a
    # small MLP ``z -> R^D`` whose target is the concatenation of slices
    # from the goal obs dict. The loss is an MSE added to the FB loss,
    # weighted by ``FBCprAuxAlgorithmCfg.reg_recons_coeff``. Useful for
    # anchoring ``B`` onto task-relevant features (e.g. end-effector XYZ)
    # so the latent ``z`` retains spatial information the FB-only
    # objective might otherwise discard.
    #
    # Each entry is ``(obs_key, start_dim, end_dim)`` — a half-open
    # slice of the named key's flat vector. The targets are concatenated
    # in the order listed. Example (BFM-Terrain, 31-keypoint layout, pelvis
    # stripped so local_body_pos starts at priv index 1):
    #     ankle_left  = keypoint 6  -> priv[1 + (6-1)*3 : 1 + 6*3]   = [16:19]
    #     ankle_right = keypoint 12 -> priv[1 + (12-1)*3 : 1 + 12*3] = [34:37]
    #     wrist_left  = keypoint 22 -> priv[1 + (22-1)*3 : 1 + 22*3] = [64:67]
    #     wrist_right = keypoint 29 -> priv[1 + (29-1)*3 : 1 + 29*3] = [85:88]
    recon_targets: tp.Sequence[tuple[str, int, int]] = ()
    recon_hidden_dim: int = 256
    recon_hidden_layers: int = 2
    recon_model: str = "simple"  # {"simple", "residual"} — residual = same LN residual MLP as B (ignored if recon_linear)
    # BFM-0.5: make the recon head a single LINEAR projection W (no MLP) and
    # augment the target with elementwise squares ([feats, feats^2]). This is
    # the feature-coverage map ensuring the tracking-reward features (and their
    # second moments) lie in the span of B(s) — used with backward_norm=False.
    recon_linear: bool = False
    recon_square_augment: bool = False
    # Per-feature divisor applied to the base recon/z_bar target features so the
    # (BatchNorm-unit-variance) targets land in ~[-1,1] and their squares in
    # ~[0,1] — representable by the sphere-bounded B. 1.0 = no scaling.
    recon_target_scale: float = 1.0

    # --- Manifold attractor ----------------------------------------------
    manifold_attractor: bool = False
    manifold_attractor_hidden_dim: int = 1024
    manifold_attractor_hidden_layers: int = 3
    manifold_attractor_input_keys: tp.Sequence[str] = ("state", "privileged_state")

    # --- Soft FB ---------------------------------------------------------
    soft_fb: bool = False
    entropy_critic_hidden_dim: int = 1024
    entropy_critic_hidden_layers: int = 3
    entropy_critic_input_keys: tp.Sequence[str] = (
        "state", "privileged_state", "last_action", "history_actor",
    )
    actor_learned_std: bool = False
    actor_min_std: float = 0.01
    actor_max_std: float = 0.25


##########################
# Top-level policy
##########################


class FBCprAuxPolicy(nn.Module):
    """Port of BFM-Zero's :class:`FBcprAuxModel` as a self-contained ``nn.Module``.

    Composes:
      * ``_backward_map`` / ``_target_backward_map`` -- the B network.
      * ``_forward_map`` / ``_target_forward_map``  -- the F (twin) network.
      * ``_actor``                                  -- policy head.
      * ``_critic`` / ``_target_critic``            -- twin Q for discriminator reward.
      * ``_aux_critic`` / ``_target_aux_critic``    -- twin Q for auxiliary env reward.
      * ``_discriminator``                          -- CPR style discriminator.
      * ``_obs_normalizer``                         -- per-key BatchNorm1d normalizer.
      * ``_aux_reward_normalizer``                  -- EMA reward normalizer (scale=True).

    Target networks are created lazily via :meth:`_prepare_for_train` (deepcopy).
    """

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        action_dim: int,
        cfg: FBCprNetworkCfg,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_space = obs_space
        self.action_dim = action_dim

        # Cache a few common hyperparams for readability downstream.
        self.z_dim: int = cfg.z_dim
        self.norm_z: bool = cfg.norm_z
        self.seq_length: int = cfg.seq_length
        self.actor_std: float = cfg.actor_std

        # Obs normalizer.
        self._obs_normalizer = ObsNormalizer(
            obs_space,
            per_key_momentum_dict=dict(cfg.obs_normalizer_momentum),
            allow_mismatching_keys=cfg.obs_normalizer_allow_mismatching_keys,
        )

        # Backward map.
        self._backward_map = BackwardMap(
            obs_space,
            z_dim=cfg.z_dim,
            hidden_dim=cfg.backward_hidden_dim,
            hidden_layers=cfg.backward_hidden_layers,
            norm=cfg.backward_norm,
            input_keys=cfg.backward_input_keys,
            model=cfg.backward_model,
        )

        # Optional reconstruction head (end-effector decoder from z).
        # Parameters are added to the backward optimizer inside FBCprAux
        # so ``B`` is trained jointly with the decoder.
        self._reconstruction_head: ReconstructionHead | None = None
        if cfg.recon_targets:
            self._reconstruction_head = ReconstructionHead(
                z_dim=cfg.z_dim,
                targets=cfg.recon_targets,
                hidden_dim=cfg.recon_hidden_dim,
                hidden_layers=cfg.recon_hidden_layers,
                linear=bool(getattr(cfg, "recon_linear", False)),
                square_augment=bool(getattr(cfg, "recon_square_augment", False)),
                target_scale=float(getattr(cfg, "recon_target_scale", 1.0)),
                model=str(getattr(cfg, "recon_model", "simple")),
            )

        # Forward map (z-output).
        self._forward_map = ForwardMap(
            obs_space,
            z_dim=cfg.z_dim,
            action_dim=action_dim,
            hidden_dim=cfg.forward_hidden_dim,
            model=cfg.forward_model,
            hidden_layers=cfg.forward_hidden_layers,
            embedding_layers=cfg.forward_embedding_layers,
            num_parallel=cfg.forward_num_parallel,
            input_keys=cfg.forward_input_keys,
            gamma_embed_dim=int(getattr(cfg, "forward_gamma_embed_dim", 0)),
        )
        # Whether F consumes a gamma argument (drives call sites in the algorithm).
        self.forward_gamma_conditioned = int(getattr(cfg, "forward_gamma_embed_dim", 0)) > 0
        # Default gamma for the public forward_map() accessor when a caller omits
        # it (e.g. play/eval Q-probes). Long horizon; the algorithm always passes
        # explicit per-row gamma during training so this is inference-only.
        self.fb_gamma_default = float(getattr(cfg, "fb_gamma_default", 0.98))

        # Soft FB flag.
        self.soft_fb: bool = bool(getattr(cfg, "soft_fb", False))

        # Actor.
        _learned_std = self.soft_fb or bool(getattr(cfg, "actor_learned_std", False))
        if str(getattr(cfg, "actor_arch", "mlp")) == "transformer":
            self._actor = TransformerActorWrapper(
                obs_space,
                z_dim=cfg.z_dim,
                action_dim=action_dim,
                frame_dim=int(cfg.actor_frame_dim),
                history_len=int(cfg.actor_history_len),
                d_model=int(cfg.actor_tf_d_model),
                n_layers=int(cfg.actor_tf_layers),
                n_heads=int(cfg.actor_tf_heads),
                mlp_ratio=int(cfg.actor_tf_mlp_ratio),
            )
        else:
            self._actor = Actor(
                obs_space,
                z_dim=cfg.z_dim,
                action_dim=action_dim,
                hidden_dim=cfg.actor_hidden_dim,
                model=cfg.actor_model,
                hidden_layers=cfg.actor_hidden_layers,
                embedding_layers=cfg.actor_embedding_layers,
                input_keys=cfg.actor_input_keys,
                learned_std=_learned_std,
                min_std=float(getattr(cfg, "actor_min_std", 0.01)),
                max_std=float(getattr(cfg, "actor_max_std", 1.0)),
            )

        # Discriminator.
        self._discriminator = Discriminator(
            obs_space,
            z_dim=cfg.z_dim,
            hidden_dim=cfg.discriminator_hidden_dim,
            hidden_layers=cfg.discriminator_hidden_layers,
            input_keys=cfg.discriminator_input_keys,
            zero_obs_tail_dims=getattr(cfg, "discriminator_zero_obs_tail_dims", 0),
        )

        # Critic (twin Q for discriminator reward). Output is 1 scalar Q per
        # ensemble member by default, or ``critic_n_quantiles`` (QR) when
        # ``critic_distributional=True``.
        critic_out = cfg.critic_n_quantiles if cfg.critic_distributional else 1
        self._critic = ForwardMap(
            obs_space,
            z_dim=cfg.z_dim,
            action_dim=action_dim,
            hidden_dim=cfg.critic_hidden_dim,
            model=cfg.critic_model,
            hidden_layers=cfg.critic_hidden_layers,
            embedding_layers=cfg.critic_embedding_layers,
            num_parallel=cfg.critic_num_parallel,
            input_keys=cfg.critic_input_keys,
            output_dim=critic_out,
        )

        # Aux critic (twin Q for aux env reward). Same contract as above.
        aux_critic_out = cfg.aux_critic_n_quantiles if cfg.aux_critic_distributional else 1
        self._aux_critic = ForwardMap(
            obs_space,
            z_dim=cfg.z_dim,
            action_dim=action_dim,
            hidden_dim=cfg.aux_critic_hidden_dim,
            model=cfg.aux_critic_model,
            hidden_layers=cfg.aux_critic_hidden_layers,
            embedding_layers=cfg.aux_critic_embedding_layers,
            num_parallel=cfg.aux_critic_num_parallel,
            input_keys=cfg.aux_critic_input_keys,
            output_dim=aux_critic_out,
        )

        # Aux reward normalizer (EMA).
        self._aux_reward_normalizer = EMA(
            translate=cfg.aux_reward_normalizer_translate,
            scale=cfg.aux_reward_normalizer_scale,
        )

        # Manifold attractor D_ma(s_t, s_{t+1}) — unconditional disc.
        self._manifold_attractor: ManifoldAttractor | None = None
        if getattr(cfg, "manifold_attractor", False):
            self._manifold_attractor = ManifoldAttractor(
                obs_space,
                hidden_dim=cfg.manifold_attractor_hidden_dim,
                hidden_layers=cfg.manifold_attractor_hidden_layers,
                input_keys=cfg.manifold_attractor_input_keys,
            )

        # Entropy critic Q_H(s, a, z) → scalar. Built only for Soft FB.
        self._entropy_critic: ForwardMap | None = None
        if self.soft_fb:
            self._entropy_critic = ForwardMap(
                obs_space,
                z_dim=cfg.z_dim,
                action_dim=action_dim,
                hidden_dim=cfg.entropy_critic_hidden_dim,
                model="simple",
                hidden_layers=cfg.entropy_critic_hidden_layers,
                embedding_layers=2,
                num_parallel=1,
                input_keys=cfg.entropy_critic_input_keys,
                output_dim=1,
            )

        # Target networks are lazily built in `_prepare_for_train()`.
        self._target_backward_map: nn.Module | None = None
        self._target_forward_map: nn.Module | None = None
        self._target_critic: nn.Module | None = None
        self._target_aux_critic: nn.Module | None = None
        self._target_entropy_critic: nn.Module | None = None

        # By default we keep the policy in eval mode with grads off (trainer flips these on).
        self.train(False)
        self.requires_grad_(False)

    # ---- training setup ----

    def _prepare_for_train(self) -> None:
        """Create target networks as deepcopies of their live counterparts."""
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)
        self._target_critic = copy.deepcopy(self._critic)
        self._target_aux_critic = copy.deepcopy(self._aux_critic)
        if self._entropy_critic is not None:
            self._target_entropy_critic = copy.deepcopy(self._entropy_critic)
        # Targets are never optimized directly.
        for target in (
            self._target_backward_map,
            self._target_forward_map,
            self._target_critic,
            self._target_aux_critic,
            self._target_entropy_critic,
        ):
            if target is not None:
                target.requires_grad_(False)

    # ---- analytic task-embedding from the linear-W feature decoder ----

    @torch.no_grad()
    def zbar_from_goal(self, goal_x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Construct the (projected) FB task embedding ``z_bar_g`` for a tracking
        goal analytically from the learned linear feature decoder W.

        The W head reconstructs ``y(s) = [x(s), x(s)^2] in R^{2n}`` from B(s) via a
        no-bias linear map ``W: R^d -> R^{2n}`` (``y ~= W phi(s)``, i.e. the
        feature decoder is ``D = W``). A diagonal-Lambda tracking reward for goal
        ``g`` is, up to a state-independent constant,
            r_g(s) = 2 x^T Lambda g - x^T Lambda x
                   = c_g^T y(s),   c_g = [2 Lambda g ; -diag(Lambda)] in R^{2n}.
        Hence the un-normalized task embedding is ``z_bar_g = W^T c_g`` (since
        ``r_g(s) ~= c_g^T W phi(s) = phi(s)^T (W^T c_g)``). Returned z is
        ``project_z``'d onto the z-sphere (z is normalized; B is not).

        Args:
            goal_x: ``[B, n]`` the n base goal features (same slices/order as the
                W head's targets, NO square augmentation — just ``x(g)``).
            weights: ``[n]`` diagonal Lambda (per-feature tracking weight).
        Returns:
            ``[B, z_dim]`` projected task embeddings, or ``None`` if no linear
            square-augmented recon head is present.
        """
        head = getattr(self, "_reconstruction_head", None)
        if head is None or not getattr(head, "square_augment", False):
            return None
        W = head.net.weight  # [2n, d]  (no bias)
        n = head.base_dim
        w = weights.to(goal_x.device, goal_x.dtype).view(1, n)
        c_linear = 2.0 * w * goal_x          # [B, n]
        c_square = (-weights.view(1, n)).expand(goal_x.shape[0], n).to(goal_x.device, goal_x.dtype)  # [B, n]
        c_goal = torch.cat([c_linear, c_square], dim=-1)  # [B, 2n]
        z_bar = c_goal @ W                    # [B, d]   (= c_goal . W = W^T c_goal per-row)
        return self.project_z(z_bar)

    # ---- latent sampling / projection ----

    def sample_z(self, batch_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """Sample a batch of latent ``z``.

        Standard FB: project to sphere surface of radius ``sqrt(z_dim)``.
        Soft FB: sample from unit ball (R=1) with uniform radius.
        """
        z = torch.randn((batch_size, self.z_dim), dtype=torch.float32, device=device)
        z = F.normalize(z, dim=-1)
        if self.soft_fb:
            r = torch.rand(batch_size, 1, dtype=torch.float32, device=device)
            return z * r
        R = math.sqrt(self.z_dim)
        if self.norm_z:
            return z * R
        return z

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        """Project z.

        Standard FB: project to sphere surface (radius sqrt(z_dim)).
        Soft FB: squash norm via z / (||z|| + 1) — smoothly maps
        any norm into [0, 1) while preserving direction. R=1.
        """
        if self.soft_fb:
            norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return z / (norm + 1.0)
        R = math.sqrt(z.shape[-1])
        if self.norm_z:
            return R * F.normalize(z, dim=-1)
        return z

    # ---- normalization ----

    def _normalize(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    # ---- no-grad inference wrappers (match BFM's convention) ----

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self._backward_map(self._normalize(obs))

    @torch.no_grad()
    def forward_map(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        action: torch.Tensor,
        gamma: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if self.forward_gamma_conditioned:
            # Default to the long horizon (fb_gamma_default) when a caller (e.g.
            # a play/eval Q-probe) does not supply gamma, so external callers do
            # not need to know F is conditioned.
            if gamma is None:
                gamma = float(getattr(self, "fb_gamma_default", 0.98))
            if not torch.is_tensor(gamma):
                n = z.shape[0]
                gamma = torch.full((n,), float(gamma), device=z.device)
            return self._forward_map(self._normalize(obs), z, action, gamma)
        return self._forward_map(self._normalize(obs), z, action)

    @torch.no_grad()
    def actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        std: float | torch.Tensor,
    ) -> TruncatedNormal:
        # The transformer actor consumes RAW obs (it owns a shared per-frame
        # BatchNorm); the MLP actor uses the per-key obs normalizer.
        if isinstance(self._actor, TransformerActorWrapper):
            return self._actor(obs, z, std)
        return self._actor(self._normalize(obs), z, std)

    @torch.no_grad()
    def critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self._critic(self._normalize(obs), z, action)

    @torch.no_grad()
    def aux_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self._aux_critic(self._normalize(obs), z, action)

    @torch.no_grad()
    def discriminator(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        return self._discriminator(self._normalize(obs), z)

    # ---- action sampling ----

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        mean: bool = True,
        std: "float | torch.Tensor | None" = None,
    ) -> torch.Tensor:
        # ``std`` overrides the scalar ``self.actor_std`` for exploration. A
        # per-env tensor (shape [N] or [N,1]) broadcasts against the action mean
        # so each env can roll out with its own exploration scale. mean=True
        # (deterministic) ignores std.
        act_std = self.actor_std if std is None else std
        dist = self.actor(obs, z, act_std)
        if mean:
            return dist.mean.float()
        return dist.sample().float()


##########################
# FB-CPR-Aux-Cond variant — extra exteroceptive "measure_cond" obs key
##########################


@configclass
class FBCprCondNetworkCfg(FBCprNetworkCfg):
    """FB-CPR-Aux variant conditioned on an extra exteroceptive obs key.

    The obs key (default ``"measure_cond"``) is appended to:
      * the Forward (F) network inputs — successor-measure condition,
      * the Actor inputs,
      * the Aux-critic inputs.

    It is NOT fed into the Backward map (B), the discriminator-reward
    critic, or the discriminator itself. An ``ObsNormalizer`` entry is
    added so BatchNorm1d stats are tracked per-key like the others.

    All other FBCprNetworkCfg defaults are inherited unchanged.
    """

    measure_cond_key: str = "measure_cond"
    include_measure_cond_in_forward: bool = True
    include_measure_cond_in_actor: bool = True
    include_measure_cond_in_aux_critic: bool = True
    measure_cond_momentum: float = 0.01

    def __post_init__(self):
        # __post_init__ on a subclass: configclass propagates parent defaults
        # before this runs, so we just augment the key tuples and the
        # normalizer dict.
        mk = self.measure_cond_key

        def _append_if_missing(keys: tp.Sequence[str]) -> tp.Sequence[str]:
            return tuple(keys) if mk in keys else tuple(list(keys) + [mk])

        if self.include_measure_cond_in_forward:
            self.forward_input_keys = _append_if_missing(self.forward_input_keys)
        if self.include_measure_cond_in_actor:
            self.actor_input_keys = _append_if_missing(self.actor_input_keys)
        if self.include_measure_cond_in_aux_critic:
            self.aux_critic_input_keys = _append_if_missing(self.aux_critic_input_keys)

        # Normalizer entry for the new key.
        if mk not in self.obs_normalizer_momentum:
            self.obs_normalizer_momentum = {
                **self.obs_normalizer_momentum,
                mk: self.measure_cond_momentum,
            }


class FBCprCondPolicy(FBCprAuxPolicy):
    """:class:`FBCprAuxPolicy` conditioned on an extra exteroceptive obs
    key (``cfg.measure_cond_key``), fed into the F network, the actor,
    and the aux critic (but NOT B, the disc-reward critic, or the
    discriminator).

    All of the plumbing — ``DictInputConcatFilter``, ``ObsNormalizer``,
    target nets, act/critic/aux_critic forwards — is inherited unchanged.
    The cfg's ``__post_init__`` augments the input-key tuples; the base
    class then builds each net with the extended inputs.
    """

    def __init__(self, obs_space, action_dim: int, cfg: FBCprCondNetworkCfg):
        mk = cfg.measure_cond_key
        if mk not in obs_space.spaces:
            raise ValueError(
                f"FBCprCondPolicy: obs_space is missing measure_cond_key "
                f"'{mk}'. Available keys: {list(obs_space.spaces)}"
            )
        super().__init__(obs_space, action_dim, cfg)
        self.measure_cond_key = mk

    def load_state_dict(self, state_dict, strict: bool = True):
        """Be forgiving when loading an older FBCprAux checkpoint.

        Old checkpoints do not have a ``measure_cond`` entry in the
        obs-normalizer, and the F / actor / aux_critic first-layer
        shapes differ by ``measure_cond_dim`` columns. Fall back to
        ``strict=False`` and rely on fresh init for the missing rows.
        """
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            print(
                "[FBCprCondPolicy] load_state_dict strict=True failed "
                f"({e.__class__.__name__}). Retrying with strict=False; "
                "F/actor/aux_critic first layers and measure_cond "
                "normalizer stats will be re-initialized."
            )
            return super().load_state_dict(state_dict, strict=False)


__all__ = [
    # Configs
    "FBCprNetworkCfg",
    "FBCprCondNetworkCfg",
    # Top-level policy
    "FBCprAuxPolicy",
    "FBCprCondPolicy",
    # Networks
    "BackwardMap",
    "ForwardMap",
    "Actor",
    "Discriminator",
    # Normalizers / filters / helpers
    "ObsNormalizer",
    "BatchNormNormalizer",
    "DictInputFilter",
    "DictInputConcatFilter",
    "build_input_filter",
    "EMA",
    "Norm",
    "TruncatedNormal",
    "SquashedNormal",
    "DenseParallel",
    "ParallelLayerNorm",
    "ResidualBlock",
    "Block",
    "residual_embedding",
    "simple_embedding",
    "linear",
    "layernorm",
    "parallel_orthogonal_",
    "weight_init",
    "soft_update_params",
    "_soft_update_params",
    "eval_mode",
]
