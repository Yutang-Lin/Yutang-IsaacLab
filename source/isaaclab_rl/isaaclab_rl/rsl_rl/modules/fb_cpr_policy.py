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
    """BFM backward map ``B(s_+) -> z`` with optional sphere projection."""

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        z_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 1,
        norm: bool = True,
        input_keys: str | tp.Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.input_filter = build_input_filter(obs_space, input_keys)
        filtered_space = self.input_filter.output_space
        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}."
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        seq: list[nn.Module] = [nn.Linear(filtered_space.shape[0], hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [nn.Linear(hidden_dim, z_dim)]
        if norm:
            seq += [Norm()]
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.input_filter(x)
        return self.net(x)


class ForwardMap(nn.Module):
    """BFM forward map ``F(s, z, a) -> z`` (also reused as the critic / aux-critic)."""

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

        if model == "residual":
            embed_fn = residual_embedding
        elif model == "simple":
            embed_fn = simple_embedding
        else:
            raise ValueError(f"Unsupported forward_map model {model}")

        # BFM quirk: the residual variant of ForwardMap/ResidualForwardMap
        # passes ``cfg.hidden_layers`` (not ``cfg.embedding_layers``) as the
        # embedding depth — see BFM-Zero nn_models.py:484-485. The simple
        # variant honours ``embedding_layers``. Keep parity.
        embed_depth = hidden_layers if model == "residual" else embedding_layers
        self.embed_z = embed_fn(obs_dim + z_dim, hidden_dim, embed_depth, num_parallel)
        self.embed_sa = embed_fn(obs_dim + action_dim, hidden_dim, embed_depth, num_parallel)

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
    ) -> torch.Tensor:
        obs = self.input_filter(obs)
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))
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

        if model == "residual":
            embed_fn = residual_embedding
        elif model == "simple":
            embed_fn = simple_embedding
        else:
            raise ValueError(f"Unsupported actor model {model}")

        # Actor has no parallel ensemble dim.
        self.embed_z = embed_fn(obs_dim + z_dim, hidden_dim, embedding_layers, 1)
        self.embed_s = embed_fn(obs_dim, hidden_dim, embedding_layers, 1)

        if model == "residual":
            seq: list[nn.Module] = [ResidualBlock(hidden_dim) for _ in range(hidden_layers)]
            seq += [Block(hidden_dim, action_dim, False)]
        else:
            seq = []
            for _ in range(hidden_layers):
                seq += [linear(hidden_dim, hidden_dim), nn.ReLU()]
            seq += [linear(hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

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
        mu = torch.tanh(self.policy(embedding))
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

    # Backward map (B)
    backward_hidden_dim: int = 256
    backward_hidden_layers: int = 1
    backward_norm: bool = True
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

    # Actor
    actor_hidden_dim: int = 2048
    actor_model: str = "residual"
    actor_hidden_layers: int = 6
    actor_embedding_layers: int = 2
    actor_std: float = 0.05
    actor_input_keys: tp.Sequence[str] = ("state", "last_action", "history_actor")

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
        )

        # Actor.
        self._actor = Actor(
            obs_space,
            z_dim=cfg.z_dim,
            action_dim=action_dim,
            hidden_dim=cfg.actor_hidden_dim,
            model=cfg.actor_model,
            hidden_layers=cfg.actor_hidden_layers,
            embedding_layers=cfg.actor_embedding_layers,
            input_keys=cfg.actor_input_keys,
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

        # Target networks are lazily built in `_prepare_for_train()`.
        self._target_backward_map: nn.Module | None = None
        self._target_forward_map: nn.Module | None = None
        self._target_critic: nn.Module | None = None
        self._target_aux_critic: nn.Module | None = None

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
        # Targets are never optimized directly.
        for target in (
            self._target_backward_map,
            self._target_forward_map,
            self._target_critic,
            self._target_aux_critic,
        ):
            if target is not None:
                target.requires_grad_(False)

    # ---- latent sampling / projection ----

    def sample_z(self, batch_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """Sample a batch of latent ``z``. Projected to the sphere of radius ``sqrt(z_dim)`` if ``norm_z``."""
        z = torch.randn((batch_size, self.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.norm_z:
            return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
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
    ) -> torch.Tensor:
        return self._forward_map(self._normalize(obs), z, action)

    @torch.no_grad()
    def actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        std: float | torch.Tensor,
    ) -> TruncatedNormal:
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
    ) -> torch.Tensor:
        dist = self.actor(obs, z, self.actor_std)
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
