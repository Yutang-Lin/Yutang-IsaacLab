# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from .actor_critic import ActorCritic
from isaaclab_rl.rsl_rl.networks.hyper_mlp import HyperMLP
from isaaclab_rl.rsl_rl.utils import TensorDict


class ActorCriticHyperMLP(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        hyper_layer_idx: int = 0,
        proprio_horizon: int = 5,
        
        init_noise_std=1.0,
        load_noise_std: bool = True,
        learnable_noise_std: bool = True,
        noise_std_type: str = "scalar",
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
        residual: bool = False,
        actor_obs_meta: dict | None = None,
        critic_obs_meta: dict | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorDoubleCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        nn.Module.__init__(self)
        activation = resolve_nn_activation(activation) # type: ignore

        self.load_noise_std = load_noise_std    
        self.learnable_noise_std = learnable_noise_std
        self.actor_obs_meta = actor_obs_meta
        self.critic_obs_meta = critic_obs_meta
        self.actor_proprio_ids, self.actor_control_ids = self._resolve_obs_meta(num_actor_obs, actor_obs_meta)

        self.proprio_horizon = proprio_horizon
        num_proprio_obs = self.actor_proprio_ids.shape[0]
        num_control_obs = self.actor_control_ids.shape[0]
        assert num_proprio_obs % self.proprio_horizon == 0, "Number of proprio observations must be divisible by proprio horizon"
        proprio_dim = num_proprio_obs // self.proprio_horizon
        control_dim = proprio_dim * (self.proprio_horizon - 1) + num_control_obs

        # Policy
        self.actor = HyperMLP(proprio_dim, actor_hidden_dims, num_actions, control_dim, hyper_layer_idx + 2)

        # Value function
        self.critic = None

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution: Normal = None # type: ignore
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def extra_loss(self, **kwargs):
        loss_dict = {'orthogonal': self.actor.orthogonal_loss()}
        return loss_dict

    def _resolve_obs_meta(self, num_obs, obs_meta):
        all_obs = torch.arange(num_obs)
        proprio_obs = torch.ones(num_obs, dtype=torch.bool)
        control_obs = []
        for seg in obs_meta['controls']:
            control_obs.append(all_obs[seg['start']:seg['end']].clone())
            proprio_obs[seg['start']:seg['end']] = False

        proprio_obs = all_obs[proprio_obs].clone().contiguous()
        control_obs = torch.cat(control_obs).contiguous()
        return proprio_obs, control_obs
    
    def _split_observations(self, observations: torch.Tensor):
        proprio_obs = observations[..., self.actor_proprio_ids]
        control_obs = observations[..., self.actor_control_ids]
        proprio_obs = proprio_obs.reshape(proprio_obs.shape[0], self.proprio_horizon, -1)
        control_obs = torch.cat([proprio_obs[:, :-1].flatten(start_dim=1), control_obs], dim=-1).contiguous()
        proprio_obs = proprio_obs[:, -1].contiguous()
        return TensorDict(proprio=proprio_obs, control=control_obs)
    
    def act(self, observations, **kwargs):
        self.update_distribution(self._split_observations(observations))
        return self.distribution.sample()

    def act_inference(self, observations, **kwargs):
        actions_mean = self.actor(self._split_observations(observations))
        return actions_mean