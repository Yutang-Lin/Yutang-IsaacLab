# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from .actor_critic import ActorCritic


class ActorDoubleCritic(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        load_noise_std: bool = True,

        step_dt: float = 0.02,
        init_theta: float = 0.25,
        init_sigma: float = 0.1,
        theta_range: list[float] = [0.1, 0.9],
        sigma_range: list[float] = [0.1, 0.25],
        noise_std_type: str = "scalar",
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
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

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                if layer_norm:
                    actor_layers.append(nn.LayerNorm(actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
                if dropout_rate > 0:
                    actor_layers.append(nn.Dropout(dropout_rate))
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                if layer_norm:
                    critic_layers.append(nn.LayerNorm(critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
                if dropout_rate > 0:
                    critic_layers.append(nn.Dropout(dropout_rate))
        self.critic_behave = nn.Sequential(*critic_layers)
        self.critic_target = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic_behave}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # OU noise
        self.theta = init_theta
        self.sigma = init_sigma
        self.theta_range = theta_range
        self.sigma_range = sigma_range
        self.step_dt = step_dt
        self.sqrt_step_dt = np.sqrt(step_dt)
        self.ou_noise: torch.Tensor = None # type: ignore

        # Action distribution (populated in update_distribution)
        self.distribution: Normal = None # type: ignore
        self.ou_distribution: Normal = None # type: ignore
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        if self.ou_noise is not None:
            if dones is not None:
                self.ou_noise[dones] = 0.
            else:
                self.ou_noise.zero_()

    def forward(self):
        raise NotImplementedError
    
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        mean = self.action_mean

        with torch.no_grad():
            if self.ou_noise is None or mean.shape != self.ou_noise.shape:
                self.ou_noise = torch.zeros_like(mean)

            ou_mean = (1 - self.theta * self.step_dt) * self.ou_noise + mean
            self.ou_distribution = Normal(ou_mean, self.sigma * torch.ones_like(ou_mean) * self.sqrt_step_dt)
            actions = self.ou_distribution.sample()

            self.ou_noise = actions - mean
        return actions

    def get_actions_log_prob(self, actions, collecting: bool = False, **kwargs):
        if collecting:
            return self.ou_distribution.log_prob(actions).sum(dim=-1)
        else:
            return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        value_behave = self.critic_behave(critic_observations)
        value_target = self.critic_target(critic_observations)
        return value_behave, value_target