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


class ActorCriticOU(nn.Module):
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
                "ActorCriticOU.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation) # type: ignore

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
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # OU noise
        self.log_theta = nn.Parameter(torch.log(init_theta * torch.ones(num_actions)))
        self.log_sigma = nn.Parameter(torch.log(init_sigma * torch.ones(num_actions)))
        self.theta_range = theta_range
        self.sigma_range = sigma_range
        self.log_theta_range = [np.log(theta_range[0]), np.log(theta_range[1])]
        self.log_sigma_range = [np.log(sigma_range[0]), np.log(sigma_range[1])]

        self.step_dt = step_dt
        self.sqrt_step_dt = np.sqrt(step_dt)
        self.ou_noise: torch.Tensor = None # type: ignore
        self.last_ou_noise: torch.Tensor = None # type: ignore

        # Action distribution (populated in update_distribution)
        self.distribution: Normal = None # type: ignore
        self.ou_distribution: Normal = None # type: ignore
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        self.mean = mean
        # compute standard deviation
        theta = torch.exp(self.log_theta).expand_as(mean)
        sigma = torch.exp(self.log_sigma).expand_as(mean)

        if self.ou_noise is None or mean.shape != self.ou_noise.shape:
            self.ou_noise = torch.zeros_like(mean)

        ou_mean = (1 - theta * self.step_dt) * self.ou_noise + mean
        self.ou_distribution = Normal(ou_mean, sigma * self.sqrt_step_dt)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        return
    
        if self.ou_noise is not None:
            if dones is not None:
                self.ou_noise[dones] = 0.
            else:
                self.ou_noise.zero_()

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.ou_distribution.mean

    @property
    def action_std(self):
        return self.ou_distribution.stddev

    @property
    def entropy(self):
        return self.ou_distribution.entropy().sum(dim=-1)
    
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        actions = self.ou_distribution.sample()
        
        self.last_ou_noise = self.ou_noise.detach()
        self.ou_noise = actions - self.mean
        return actions

    def get_actions_log_prob(self, actions, **kwargs):
        return self.ou_distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
