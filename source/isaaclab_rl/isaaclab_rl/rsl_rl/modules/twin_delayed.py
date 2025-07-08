# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation
from copy import deepcopy
from .actor_critic import ActorCritic

class DoubleCritic(nn.Module):
    def __init__(self, num_critic_obs, critic_hidden_dims=[256, 256, 256], activation="elu"):
        super().__init__()
        activation = resolve_nn_activation(activation)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic_1 = nn.Sequential(*critic_layers)
        self.critic_2 = nn.Sequential(*critic_layers)

    def forward(self, critic_observations):
        value_1 = self.critic_1(critic_observations)
        value_2 = self.critic_2(critic_observations)
        return value_1, value_2
    
    def evaluate(self, critic_observations):
        value_1, value_2 = self(critic_observations)
        return torch.min(value_1, value_2)

class TwinDelayed(ActorCritic):
    '''Deterministic Twin Delayed Model for TD3'''
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        nn.Module.__init__(self)
        self.activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs + num_actions
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(self.activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(self.activation)
        self.actor = nn.Sequential(*actor_layers)
        self.critic = DoubleCritic(mlp_input_dim_c, critic_hidden_dims, activation)  

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self._action_mean = None

    @property
    def action_mean(self):
        return self._action_mean

    @property
    def action_std(self):
        return 0.0

    @property
    def entropy(self):
        raise ValueError("Entropy is not defined for deterministic models")

    def act(self, observations, **kwargs):
        actions = self.actor(observations)
        return actions

    def get_actions_log_prob(self, actions):
        raise ValueError("Log probability is not defined for deterministic models")

    def evaluate(self, critic_observations, actions, return_both=False, **kwargs):
        critic_observations = torch.cat([critic_observations, actions], dim=1)
        value_1, value_2 = self.critic(critic_observations)
        if return_both:
            return value_1, value_2
        else:
            return torch.min(value_1, value_2)
