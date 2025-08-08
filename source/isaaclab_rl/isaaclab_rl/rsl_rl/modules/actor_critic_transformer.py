# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

from isaaclab_rl.rsl_rl.modules import ActorCritic
from isaaclab_rl.rsl_rl.networks import TransformerMemory
from isaaclab_rl.rsl_rl.utils import resolve_nn_activation


class ActorCriticTransformer(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        tf_d_model=256,
        tf_num_input_tokens=4,
        tf_num_history_tokens=4,
        tf_num_layers=1,
        tf_num_heads=4,
        tf_hidden_dim=256,
        tf_dropout=0.05,
        tf_activation="gelu",
        tf_lnn_dt=0.02,
        tf_lnn_tau=0.5,
        init_noise_std=1.0,
        load_noise_std: bool = True,
        learnable_noise_std: bool = True,
        noise_std_type: str = "scalar",
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
        residual: bool = False,
        actor_obs_meta: dict = None,
        critic_obs_meta: dict = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=tf_d_model,
            num_critic_obs=tf_d_model,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            load_noise_std=load_noise_std,
            learnable_noise_std=learnable_noise_std,
            noise_std_type=noise_std_type,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
            residual=residual,
            actor_obs_meta=actor_obs_meta,
            critic_obs_meta=critic_obs_meta,
        )

        tf_activation = resolve_nn_activation(tf_activation)
        self.memory_a = TransformerMemory(num_actor_obs, 
                                         num_input_tokens=tf_num_input_tokens,
                                         num_history_tokens=tf_num_history_tokens,
                                         num_layers=tf_num_layers, 
                                         d_model=tf_d_model, 
                                         hidden_dim=tf_hidden_dim, 
                                         num_heads=tf_num_heads, 
                                         dropout=tf_dropout, 
                                         activation=tf_activation,
                                         lnn_dt=tf_lnn_dt,
                                         lnn_tau=tf_lnn_tau)
        self.memory_c = TransformerMemory(num_critic_obs, 
                                         num_input_tokens=tf_num_input_tokens,
                                         num_history_tokens=tf_num_history_tokens,
                                         num_layers=tf_num_layers, 
                                         d_model=tf_d_model, 
                                         hidden_dim=tf_hidden_dim, 
                                         num_heads=tf_num_heads, 
                                         dropout=tf_dropout, 
                                         activation=tf_activation,
                                         lnn_dt=tf_lnn_dt,
                                         lnn_tau=tf_lnn_tau)

        print(f"Actor Transformer: {self.memory_a}")
        print(f"Critic Transformer: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states
