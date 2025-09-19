# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

from isaaclab_rl.rsl_rl.modules import ActorCritic
from isaaclab_rl.rsl_rl.networks import TransformerPolicy
from isaaclab_rl.rsl_rl.utils import resolve_nn_activation

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCriticTransformer(ActorCritic):
    is_recurrent = False

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
        tf_num_layers=1,
        tf_num_heads=4,
        tf_hidden_dim=256,
        tf_dropout=0.05,
        tf_activation="gelu",
        tf_critic_d_model=None,
        tf_critic_num_input_tokens=None,
        tf_critic_num_layers=None,
        tf_critic_num_heads=None,
        tf_critic_hidden_dim=None,
        tf_critic_dropout=None,
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

        nn.Module.__init__(self)
        activation = resolve_nn_activation(activation)
        tf_activation = resolve_nn_activation(tf_activation)
        tf_critic_d_model = tf_critic_d_model if tf_critic_d_model is not None else tf_d_model
        tf_critic_num_input_tokens = tf_critic_num_input_tokens if tf_critic_num_input_tokens is not None else tf_num_input_tokens
        tf_critic_num_layers = tf_critic_num_layers if tf_critic_num_layers is not None else tf_num_layers
        tf_critic_num_heads = tf_critic_num_heads if tf_critic_num_heads is not None else tf_num_heads
        tf_critic_hidden_dim = tf_critic_hidden_dim if tf_critic_hidden_dim is not None else tf_hidden_dim
        tf_critic_dropout = tf_critic_dropout if tf_critic_dropout is not None else tf_dropout

        self.load_noise_std = load_noise_std
        self.learnable_noise_std = learnable_noise_std
        self.actor_obs_meta = actor_obs_meta
        self.critic_obs_meta = critic_obs_meta

        self.actor = TransformerPolicy(num_actor_obs,
                                       num_actions,
                                       mlp_hidden_dims=actor_hidden_dims,
                                       mlp_activation=activation,
                                       num_input_tokens=tf_num_input_tokens,
                                       num_layers=tf_num_layers, 
                                       d_model=tf_d_model, 
                                       hidden_dim=tf_hidden_dim, 
                                       num_heads=tf_num_heads, 
                                       dropout=tf_dropout,
                                       activation=tf_activation)

        self.critic = TransformerPolicy(num_critic_obs,
                                        1,
                                        mlp_hidden_dims=critic_hidden_dims,
                                        mlp_activation=activation,
                                        num_input_tokens=tf_critic_num_input_tokens,
                                        num_layers=tf_critic_num_layers, 
                                        d_model=tf_critic_d_model, 
                                        hidden_dim=tf_critic_hidden_dim, 
                                        num_heads=tf_critic_num_heads, 
                                        dropout=tf_critic_dropout, # NOTE: no dropout in critic
                                        activation=tf_activation)

        print(f"Actor Transformer: {self.actor}")
        print(f"Critic Transformer: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)