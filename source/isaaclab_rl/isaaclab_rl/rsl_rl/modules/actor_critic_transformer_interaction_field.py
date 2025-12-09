# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

from isaaclab_rl.rsl_rl.modules import ActorCritic
from isaaclab_rl.rsl_rl.networks import TransformerPolicyInteractionField, TransformerPolicy
from isaaclab_rl.rsl_rl.utils import resolve_nn_activation, TensorDict

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCriticTransformerInteractionField(ActorCritic):
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
        tf_num_fusion_heads=4,
        tf_num_layers=1,
        tf_num_heads=4,
        tf_hidden_dim=256,
        tf_dropout=0.05,
        tf_activation="gelu",
        init_noise_std=1.0,
        load_noise_std: bool = True,
        learnable_noise_std: bool = True,
        noise_std_type: str = "scalar",
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
        residual: bool = False,
        actor_obs_meta: dict = None, # type: ignore
        critic_obs_meta: dict = None, # type: ignore
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        nn.Module.__init__(self)
        activation = resolve_nn_activation(activation)
        tf_activation = resolve_nn_activation(tf_activation)

        self.load_noise_std = load_noise_std
        self.learnable_noise_std = learnable_noise_std
        self.actor_obs_meta = actor_obs_meta
        self.critic_obs_meta = critic_obs_meta
        assert 'task_condition' in actor_obs_meta, "task_condition must be provided in actor_obs_meta"
        assert 'interaction_field' in actor_obs_meta, "interaction_field must be provided in actor_obs_meta"
        assert 'movement_goal' in actor_obs_meta, "movement_goal must be provided in actor_obs_meta"

        # resolve obs meta
        self.actor_proprio_ids, self.actor_task_condition_ids, self.actor_interaction_field_ids, self.actor_movement_goal_ids = self._resolve_obs_meta(num_actor_obs, actor_obs_meta)

        self.actor = TransformerPolicyInteractionField(self.actor_proprio_ids.shape[0],
                                                        self.actor_interaction_field_ids.shape[0],
                                                        self.actor_movement_goal_ids.shape[0],
                                                        self.actor_task_condition_ids.shape[0],
                                                        num_actions,
                                                        num_fusion_heads=tf_num_fusion_heads,
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
                                        num_input_tokens=tf_num_input_tokens,
                                        num_layers=tf_num_layers, 
                                        d_model=tf_d_model, 
                                        hidden_dim=tf_hidden_dim, 
                                        num_heads=tf_num_heads, 
                                        dropout=tf_dropout,
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

    def _resolve_obs_meta(self, num_obs, obs_meta):
        all_obs = torch.arange(num_obs)
        proprio_obs = torch.ones(num_obs, dtype=torch.bool)
        task_condition_obs = []
        interaction_field_obs = []
        movement_goal_obs = []
        for seg in obs_meta['task_condition']:
            task_condition_obs.append(all_obs[seg['start']:seg['end']].clone())
            proprio_obs[seg['start']:seg['end']] = False
        for seg in obs_meta['interaction_field']:
            interaction_field_obs.append(all_obs[seg['start']:seg['end']].clone())
            proprio_obs[seg['start']:seg['end']] = False
        for seg in obs_meta['movement_goal']:
            movement_goal_obs.append(all_obs[seg['start']:seg['end']].clone())
            proprio_obs[seg['start']:seg['end']] = False

        proprio_obs = all_obs[proprio_obs].clone().contiguous()
        task_condition_obs = torch.cat(task_condition_obs).contiguous()
        interaction_field_obs = torch.cat(interaction_field_obs).contiguous()
        movement_goal_obs = torch.cat(movement_goal_obs).contiguous()
        return proprio_obs, task_condition_obs, interaction_field_obs, movement_goal_obs
    
    def _split_observations(self, observations: torch.Tensor):
        proprio_obs = observations[..., self.actor_proprio_ids].contiguous()
        task_condition_obs = observations[..., self.actor_task_condition_ids].contiguous()
        interaction_field_obs = observations[..., self.actor_interaction_field_ids].contiguous()
        movement_goal_obs = observations[..., self.actor_movement_goal_ids].contiguous()
        return TensorDict(proprio=proprio_obs, task_condition=task_condition_obs, interaction_field=interaction_field_obs, movement_goal=movement_goal_obs)
    
    def update_distribution(self, observations, **kwargs):
        # compute mean
        mean = self.actor(**observations, **kwargs)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        if not self.learnable_noise_std:
            std = std.detach()

        nan_std = torch.isnan(std).any(dim=1).nonzero().squeeze(-1)
        if len(nan_std) > 0:
            print(f"Found {len(nan_std)} NaN stds")
            std = std.clone()
            std[nan_std] = 1e-3
        std = std.clamp(min=1e-3, max=10.0)
        
        # create distribution
        self.distribution = Normal(mean, std)
    
    def act(self, observations, **kwargs):
        self.update_distribution(self._split_observations(observations))
        return self.distribution.sample()

    def act_inference(self, observations=None,
                      proprio=None,
                      task_condition=None,
                      interaction_field=None,
                      movement_goal=None,
                      **kwargs):
        if observations is not None:
            obs_dict = self._split_observations(observations)
        else:
            assert proprio is not None and task_condition is not None and interaction_field is not None and movement_goal is not None
            obs_dict = TensorDict(proprio=proprio, task_condition=task_condition, interaction_field=interaction_field, movement_goal=movement_goal)
        actions_mean = self.actor(**obs_dict)
        return actions_mean