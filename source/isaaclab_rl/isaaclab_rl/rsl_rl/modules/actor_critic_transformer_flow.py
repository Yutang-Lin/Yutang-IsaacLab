# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

from isaaclab_rl.rsl_rl.modules import ActorCritic
from isaaclab_rl.rsl_rl.networks import TransformerPolicyFlow, TransformerPolicyFlowConfig
from isaaclab_rl.rsl_rl.utils import resolve_nn_activation, TensorDict

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCriticTransformerFlow(ActorCritic):
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
        tf_num_proprio_tokens=4,
        tf_num_action_tokens=4,
        tf_control_obs_horizon=20,
        tf_num_layers=1,
        tf_num_heads=4,
        tf_hidden_dim=256,
        tf_dropout=0.05,
        tf_activation="gelu",
        denoise_loss_coef=1.0,
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
                "ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        nn.Module.__init__(self)
        activation = resolve_nn_activation(activation)
        tf_activation = resolve_nn_activation(tf_activation)

        self.load_noise_std = load_noise_std
        self.learnable_noise_std = learnable_noise_std
        self.actor_obs_meta = actor_obs_meta
        self.control_obs_horizon = tf_control_obs_horizon
        assert actor_obs_meta is not None, "actor_obs_meta must be provided"
        assert 'proprios' in actor_obs_meta, "proprios must be provided in actor_obs_meta"
        assert 'control_observations' in actor_obs_meta, "control_observations must be provided in actor_obs_meta"

        # resolve obs meta
        self.actor_proprio_ids, self.actor_control_ids = self._resolve_obs_meta(num_actor_obs, actor_obs_meta)

        config = TransformerPolicyFlowConfig(
            proprio_dim=self.actor_proprio_ids.shape[0],
            control_obs_dim=self.actor_control_ids.shape[0],
            action_dim=num_actions,
            control_obs_horizon=tf_control_obs_horizon,
            mlp_hidden_dims=actor_hidden_dims,
            mlp_activation=activation,
            num_proprio_tokens=tf_num_proprio_tokens,
            num_action_tokens=tf_num_action_tokens,
            d_model=tf_d_model,
            num_layers=tf_num_layers,
            num_heads=tf_num_heads,
            hidden_dim=tf_hidden_dim,
            dropout=tf_dropout,
            activation=tf_activation)
        self.actor = TransformerPolicyFlow(config)
        self.critic = None

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

        # denoise loss
        self.denoise_loss_coef = denoise_loss_coef
        self.mse_loss = nn.MSELoss()
        self.save_denoise_velocity = False
        self.denoise_buffer = dict()

    def extra_loss(self, **kwargs):
        denoise_loss = self.denoise_loss_coef * self.mse_loss(self.denoise_buffer['velocity'], self.denoise_buffer['control_obs_velocity'])
        self.denoise_buffer.clear()
        return {"denoise_loss": denoise_loss}
    
    def pre_train(self):
        self.save_denoise_velocity = True

    def after_train(self):
        self.save_denoise_velocity = False
        self.denoise_buffer.clear()

    def _resolve_obs_meta(self, num_obs, obs_meta):
        all_obs = torch.arange(num_obs)
        proprio_obs = torch.ones(num_obs, dtype=torch.bool)
        control_obs = []
        for seg in obs_meta['control_observations']:
            control_obs.append(all_obs[seg['start']:seg['end']].clone())
            proprio_obs[seg['start']:seg['end']] = False

        proprio_obs = all_obs[proprio_obs].clone().contiguous()
        control_obs = torch.cat(control_obs).contiguous()
        return proprio_obs, control_obs
    
    def _split_observations(self, observations: torch.Tensor):
        proprio_obs = observations[..., self.actor_proprio_ids].contiguous()
        control_obs = observations[..., self.actor_control_ids].contiguous()
        return TensorDict(proprio=proprio_obs, control=control_obs)
    
    def _split_critic_observations(self, observations: torch.Tensor):
        raise NotImplementedError
    
    def _apply_noise(self, control: torch.Tensor, timestep: torch.Tensor):
        control = control.reshape(control.shape[0], self.control_obs_horizon, -1)
        noise = torch.randn_like(control)
        velocity = control - noise
        return control * timestep.unsqueeze(-1) + noise * (1 - timestep.unsqueeze(-1)), noise, velocity
    
    def update_distribution(self, observations, *args, **kwargs):
        # compute mean
        mean, _ = self.actor(*args, **observations, **kwargs)
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
        obs_dict = self._split_observations(observations)
        obs_dict['timestep'] = torch.linspace(1, 0, self.control_obs_horizon, device=observations.device).unsqueeze(0).repeat(observations.shape[0], 1)
        obs_dict['control'], _, _ = self._apply_noise(obs_dict['control'], obs_dict['timestep'])
        self.update_distribution(obs_dict)
        return self.distribution.sample()

    def act_inference(self, observations, **kwargs):
        obs_dict = self._split_observations(observations)
        if self.save_denoise_velocity:
            obs_dict['timestep'] = torch.rand(observations.shape[0], self.control_obs_horizon, device=observations.device)
        else:
            obs_dict['timestep'] = torch.linspace(1, 0, self.control_obs_horizon, device=observations.device).unsqueeze(0).repeat(observations.shape[0], 1)
        obs_dict['control'], _, self.denoise_buffer['velocity'] = self._apply_noise(obs_dict['control'], obs_dict['timestep'])

        actions_mean, control_obs_velocity = self.actor(**obs_dict, **kwargs)
        if self.save_denoise_velocity:
            self.denoise_buffer['control_obs_velocity'] = control_obs_velocity
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        raise NotImplementedError