# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

from isaaclab_rl.rsl_rl.modules import ActorCritic
from isaaclab_rl.rsl_rl.networks import TransformerPolicyDDIM, TransformerPolicyDDIMConfig
from isaaclab_rl.rsl_rl.utils import resolve_nn_activation, TensorDict

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCriticTransformerDDIM(ActorCritic):
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
        sim_learning_epochs=1,
        sim_action_loss_coef=1.0,
        sim_state_loss_coef=1.0,
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
        self.sim_learning_epochs = sim_learning_epochs
        self.sim_action_loss_coef = sim_action_loss_coef
        self.sim_state_loss_coef = sim_state_loss_coef
        assert actor_obs_meta is not None, "actor_obs_meta must be provided"
        assert 'proprios' in actor_obs_meta, "proprios must be provided in actor_obs_meta"
        assert 'control_observations' in actor_obs_meta, "control_observations must be provided in actor_obs_meta"

        # resolve obs meta
        self.actor_proprio_ids, self.actor_control_ids = self._resolve_obs_meta(num_actor_obs, actor_obs_meta)

        config = TransformerPolicyDDIMConfig(
            proprio_dim=self.actor_proprio_ids.shape[0],
            control_obs_dim=self.actor_control_ids.shape[0],
            action_dim=num_actions,
            control_obs_horizon=tf_control_obs_horizon,
            mlp_hidden_dims=actor_hidden_dims,
            mlp_activation=activation, # type: ignore
            num_proprio_tokens=tf_num_proprio_tokens,
            num_action_tokens=tf_num_action_tokens,
            d_model=tf_d_model,
            num_layers=tf_num_layers,
            num_heads=tf_num_heads,
            hidden_dim=tf_hidden_dim,
            dropout=tf_dropout,
            activation=tf_activation) # type: ignore
        self.actor = TransformerPolicyDDIM(config)
        self.critic = None

        print(f"Actor Transformer: {self.actor}")
        print(f"Critic Transformer: {self.critic}")

        # timestep step size
        self.control_dt = int(self.actor.num_timesteps / self.control_obs_horizon)

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

    def extra_loss(self, student_actions_batch=None, flow_state_batch=None, flow_dones_batch=None, **kwargs):
        loss_dict = dict()
        
        denoise_loss = self.denoise_loss_coef * self.mse_loss(
            self.denoise_buffer['control'], 
            self.denoise_buffer['clean_control']
        )
        loss_dict['denoise'] = denoise_loss

        if flow_state_batch is not None and self.sim_learning_epochs > 0:
            assert student_actions_batch is not None, "Student actions must be provided for Flow DAgger."
            assert flow_dones_batch is not None, "Flow dones must be provided for Flow DAgger."
            num_envs, num_steps = flow_state_batch.shape[:2]
            assert num_steps == self.control_obs_horizon, "Flow state must have the same number of steps as the control observation horizon."
            flow_mask = (torch.cumsum(flow_dones_batch, dim=1) > 0).float().view(num_envs, num_steps)
            state_loss_weight = (1 - flow_mask.float()).unsqueeze(-1) * (1 / num_steps)
            
            sim_action_loss = torch.zeros(1, device=flow_state_batch.device)
            sim_state_loss = torch.zeros(1, device=flow_state_batch.device)
            for _ in range(self.sim_learning_epochs):
                timestep = torch.rand(num_envs, self.control_obs_horizon, device=flow_state_batch.device)
                timestep = (timestep + flow_mask).clamp(max=1.0)
                control = self._apply_noise(flow_state_batch, timestep)
                actions, clean_control = self.actor(self.denoise_buffer['proprio'], control, timestep)
                action_loss = self.sim_action_loss_coef * (
                    actions - student_actions_batch
                ).square()
                sim_action_loss = sim_action_loss + action_loss.mean()

                state_loss = self.sim_state_loss_coef * (
                    clean_control - flow_state_batch
                ).square() * state_loss_weight
                sim_state_loss = sim_state_loss + state_loss.mean()

            sim_action_loss /= self.sim_learning_epochs
            sim_state_loss /= self.sim_learning_epochs
            loss_dict['sim_action'] = sim_action_loss * self.sim_action_loss_coef
            loss_dict['sim_state'] = sim_state_loss * self.sim_state_loss_coef

        self.denoise_buffer.clear()
        return loss_dict
    
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
        return self.actor.apply_noise(control, timestep)
    
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
        obs_dict['timestep'] = torch.linspace(self.control_dt, self.actor.num_timesteps - 1, self.control_obs_horizon, device=observations.device).long().unsqueeze(0).repeat(observations.shape[0], 1)
        obs_dict['control'] = self._apply_noise(obs_dict['control'], obs_dict['timestep'])
        self.update_distribution(obs_dict)
        return self.distribution.sample()

    def act_inference(self, observations, **kwargs):
        obs_dict = self._split_observations(observations)
        if self.save_denoise_velocity:
            obs_dict['timestep'] = torch.randint(0, self.actor.num_timesteps, (observations.shape[0], self.control_obs_horizon), device=observations.device).long()
            self.denoise_buffer['control'] = obs_dict['control']
        else:
            obs_dict['timestep'] = torch.linspace(self.control_dt, self.actor.num_timesteps - 1, self.control_obs_horizon, device=observations.device).long().unsqueeze(0).repeat(observations.shape[0], 1)
        obs_dict['control'] = self._apply_noise(obs_dict['control'], obs_dict['timestep'])

        actions_mean, clean_control = self.actor(**obs_dict, **kwargs)
        if self.save_denoise_velocity:
            self.denoise_buffer['proprio'] = obs_dict['proprio']
            self.denoise_buffer['clean_control'] = clean_control
            self.denoise_buffer['control'] = self.denoise_buffer['control'].reshape(*clean_control.shape)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        raise NotImplementedError