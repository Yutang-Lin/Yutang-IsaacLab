# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

from isaaclab_rl.rsl_rl.modules import ActorCritic
from isaaclab_rl.rsl_rl.networks import TransformerPolicyCoMeanFlow, TransformerPolicyCoMeanFlowConfig
from isaaclab_rl.rsl_rl.utils import resolve_nn_activation, TensorDict

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCriticTransformerCoMeanFlow(ActorCritic):
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
        tf_action_horizon=4,
        tf_proprio_horizon=5,
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
        flow_r_neq_t_prob=0.25,
        flow_loss_coef_p=1.0,
        flow_loss_coef_c=1e-3,
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
        self.proprio_horizon = tf_proprio_horizon
        self.control_obs_horizon = tf_control_obs_horizon
        self.action_dim = num_actions
        self.action_horizon = tf_action_horizon
        assert self.action_horizon < self.control_obs_horizon, "Action horizon must be less than control observation horizon"
        self.action_dt = 1 / tf_action_horizon
        self.control_dt = 1 / tf_control_obs_horizon
        self.sim_learning_epochs = sim_learning_epochs
        self.sim_action_loss_coef = sim_action_loss_coef
        self.sim_state_loss_coef = sim_state_loss_coef
        self.flow_r_neq_t_prob = flow_r_neq_t_prob
        self.flow_loss_coef_p = flow_loss_coef_p
        self.flow_loss_coef_c = flow_loss_coef_c
        assert actor_obs_meta is not None, "actor_obs_meta must be provided"
        assert 'proprios' in actor_obs_meta, "proprios must be provided in actor_obs_meta"
        assert 'control_observations' in actor_obs_meta, "control_observations must be provided in actor_obs_meta"

        # resolve obs meta
        self.actor_proprio_ids, self.actor_control_ids = self._resolve_obs_meta(num_actor_obs, actor_obs_meta)

        config = TransformerPolicyCoMeanFlowConfig(
            proprio_dim=self.actor_proprio_ids.shape[0],
            control_obs_dim=self.actor_control_ids.shape[0],
            action_dim=num_actions,
            proprio_horizon=tf_proprio_horizon,
            control_obs_horizon=tf_control_obs_horizon,
            action_horizon=tf_action_horizon,
            mlp_hidden_dims=actor_hidden_dims,
            mlp_activation=activation,
            d_model=tf_d_model,
            num_layers=tf_num_layers,
            num_heads=tf_num_heads,
            hidden_dim=tf_hidden_dim,
            dropout=tf_dropout,
            activation=tf_activation)
        self.actor = TransformerPolicyCoMeanFlow(config)
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
        self.mse_loss = nn.MSELoss(reduction='none')
        self.save_denoise_velocity = False
        self.denoise_buffer = dict()

    def extra_loss(self,
                   flow_state_batch=None, 
                   flow_dones_batch=None,
                   flow_actions_batch=None,
                   **kwargs):
        loss_dict = dict()
        value_dict = dict()
        
        if self.denoise_loss_coef > 0.0:
            denoise_loss = self.denoise_loss_coef * self.denoise_buffer['u_loss']
            loss_dict['denoise'] = denoise_loss
            value_dict['denoise'] = self.denoise_buffer['u_loss_value']

        if flow_state_batch is not None and self.sim_learning_epochs > 0:
            assert flow_actions_batch is not None, "Flow actions must be provided for Flow DAgger."
            assert flow_dones_batch is not None, "Flow dones must be provided for Flow DAgger."
            num_envs, num_steps = flow_state_batch.shape[:2]
            assert num_steps == self.control_obs_horizon, "Flow state must have the same number of steps as the control observation horizon."
            c_flow_mask = 1 - (torch.cumsum(flow_dones_batch, dim=1) > 0).float().view(num_envs, num_steps)
            a_flow_mask = c_flow_mask[:, :self.action_horizon]
            
            value_dict['sim_action'] = 0.0
            value_dict['sim_state'] = 0.0
            sim_action_loss = torch.zeros(1, device=flow_state_batch.device)
            sim_state_loss = torch.zeros(1, device=flow_state_batch.device)
            for _ in range(self.sim_learning_epochs):
                c_t = torch.rand(num_envs, self.control_obs_horizon, device=flow_state_batch.device)
                c_t = (c_t + (1 - c_flow_mask)).clamp(max=1.0)
                a_t = torch.rand(num_envs, self.action_horizon, device=flow_state_batch.device)
                a_t = (a_t + (1 - a_flow_mask)).clamp(max=1.0)
                _, _, a_loss, c_loss, a_loss_value, c_loss_value = self._standard_loss(
                    self.denoise_buffer['proprio'], flow_state_batch, flow_actions_batch[:, :self.action_horizon], 
                    c_t, a_t, c_flow_mask.unsqueeze(-1), a_flow_mask.unsqueeze(-1)
                )
                sim_action_loss = sim_action_loss + a_loss # type: ignore
                sim_state_loss = sim_state_loss + c_loss
                value_dict['sim_action'] += a_loss_value
                value_dict['sim_state'] += c_loss_value

            sim_action_loss /= self.sim_learning_epochs
            sim_state_loss /= self.sim_learning_epochs
            value_dict['sim_action'] /= self.sim_learning_epochs
            value_dict['sim_state'] /= self.sim_learning_epochs
            loss_dict['sim_action'] = sim_action_loss * self.sim_action_loss_coef
            loss_dict['sim_state'] = sim_state_loss * self.sim_state_loss_coef

        self.denoise_buffer.clear()
        return loss_dict, value_dict
    
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
    
    def _apply_noise(self, value: torch.Tensor, horizon: int, t: torch.Tensor):
        value = value.reshape(value.shape[0], horizon, -1)
        noise = torch.randn_like(value)
        unsqueezed_t = t.unsqueeze(-1)
        return value * (1 - unsqueezed_t) + noise * unsqueezed_t, noise
    
    def _sample_r(self, t: torch.Tensor, neq_prob: float = 0.5):
        r = torch.rand_like(t) * t
        r_eq_t = torch.rand_like(t) > neq_prob
        r[r_eq_t] = t[r_eq_t]
        return r
    
    def _standard_loss(self, proprio: torch.Tensor, 
                       control: torch.Tensor, 
                       action: torch.Tensor,
                       c_t: torch.Tensor | None = None,
                       a_t: torch.Tensor | None = None,
                       c_u_mask: torch.Tensor | None = None,
                       a_u_mask: torch.Tensor | None = None, **kwargs):
        if c_t is None:
            c_t = torch.rand(proprio.shape[0], self.control_obs_horizon, device=proprio.device, dtype=proprio.dtype)
        if a_t is None:
            a_t = torch.rand(proprio.shape[0], self.action_horizon, device=proprio.device, dtype=proprio.dtype)
        c_r = self._sample_r(c_t, self.flow_r_neq_t_prob)
        a_r = self._sample_r(a_t, self.flow_r_neq_t_prob)
        control = control.view(control.shape[0], self.control_obs_horizon, -1)
        action = action.view(action.shape[0], self.action_horizon, -1)
        c_noise = torch.randn_like(control)
        a_noise = torch.randn_like(action)
        a_u, c_u, a_u_tgt, c_u_tgt = self.actor.loss(
            proprio, 
            control, 
            action, 
            c_noise, a_noise, 
            c_r, c_t, a_r, a_t
        )
        if c_u_mask is not None:
            c_u_tgt = c_u_tgt * c_u_mask
            c_u = c_u * c_u_mask
        if a_u_mask is not None:
            a_u_tgt = a_u_tgt * a_u_mask
            a_u = a_u * a_u_mask

        # standard loss
        a_u_loss = self.mse_loss(a_u, a_u_tgt).mean(dim=-1)
        c_u_loss = self.mse_loss(c_u, c_u_tgt).mean(dim=-1) # sum over the single control dimension
        a_u_loss_value = a_u_loss.mean().item()
        c_u_loss_value = c_u_loss.mean().item()
        # adaptive loss
        a_u_loss = a_u_loss * (1 / torch.pow(a_u_loss.detach() + self.flow_loss_coef_c, self.flow_loss_coef_p))
        a_u_loss = a_u_loss.mean()
        c_u_loss = c_u_loss * (1 / torch.pow(c_u_loss.detach() + self.flow_loss_coef_c, self.flow_loss_coef_p))
        c_u_loss = c_u_loss.mean()
        return a_u, c_u, a_u_loss, c_u_loss, a_u_loss_value, c_u_loss_value
    
    def _standard_inference(self, proprio: torch.Tensor, 
                            control: torch.Tensor, 
                            action: torch.Tensor,
                            **kwargs):
        c_t = torch.linspace(self.control_dt, 1.0, self.control_obs_horizon, 
                            device=proprio.device, dtype=proprio.dtype).unsqueeze(0).repeat(proprio.shape[0], 1)
        a_t = torch.linspace(self.action_dt, 1.0, self.action_horizon, 
                            device=proprio.device, dtype=proprio.dtype).unsqueeze(0).repeat(proprio.shape[0], 1)
        control = control.view(control.shape[0], self.control_obs_horizon, -1)
        action = action.view(action.shape[0], self.action_horizon, -1)
        control, _ = self._apply_noise(control, self.control_obs_horizon, c_t)
        action, _ = self._apply_noise(action, self.action_horizon, a_t)
        # clamp the r to be between 0 and 1
        a_r = (a_t - self.action_dt).clamp(0.0, 1.0)
        c_r = (c_t - self.control_dt).clamp(0.0, 1.0)
        a_u, c_u = self.actor(proprio, control, action, c_r, c_t, a_r, a_t)
        return a_u, c_u
    
    def update_distribution(self, observations, *args, **kwargs):
        # compute mean
        mean, _ = self._standard_inference(*args, **observations, **kwargs)
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
        self.update_distribution(obs_dict)
        return self.distribution.sample()

    def act_inference(self, observations, **kwargs):
        obs_dict = self._split_observations(observations)
        if self.save_denoise_velocity:
            self.denoise_buffer['proprio'] = obs_dict['proprio']
        return torch.zeros(observations.shape[0], self.action_dim, device=observations.device).requires_grad_(True) # type: ignore

    def evaluate(self, critic_observations, **kwargs):
        raise NotImplementedError