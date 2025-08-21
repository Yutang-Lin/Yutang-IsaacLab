# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

from isaaclab_rl.rsl_rl.modules import ActorCritic
from isaaclab_rl.rsl_rl.networks import TransformerPolicyLatent
from isaaclab_rl.rsl_rl.utils import resolve_nn_activation, TensorDict

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCriticTransformerLatent(ActorCritic):
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
        tf_num_latent_tokens=4,
        tf_num_layers=1,
        tf_num_heads=4,
        tf_hidden_dim=256,
        tf_dropout=0.05,
        tf_activation="gelu",
        latent_kl_coef=1e-5,
        latent_recons_coef=1.0,
        latent_stable_coef=1e-3,
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

        self.load_noise_std = load_noise_std
        self.learnable_noise_std = learnable_noise_std
        self.actor_obs_meta = actor_obs_meta
        self.critic_obs_meta = critic_obs_meta
        assert 'conditions' in actor_obs_meta, "conditions must be provided in actor_obs_meta"
        self.compute_latent_loss = False

        self.latent_kl_coef = latent_kl_coef
        self.latent_recons_coef = latent_recons_coef
        self.latent_stable_coef = latent_stable_coef
        self.compute_stable_loss = self.latent_stable_coef > 0

        # resolve obs meta
        self.actor_proprio_ids, self.actor_condition_ids = self._resolve_obs_meta(num_actor_obs, actor_obs_meta)
        self.critic_proprio_ids, self.critic_condition_ids = self._resolve_obs_meta(num_critic_obs, critic_obs_meta)

        self.actor = TransformerPolicyLatent(self.actor_proprio_ids.shape[0],
                                            self.actor_condition_ids.shape[0],
                                            num_actions,
                                            mlp_hidden_dims=actor_hidden_dims,
                                            mlp_activation=activation,
                                            num_input_tokens=tf_num_input_tokens,
                                            num_latent_tokens=tf_num_latent_tokens,
                                            num_layers=tf_num_layers, 
                                            d_model=tf_d_model, 
                                            hidden_dim=tf_hidden_dim, 
                                            num_heads=tf_num_heads, 
                                            dropout=tf_dropout,
                                            activation=tf_activation)

        self.critic = TransformerPolicyLatent(self.critic_proprio_ids.shape[0],
                                            self.critic_condition_ids.shape[0],
                                            1,
                                            mlp_hidden_dims=critic_hidden_dims,
                                            mlp_activation=activation,
                                            num_input_tokens=tf_num_input_tokens,
                                            num_latent_tokens=tf_num_latent_tokens,
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

    def extra_loss(self, **kwargs):
        loss_dict = {}
        kl_loss = self.actor._save_dict['kl_loss'] * self.latent_kl_coef
        recons_loss = self.actor._save_dict['recons_loss'] * self.latent_recons_coef
        loss_dict['latent_kl'] = kl_loss
        loss_dict['latent_recons'] = recons_loss
        if self.compute_stable_loss:
            loss_dict['latent_stable'] = self.actor._save_dict['stable_loss'] * self.latent_stable_coef
        self.actor._save_dict.clear()
        return loss_dict
    
    def pre_train(self):
        self.compute_latent_loss = True

    def after_train(self):
        self.compute_latent_loss = False

    def _resolve_obs_meta(self, num_obs, obs_meta):
        all_obs = torch.arange(num_obs)
        proprio_obs = torch.ones(num_obs, dtype=torch.bool)
        condition_obs = []
        for seg in obs_meta['conditions']:
            condition_obs.append(all_obs[seg['start']:seg['end']].clone())
            proprio_obs[seg['start']:seg['end']] = False

        proprio_obs = all_obs[proprio_obs].clone().contiguous()
        condition_obs = torch.cat(condition_obs).contiguous()
        return proprio_obs, condition_obs
    
    def _split_observations(self, observations: torch.Tensor):
        proprio_obs = observations[..., self.actor_proprio_ids].contiguous()
        condition_obs = observations[..., self.actor_condition_ids].contiguous()
        return TensorDict(proprio=proprio_obs, condition=condition_obs)
    
    def _split_critic_observations(self, observations: torch.Tensor):
        proprio_obs = observations[..., self.critic_proprio_ids].contiguous()
        condition_obs = observations[..., self.critic_condition_ids].contiguous()
        return TensorDict(proprio=proprio_obs, condition=condition_obs)
    
    def update_distribution(self, observations, *args, **kwargs):
        # compute mean
        mean = self.actor(observations, *args, **kwargs)
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
        self.update_distribution(self._split_observations(observations), 
                                 compute_latent_loss=self.compute_latent_loss,
                                 compute_stable_loss=self.compute_stable_loss and self.compute_latent_loss,
                                 apply_vae_noise=True)
        return self.distribution.sample()

    def act_inference(self, observations=None,
                      proprio=None,
                      condition=None,
                      latent=None,
                      apply_vae_noise=False,
                      return_records=False,
                      **kwargs):
        if observations is not None:
            obs_dict = self._split_observations(observations)
        else:
            assert proprio is not None and latent is not None
            obs_dict = TensorDict(proprio=proprio, condition=condition, latent=latent)
        actions_mean = self.actor(obs_dict, compute_latent_loss=self.compute_latent_loss,
                                compute_stable_loss=self.compute_stable_loss and self.compute_latent_loss,
                                apply_vae_noise=apply_vae_noise,
                                return_latent=return_records)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        tensor_dict = self._split_critic_observations(critic_observations)
        return self.critic(tensor_dict, compute_latent_loss=False,
                                compute_stable_loss=False,
                                apply_vae_noise=False)