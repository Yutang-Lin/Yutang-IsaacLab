# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch
from isaaclab_rl.rsl_rl.modules import ActorCritic
from isaaclab_rl.rsl_rl.networks import TransformerMemoryLL
from isaaclab_rl.rsl_rl.utils import resolve_nn_activation, TensorDict


class ActorCriticTFRecurrentLL(ActorCritic):
    """Language-latent version of ActorCriticTransformer, with text and latent modalities"""
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
        tf_num_task_tokens=4,
        tf_num_history_tokens=4,
        tf_num_latent_tokens=4,
        tf_num_layers=1,
        tf_num_heads=4,
        tf_hidden_dim=256,
        tf_dropout=0.05,
        tf_activation="gelu",
        tf_lnn_dt=0.02,
        tf_lnn_tau=0.5,
        latent_kl_coef=1e-5,
        latent_recons_coef=1.0,
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
                "ActorCriticTFRecurrentML.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
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
        self.actor_obs_meta = actor_obs_meta
        self.critic_obs_meta = critic_obs_meta
        assert 'text_embeddings' in actor_obs_meta, "text_embeddings must be provided in actor_obs_meta"
        assert 'extra_conditions' in actor_obs_meta, "extra_conditions must be provided in actor_obs_meta"

        # resolve obs meta
        self.actor_proprio_ids, self.actor_text_ids, self.actor_extra_ids = self._resolve_obs_meta(num_actor_obs, actor_obs_meta)
        self.critic_proprio_ids, self.critic_text_ids, self.critic_extra_ids = self._resolve_obs_meta(num_critic_obs, critic_obs_meta)

        tf_activation = resolve_nn_activation(tf_activation)
        self.memory_a = TransformerMemoryLL(self.actor_proprio_ids.shape[0], 
                                         self.actor_text_ids.shape[0], 
                                         self.actor_extra_ids.shape[0], 
                                         num_input_tokens=tf_num_input_tokens,
                                         num_task_tokens=tf_num_task_tokens,
                                         num_history_tokens=tf_num_history_tokens,
                                         num_latent_tokens=tf_num_latent_tokens,
                                         num_layers=tf_num_layers, 
                                         d_model=tf_d_model, 
                                         hidden_dim=tf_hidden_dim, 
                                         num_heads=tf_num_heads, 
                                         dropout=tf_dropout, 
                                         activation=tf_activation,
                                         lnn_dt=tf_lnn_dt,
                                         lnn_tau=tf_lnn_tau)
        self.memory_c = TransformerMemoryLL(self.critic_proprio_ids.shape[0], 
                                         self.critic_text_ids.shape[0], 
                                         self.critic_extra_ids.shape[0], 
                                         num_input_tokens=tf_num_input_tokens,
                                         num_task_tokens=tf_num_task_tokens,
                                         num_history_tokens=tf_num_history_tokens,
                                         num_latent_tokens=tf_num_latent_tokens,
                                         num_layers=tf_num_layers, 
                                         d_model=tf_d_model, 
                                         hidden_dim=tf_hidden_dim, 
                                         num_heads=tf_num_heads, 
                                         dropout=tf_dropout, 
                                         activation=tf_activation,
                                         lnn_dt=tf_lnn_dt,
                                         lnn_tau=tf_lnn_tau)
        
        self.latent_kl_coef = latent_kl_coef
        self.latent_recons_coef = latent_recons_coef

        print(f"Actor Transformer: {self.memory_a}")
        print(f"Critic Transformer: {self.memory_c}")
        self.compute_latent_loss = False

    def extra_loss(self, **kwargs):
        kl_loss = self.memory_a.rnn._save_dict['kl_loss'] * self.latent_kl_coef
        recons_loss = self.memory_a.rnn._save_dict['recons_loss'] * self.latent_recons_coef
        self.memory_a.rnn._save_dict.clear()
        return {'latent_kl': kl_loss, 'latent_recons': recons_loss}
    
    def pre_train(self):
        self.compute_latent_loss = True

    def after_train(self):
        self.compute_latent_loss = False

    def _resolve_obs_meta(self, num_obs, obs_meta):
        all_obs = torch.arange(num_obs)
        proprio_obs = torch.ones(num_obs, dtype=torch.bool)
        text_obs = []
        extra_obs = []
        for seg in obs_meta['text_embeddings']:
            text_obs.append(all_obs[seg['start']:seg['end']].clone())
            proprio_obs[seg['start']:seg['end']] = False

        for seg in obs_meta['extra_conditions']:
            extra_obs.append(all_obs[seg['start']:seg['end']].clone())
            proprio_obs[seg['start']:seg['end']] = False

        proprio_obs = all_obs[proprio_obs].clone().contiguous()
        text_obs = torch.cat(text_obs).contiguous()
        extra_obs = torch.cat(extra_obs).contiguous()
        return proprio_obs, text_obs, extra_obs
    
    def _split_observations(self, observations: torch.Tensor):
        proprio_obs = observations[..., self.actor_proprio_ids].contiguous()
        text_obs = observations[..., self.actor_text_ids].contiguous()
        extra_obs = observations[..., self.actor_extra_ids].contiguous()
        return TensorDict(proprio=proprio_obs, text=text_obs, latent=extra_obs)
    
    def _split_critic_observations(self, observations: torch.Tensor):
        proprio_obs = observations[..., self.critic_proprio_ids].contiguous()
        text_obs = observations[..., self.critic_text_ids].contiguous()
        extra_obs = observations[..., self.critic_extra_ids].contiguous()
        return TensorDict(proprio=proprio_obs, text=text_obs, latent=extra_obs)

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None, **kwargs):
        tensor_dict = self._split_observations(observations)
        input_a = self.memory_a(tensor_dict, masks, hidden_states, compute_latent_loss=self.compute_latent_loss)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations=None,
                      proprio=None,
                      text=None,
                      extra=None,
                      **kwargs):
        if observations is not None:
            obs_dict = self._split_observations(observations)
        else:
            assert proprio is not None and text is not None
            obs_dict = TensorDict(proprio=proprio, text=text, latent=extra)
        input_a = self.memory_a(obs_dict, compute_latent_loss=self.compute_latent_loss)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None, **kwargs):
        tensor_dict = self._split_critic_observations(critic_observations)
        input_c = self.memory_c(tensor_dict, masks, hidden_states, compute_latent_loss=False, no_encode_latent=True)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states
