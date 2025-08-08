# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch
from isaaclab_rl.rsl_rl.modules import ActorCritic
from isaaclab_rl.rsl_rl.networks import TransformerMemoryML
from isaaclab_rl.rsl_rl.utils import resolve_nn_activation, TensorDict


class ActorCriticMLTransformer(ActorCritic):
    """Motion-language version of ActorCriticTransformer, with text and ref motion modalities"""
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
                "ActorCriticMLTransformer.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
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
        assert 'motion_reference' in actor_obs_meta, "motion_reference must be provided in actor_obs_meta"
        assert 'task_blend_mask' in actor_obs_meta, "task_blend_mask must be provided in actor_obs_meta"

        # resolve obs meta
        self.actor_proprio_ids, self.actor_text_ids, self.actor_motion_ids, self.actor_blend_mask_ids = self._resolve_obs_meta(num_actor_obs, actor_obs_meta)
        self.critic_proprio_ids, self.critic_text_ids, self.critic_motion_ids, self.critic_blend_mask_ids = self._resolve_obs_meta(num_critic_obs, critic_obs_meta)

        tf_activation = resolve_nn_activation(tf_activation)
        self.memory_a = TransformerMemoryML(self.actor_proprio_ids.shape[0], 
                                         self.actor_text_ids.shape[0], 
                                         self.actor_motion_ids.shape[0], 
                                         num_input_tokens=tf_num_input_tokens,
                                         num_task_tokens=tf_num_task_tokens,
                                         num_history_tokens=tf_num_history_tokens,
                                         num_layers=tf_num_layers, 
                                         d_model=tf_d_model, 
                                         hidden_dim=tf_hidden_dim, 
                                         num_heads=tf_num_heads, 
                                         dropout=tf_dropout, 
                                         activation=tf_activation,
                                         lnn_dt=tf_lnn_dt,
                                         lnn_tau=tf_lnn_tau)
        self.memory_c = TransformerMemoryML(self.critic_proprio_ids.shape[0], 
                                         self.critic_text_ids.shape[0], 
                                         self.critic_motion_ids.shape[0], 
                                         num_input_tokens=tf_num_input_tokens,
                                         num_task_tokens=tf_num_task_tokens,
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
        self.compute_align_loss = False

    def extra_loss(self, **kwargs):
        hidden_align = self.memory_a.rnn._save_dict['hidden_align']
        task_align = self.memory_a.rnn._save_dict['task_align']
        self.memory_a.rnn._save_dict.clear()
        return {'hidden_align': hidden_align, 'task_align': task_align}
    
    def pre_train(self):
        self.compute_align_loss = True

    def after_train(self):
        self.compute_align_loss = False

    def _resolve_obs_meta(self, num_obs, obs_meta):
        all_obs = torch.arange(num_obs)
        proprio_obs = torch.ones(num_obs, dtype=torch.bool)
        text_obs = []
        motion_obs = []
        blend_mask_obs = []
        for seg in obs_meta['text_embeddings']:
            text_obs.append(all_obs[seg['start']:seg['end']].clone())
            proprio_obs[seg['start']:seg['end']] = False

        for seg in obs_meta['motion_reference']:
            motion_obs.append(all_obs[seg['start']:seg['end']].clone())
            proprio_obs[seg['start']:seg['end']] = False
        
        for seg in obs_meta['task_blend_mask']:
            blend_mask_obs.append(all_obs[seg['start']:seg['end']].clone())
            proprio_obs[seg['start']:seg['end']] = False

        proprio_obs = all_obs[proprio_obs].clone().contiguous()
        text_obs = torch.cat(text_obs).contiguous()
        motion_obs = torch.cat(motion_obs).contiguous()
        blend_mask_obs = torch.cat(blend_mask_obs).contiguous()
        return proprio_obs, text_obs, motion_obs, blend_mask_obs
    
    def _split_observations(self, observations: torch.Tensor):
        proprio_obs = observations[..., self.actor_proprio_ids].contiguous()
        text_obs = observations[..., self.actor_text_ids].contiguous()
        motion_obs = observations[..., self.actor_motion_ids].contiguous()
        blend_mask_obs = observations[..., self.actor_blend_mask_ids].contiguous()
        return TensorDict(proprio=proprio_obs, text=text_obs, motion=motion_obs, blend_mask=blend_mask_obs)
    
    def _split_critic_observations(self, observations: torch.Tensor):
        proprio_obs = observations[..., self.critic_proprio_ids].contiguous()
        text_obs = observations[..., self.critic_text_ids].contiguous()
        motion_obs = observations[..., self.critic_motion_ids].contiguous()
        blend_mask_obs = observations[..., self.critic_blend_mask_ids].contiguous()
        return TensorDict(proprio=proprio_obs, text=text_obs, motion=motion_obs, blend_mask=blend_mask_obs)

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(self._split_observations(observations), masks, hidden_states, compute_align_loss=self.compute_align_loss)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations=None,
                      proprio=None,
                      text=None,
                      motion=None,
                      blend_mask=None):
        if observations is not None:
            obs_dict = self._split_observations(observations)
        else:
            obs_dict = TensorDict(proprio=proprio, text=text, motion=motion, blend_mask=blend_mask)
        input_a = self.memory_a(obs_dict, compute_align_loss=self.compute_align_loss)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(self._split_critic_observations(critic_observations), masks, hidden_states,
                                use_all_task_tokens=True)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states
