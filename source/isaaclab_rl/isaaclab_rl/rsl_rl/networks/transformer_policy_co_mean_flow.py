# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from .transformer import (
    MLP, 
    TransformerEncoder, 
    PositionalEncoding, 
    SinusoidalTimestepEmbedder
)
import torch.autograd.forward_ad as fwAD
from functools import partial
from transformers import PreTrainedModel, PretrainedConfig # type: ignore

class TransformerPolicyCoMeanFlowConfig(PretrainedConfig):
    def __init__(self,
                 *,
                 proprio_dim,
                 control_obs_dim,
                 action_dim,
                 proprio_horizon,
                 control_obs_horizon,
                 action_horizon,
                 mlp_hidden_dims,
                 mlp_activation,
                 d_model,
                 num_layers,
                 num_heads, 
                 hidden_dim,
                 dropout,
                 activation,
                 enable_sdpa: bool = True):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.control_obs_dim = control_obs_dim
        self.action_dim = action_dim
        self.proprio_horizon = proprio_horizon
        self.control_obs_horizon = control_obs_horizon
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.action_horizon = action_horizon
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = activation
        self.enable_sdpa = enable_sdpa

class TransformerPolicyCoMeanFlow(PreTrainedModel):
    def __init__(self, 
                 config: TransformerPolicyCoMeanFlowConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.input_size = config.proprio_dim
        self.in_features = config.proprio_dim
        self.out_features = config.action_dim
        self.action_horizon = config.action_horizon

        self.proprio_dim = config.proprio_dim
        self.control_obs_dim = config.control_obs_dim
        self.proprio_horizon = config.proprio_horizon
        self.control_obs_horizon = config.control_obs_horizon
        self.single_proprio_dim = self.proprio_dim // self.proprio_horizon
        self.single_control_obs_dim = self.control_obs_dim // self.control_obs_horizon
        self.action_dim = config.action_dim
        self.mlp_hidden_dims = config.mlp_hidden_dims
        self.mlp_activation = config.mlp_activation

        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.hidden_size = config.d_model
        self.dropout = config.dropout
        self.activation = config.activation
        self.enable_sdpa = config.enable_sdpa

        # projections and transformer
        self.proprio_proj = nn.Linear(self.single_proprio_dim, self.d_model)
        self.control_obs_proj = nn.Linear(self.single_control_obs_dim, self.d_model)
        self.action_proj = nn.Linear(self.action_dim, self.d_model)
        self.model = TransformerEncoder(self.d_model, 
                                       self.num_heads, 
                                       self.hidden_dim, 
                                       self.num_layers, 
                                       self.dropout, 
                                       is_causal=False, 
                                       activation=self.activation,
                                       enable_sdpa=self.enable_sdpa)
        self.action_output_proj = MLP(self.d_model, self.mlp_hidden_dims, self.action_dim, self.mlp_activation)
        self.control_obs_output_proj = MLP(self.d_model, self.mlp_hidden_dims, self.single_control_obs_dim, self.mlp_activation)

        # embeddings
        self.proprio_embedding = nn.Parameter(torch.randn(1, self.proprio_horizon, self.d_model))
        self.action_embedding = nn.Parameter(torch.randn(1, self.action_horizon, self.d_model))
        self.control_obs_embedding = nn.Parameter(torch.randn(1, self.control_obs_horizon, self.d_model))
        self.t_embedding = SinusoidalTimestepEmbedder(self.d_model)
        self.t_minus_r_embedding = SinusoidalTimestepEmbedder(self.d_model)

        # initial tokens
        self.initial_action_tokens = nn.Parameter(torch.randn(1, self.action_horizon, self.d_model))

        # attention masks
        total_tokens = self.proprio_horizon + self.action_horizon + self.control_obs_horizon
        self.register_buffer('attn_mask', torch.ones(total_tokens, total_tokens, dtype=torch.bool))
        self.attn_mask: torch.Tensor # type hint
        for i in range(self.action_horizon):
            self.attn_mask[self.proprio_horizon + i, self.proprio_horizon + self.action_horizon + i:] = False

    def forward(self, proprio: torch.Tensor, 
                control: torch.Tensor,
                action: torch.Tensor,
                c_r: torch.Tensor, 
                c_t: torch.Tensor,
                a_r: torch.Tensor,
                a_t: torch.Tensor,
                fwd_dual: bool = False):
        batch_size = proprio.shape[0]
        proprio_tokens = self.proprio_proj(proprio.view(batch_size, self.proprio_horizon, self.single_proprio_dim)) + self.proprio_embedding
        control_obs_tokens = control.view(batch_size, self.control_obs_horizon, self.single_control_obs_dim)
        control_timestep_embedding = self.t_embedding(c_t) + self.t_minus_r_embedding(c_t - c_r)
        action_tokens = self.action_proj(action.view(batch_size, self.action_horizon, self.action_dim)) + self.action_embedding + self.t_embedding(a_t) + self.t_minus_r_embedding(a_t - a_r)
        action_timestep_embedding = self.t_embedding(a_t) + self.t_minus_r_embedding(a_t - a_r)
        control_obs_tokens = self.control_obs_proj(control_obs_tokens) + self.control_obs_embedding + control_timestep_embedding
        action_tokens = action_tokens + self.action_embedding + action_timestep_embedding

        input = torch.cat([proprio_tokens, action_tokens, control_obs_tokens], dim=1)
        input = self.model(input, attn_mask=self.attn_mask.unsqueeze(0).repeat(batch_size, 1, 1), fwd_dual=fwd_dual)
        actions_velocity = self.action_output_proj(input[:, self.proprio_horizon:self.proprio_horizon + self.action_horizon])
        control_obs_velocity = self.control_obs_output_proj(input[:, self.proprio_horizon + self.action_horizon:])
        return actions_velocity, control_obs_velocity
    
    def loss(self, proprio: torch.Tensor, 
             control: torch.Tensor,
             action: torch.Tensor,
             c_noise: torch.Tensor, 
             a_noise: torch.Tensor,
             c_r: torch.Tensor, 
             c_t: torch.Tensor,
             a_r: torch.Tensor,
             a_t: torch.Tensor):
        # compute noised control and action
        unsqueeze_c_t = c_t.unsqueeze(-1)
        noised_control = control * (1 - unsqueeze_c_t) + c_noise * unsqueeze_c_t
        c_velocity = c_noise - control
        unsqueeze_a_t = a_t.unsqueeze(-1)
        noised_action = action * (1 - unsqueeze_a_t) + a_noise * unsqueeze_a_t
        a_velocity = a_noise - action
        # make everything with grad
        proprio = proprio.clone().requires_grad_(True)
        noised_control = noised_control.clone().requires_grad_(True)
        noised_action = noised_action.clone().requires_grad_(True)
        c_r = c_r.clone().requires_grad_(True)
        c_t = c_t.clone().requires_grad_(True)
        a_r = a_r.clone().requires_grad_(True)
        a_t = a_t.clone().requires_grad_(True)
        # calculate jacobian-vector product
        (a_u, c_u), (a_dudt, c_dudt) = torch.func.jvp(partial(self.forward, proprio, fwd_dual=True),  # type: ignore
            (noised_control, noised_action, c_r, c_t, a_r, a_t), 
            ( # type: ignore
                c_velocity,
                a_velocity,
                torch.zeros_like(c_r),
                torch.ones_like(c_t),
                torch.zeros_like(a_r),
                torch.ones_like(a_t),
            ), has_aux=False
        )
        # compute the target velocity
        a_u_tgt = a_velocity - a_dudt.detach() * (a_t - a_r).unsqueeze(-1)
        c_u_tgt = c_velocity - c_dudt.detach() * (c_t - c_r).unsqueeze(-1)
        return a_u, c_u, a_u_tgt, c_u_tgt
    
    def denoise(self, proprio: torch.Tensor, 
                control: torch.Tensor, 
                action: torch.Tensor,
                c_r: torch.Tensor, 
                c_t: torch.Tensor,
                a_r: torch.Tensor,
                a_t: torch.Tensor):
        a_u, c_u = self.forward(proprio, control, action, c_r, c_t, a_r, a_t)
        control = control - c_u * (c_t - c_r).unsqueeze(-1)
        action = action - a_u * (a_t - a_r).unsqueeze(-1)
        return action, control