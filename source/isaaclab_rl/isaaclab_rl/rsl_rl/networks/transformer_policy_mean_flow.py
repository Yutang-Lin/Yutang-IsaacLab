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

class TransformerPolicyMeanFlowConfig(PretrainedConfig):
    def __init__(self,
                 *,
                 proprio_dim,
                 control_obs_dim,
                 action_dim,
                 control_obs_horizon,
                 mlp_hidden_dims,
                 mlp_activation,
                 num_proprio_tokens,
                 num_action_tokens,
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
        self.control_obs_horizon = control_obs_horizon
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.num_proprio_tokens = num_proprio_tokens
        self.num_action_tokens = num_action_tokens
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = activation
        self.enable_sdpa = enable_sdpa

class TransformerPolicyMeanFlow(PreTrainedModel):
    def __init__(self, 
                 config: TransformerPolicyMeanFlowConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.input_size = config.proprio_dim
        self.in_features = config.proprio_dim
        self.out_features = config.action_dim
        self.num_proprio_tokens = config.num_proprio_tokens
        self.num_action_tokens = config.num_action_tokens

        self.proprio_dim = config.proprio_dim
        self.control_obs_dim = config.control_obs_dim
        self.control_obs_horizon = config.control_obs_horizon
        self.single_control_obs_dim = self.control_obs_dim // self.control_obs_horizon
        self.action_dim = config.action_dim
        self.mlp_hidden_dims = config.mlp_hidden_dims
        self.mlp_activation = config.mlp_activation
        self.num_proprio_tokens = config.num_proprio_tokens
        self.num_action_tokens = config.num_action_tokens

        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.hidden_size = config.d_model
        self.dropout = config.dropout
        self.activation = config.activation
        self.enable_sdpa = config.enable_sdpa

        # projections and transformer
        self.proprio_proj = nn.Linear(self.proprio_dim, self.num_proprio_tokens * self.d_model)
        self.control_obs_proj = nn.Linear(self.single_control_obs_dim, self.d_model)
        self.model = TransformerEncoder(self.d_model, 
                                       self.num_heads, 
                                       self.hidden_dim, 
                                       self.num_layers, 
                                       self.dropout, 
                                       is_causal=False, 
                                       activation=self.activation,
                                       enable_sdpa=self.enable_sdpa)
        self.action_output_proj = MLP(self.num_action_tokens * self.d_model, self.mlp_hidden_dims, self.action_dim, self.mlp_activation)
        self.control_obs_output_proj = MLP(self.d_model, self.mlp_hidden_dims, self.single_control_obs_dim, self.mlp_activation)

        # embeddings
        self.proprio_embedding = nn.Parameter(torch.randn(1, self.num_proprio_tokens, self.d_model))
        self.action_embedding = nn.Parameter(torch.randn(1, self.num_action_tokens, self.d_model))
        self.control_obs_embedding = nn.Parameter(torch.randn(1, self.control_obs_horizon, self.d_model))
        self.timestep_embedding = SinusoidalTimestepEmbedder(self.d_model)

        # initial tokens
        self.initial_action_tokens = nn.Parameter(torch.randn(1, self.num_action_tokens, self.d_model))

        # attention masks
        total_tokens = self.num_proprio_tokens + self.num_action_tokens + self.control_obs_horizon
        self.register_buffer('attn_mask', torch.ones(total_tokens, total_tokens, dtype=torch.bool))
        self.attn_mask: torch.Tensor # type hint
        self.attn_mask[self.num_proprio_tokens:self.num_proprio_tokens + self.num_action_tokens, -self.control_obs_horizon:] = False

    def forward(self, proprio: torch.Tensor, control: torch.Tensor, r: torch.Tensor, t: torch.Tensor,
                fwd_dual: bool = False):
        batch_size = proprio.shape[0]
        proprio_tokens = self.proprio_proj(proprio).unflatten(1, (self.num_proprio_tokens, self.d_model)) + self.proprio_embedding
        control_obs_tokens = control.view(batch_size, self.control_obs_horizon, self.single_control_obs_dim)
        timestep_embedding = self.timestep_embedding(t) + self.timestep_embedding(t - r)
        control_obs_tokens = self.control_obs_proj(control_obs_tokens) + self.control_obs_embedding + timestep_embedding
        action_tokens = self.initial_action_tokens.repeat(batch_size, 1, 1) + self.action_embedding

        input = torch.cat([proprio_tokens, action_tokens, control_obs_tokens], dim=1)
        input = self.model(input, attn_mask=self.attn_mask.unsqueeze(0).repeat(batch_size, 1, 1), fwd_dual=fwd_dual)
        actions = self.action_output_proj(input[:, 
                                                self.num_proprio_tokens:self.num_proprio_tokens + self.num_action_tokens
                                            ].flatten(start_dim=1))
        control_obs_velocity = self.control_obs_output_proj(input[:, self.num_proprio_tokens + self.num_action_tokens:])
        return control_obs_velocity, actions
    
    def loss(self, proprio: torch.Tensor, 
             control: torch.Tensor,
             noise: torch.Tensor, 
             r: torch.Tensor, 
             t: torch.Tensor):
        unsqueeze_t = t.unsqueeze(-1)
        noised_control = control * (1 - unsqueeze_t) + noise * unsqueeze_t
        velocity = noise - control
        # make everything with grad
        proprio = proprio.clone().requires_grad_(True)
        noised_control = noised_control.clone().requires_grad_(True)
        r = r.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        # calculate jacobian-vector product
        u, dudt, a = torch.func.jvp(partial(self.forward, proprio, fwd_dual=True), 
            (noised_control, r, t), 
            ( # type: ignore
                velocity,
                torch.zeros_like(r),
                torch.ones_like(t),
            ), has_aux=True
        )
        # compute the target velocity
        u_tgt = velocity - dudt.detach() * (t - r).unsqueeze(-1)
        return a, u, u_tgt
    
    def denoise(self, proprio: torch.Tensor, 
                control: torch.Tensor, 
                r: torch.Tensor, 
                t: torch.Tensor):
        u, a = self.forward(proprio, control, r, t)
        control = control - u * (t - r).unsqueeze(-1)
        return a, control