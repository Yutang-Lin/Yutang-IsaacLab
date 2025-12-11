# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import math
from einops import rearrange, einsum
from .transformer import TransformerEncoder

class SelectiveFuser(nn.Module):
    def __init__(self, d_model, interaction_field_size, movement_goal_size, task_condition_size, num_fusion_heads):
        super().__init__()
        self.interaction_field_size = interaction_field_size
        self.movement_goal_size = movement_goal_size
        self.task_condition_size = task_condition_size

        self.d_model = d_model
        self.num_fusion_heads = num_fusion_heads
        assert d_model % num_fusion_heads == 0, "d_model must be divisible by num_fusion_heads"
        self.single_dim = d_model // num_fusion_heads
        
        self.interaction_field_proj = nn.Sequential(nn.Linear(interaction_field_size, d_model), nn.GELU(approximate="tanh"))
        self.movement_goal_proj = nn.Sequential(nn.Linear(movement_goal_size, d_model), nn.GELU(approximate="tanh"))
        self.task_condition_proj = nn.Sequential(nn.Linear(task_condition_size, d_model), nn.GELU(approximate="tanh"))

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.attn_temperature = math.sqrt(self.single_dim)

    def forward(self, interaction_field, movement_goal, task_condition):
        batch_size = interaction_field.shape[0]
        interaction_field = self.interaction_field_proj(interaction_field)
        movement_goal = self.movement_goal_proj(movement_goal)
        task_condition = self.task_condition_proj(task_condition)

        combined_tokens = torch.stack([interaction_field, movement_goal], dim=1)
        ks = self.k_proj(combined_tokens).view(batch_size, 2 * self.num_fusion_heads, self.single_dim)
        vs = self.v_proj(combined_tokens).view(batch_size, 2 * self.num_fusion_heads, self.single_dim)
        qs = self.q_proj(task_condition).view(batch_size, self.num_fusion_heads, self.single_dim)
        attn_score = einsum(qs, ks, "b q d, b k d -> b q k")
        attn_score = (attn_score / self.attn_temperature).softmax(dim=-1)
        out = einsum(attn_score, vs, "b q k, b k d -> b q d").flatten(start_dim=1)
        out = self.output_proj(out)
        return out / (out.norm(dim=-1, keepdim=True) + 1e-8)

class TransformerPolicyInteractionField(nn.Module):
    def __init__(self, 
                 input_size,
                 interaction_field_size,
                 movement_goal_size,
                 task_condition_size,
                 output_size,
                 num_fusion_heads,
                 mlp_hidden_dims,
                 mlp_activation,
                 num_input_tokens,
                 d_model,
                 num_layers,
                 num_heads, 
                 hidden_dim, 
                 dropout, 
                 activation,
                 enable_sdpa: bool = False):
        super().__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.in_features = input_size
        self.out_features = output_size
        self.num_input_tokens = num_input_tokens
        self.interaction_field_size = interaction_field_size
        self.movement_goal_size = movement_goal_size
        self.task_condition_size = task_condition_size
        self.num_fusion_heads = num_fusion_heads

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size = d_model
        self.dropout = dropout
        self.activation = activation

        self.selective_fuser = SelectiveFuser(
            d_model, interaction_field_size, movement_goal_size, task_condition_size, num_fusion_heads)
        self.input_proj = nn.Linear(input_size + d_model, num_input_tokens * d_model)
        self.model = TransformerEncoder(d_model, 
                                       num_heads, 
                                       hidden_dim, 
                                       num_layers, 
                                       dropout, 
                                       is_causal=True, 
                                       activation=activation,
                                       enable_sdpa=enable_sdpa)
        out_proj_layers = [nn.Linear(num_input_tokens * d_model, mlp_hidden_dims[0]), mlp_activation]
        for i in range(1, len(mlp_hidden_dims)):
            out_proj_layers.append(nn.Linear(mlp_hidden_dims[i-1], mlp_hidden_dims[i]))
            out_proj_layers.append(mlp_activation)
        out_proj_layers.append(nn.Linear(mlp_hidden_dims[-1], output_size))
        self.output_proj = nn.Sequential(*out_proj_layers)

    def forward(self, proprio: torch.Tensor, 
                interaction_field: torch.Tensor, 
                movement_goal: torch.Tensor, 
                task_condition: torch.Tensor):
        input = self.selective_fuser(interaction_field, movement_goal, task_condition)
        input = self.input_proj(torch.cat([proprio, input], dim=1))
        input = self.model(input.view(input.shape[0], self.num_input_tokens, self.d_model))
        return self.output_proj(input.flatten(start_dim=1))