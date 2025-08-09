# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from .transformer import TransformerEncoder

class TransformerPolicy(nn.Module):
    def __init__(self, 
                 input_size,
                 output_size,
                 mlp_hidden_dims,
                 mlp_activation,
                 num_input_tokens,
                 d_model,
                 num_layers,
                 num_heads, 
                 hidden_dim, 
                 dropout, 
                 activation,
                 enable_sdpa: bool = True):
        super().__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.in_features = input_size
        self.out_features = output_size
        self.num_input_tokens = num_input_tokens

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size = d_model
        self.dropout = dropout
        self.activation = activation

        self.input_proj = nn.Linear(input_size, num_input_tokens * d_model)
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

    def forward(self, input: torch.Tensor):
        input = self.model(self.input_proj(input).view(input.shape[0], self.num_input_tokens, self.d_model))
        return self.output_proj(input.flatten(start_dim=1))