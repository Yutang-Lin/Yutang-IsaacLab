# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import unpad_trajectories
from .memory import Memory
from .transformer import TransformerEncoder

class RNNStyleTransformer(nn.Module):
    def __init__(self, 
                 d_model,
                 input_size,
                 history_length,
                 hidden_history,
                 num_layers,
                 num_heads, 
                 hidden_dim, 
                 dropout, 
                 activation):
        super().__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = activation
        self.history_length = history_length
        self.hidden_history = hidden_history
        self.history_size = self.input_size if not hidden_history else self.d_model

        self.input_proj = nn.Linear(input_size, d_model)
        self.model = TransformerEncoder(d_model, 
                                       num_heads, 
                                       hidden_dim, 
                                       num_layers, 
                                       dropout, 
                                       is_causal=True, 
                                       activation=activation)

    def forward(self, input, hidden_states, attn_mask=None):
        transpose_input = input.transpose(0, 1).contiguous()
        if hidden_states is None:
            transpose_hidden_states = torch.zeros(transpose_input.shape[0], self.history_length, self.history_size, device=input.device)
        else:
            transpose_hidden_states = hidden_states.transpose(0, 1).contiguous()

        all_outpus = []
        for s in range(transpose_input.shape[1]):
            step_input = transpose_input[:, s:s+1]
            if self.hidden_history:
                inputs = torch.cat([transpose_hidden_states, self.input_proj(step_input)], dim=1)
            else:
                inputs = self.input_proj(torch.cat([transpose_hidden_states, step_input], dim=1))
            x = self.model(inputs, attn_mask)

            if self.hidden_history:
                transpose_hidden_states = x[:, -self.history_length:]
            else:
                transpose_hidden_states = transpose_hidden_states.roll(-1, dims=1)
                transpose_hidden_states[:, -1:] = step_input
            all_outpus.append(x[:, -1])

        all_outputs = torch.stack(all_outpus, dim=0)
        return all_outputs, transpose_hidden_states.transpose(0, 1).contiguous()
    

class TransformerMemory(Memory):
    def __init__(self, input_size, history_length, hidden_history=True, num_layers=1, d_model=256, hidden_dim=1024, num_heads=8, dropout=0.1, activation=None):
        super(Memory, self).__init__()
        self.rnn = RNNStyleTransformer(d_model, 
                                       input_size, 
                                       history_length,
                                       hidden_history,
                                       num_layers, 
                                       num_heads, 
                                       hidden_dim, 
                                       dropout, 
                                       activation)
        self.hidden_states = None