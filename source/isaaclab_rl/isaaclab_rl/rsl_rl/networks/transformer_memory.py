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

class LNNStyleTransformer(nn.Module):
    def __init__(self, 
                 d_model,
                 input_size,
                 num_input_tokens,
                 num_history_tokens,
                 num_layers,
                 num_heads, 
                 hidden_dim, 
                 dropout, 
                 activation,
                 lnn_dt: float = 0.02,
                 lnn_tau: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.num_input_tokens = num_input_tokens

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size = d_model
        self.dropout = dropout
        self.activation = activation

        self.num_history_tokens = num_history_tokens

        self.lnn_dt = lnn_dt
        self.lnn_tau = lnn_tau
        self.lnn_tau_inv = 1 / lnn_tau
        self.lnn_bias = nn.Parameter(torch.zeros(1, d_model))

        self.input_proj = nn.Linear(input_size, num_input_tokens * d_model)
        self.model = TransformerEncoder(d_model, 
                                       num_heads, 
                                       hidden_dim, 
                                       num_layers, 
                                       dropout, 
                                       is_causal=False, 
                                       activation=activation)
        self.initial_cls_token = nn.Parameter(torch.randn(1, d_model))
        self.initial_history_tokens = nn.Parameter(torch.zeros(num_history_tokens, d_model))
        self.history_tokens_embeddings = nn.Parameter(torch.randn(1, d_model))
        self.input_tokens_embeddings = nn.Parameter(torch.randn(1, d_model))
    
    def _lnn_update(self, hidden_states: torch.Tensor, hidden_states_vel: torch.Tensor):
        velocity = - (self.lnn_tau_inv + hidden_states_vel) * hidden_states + hidden_states_vel * self.lnn_bias.unsqueeze(0)
        hidden_states = hidden_states + velocity * self.lnn_dt
        return hidden_states.clip(min=-10, max=10)

    def forward(self, input: torch.Tensor, hidden_states: torch.Tensor | None, attn_mask=None):
        transpose_input = input.transpose(0, 1).contiguous()
        batch_size, seq_length = transpose_input.shape[:2]
        if hidden_states is None:
            transpose_hidden_states = self.initial_history_tokens.clone().unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            transpose_hidden_states = hidden_states.transpose(0, 1).contiguous()

        input = self.input_proj(input)
        input = input.view(batch_size, seq_length, self.num_input_tokens, self.d_model)

        all_outpus = []
        for s in range(seq_length):
            step_input = input[:, s]
            step_input = step_input + self.input_tokens_embeddings.unsqueeze(0)
            transpose_hidden_states = transpose_hidden_states + self.history_tokens_embeddings.unsqueeze(0)

            step_input = torch.cat([transpose_hidden_states, 
                                    step_input, 
                                    self.initial_cls_token.unsqueeze(0).repeat(step_input.shape[0], 1, 1)], dim=1)
            output = self.model(step_input, attn_mask)
            transpose_hidden_states = self._lnn_update(transpose_hidden_states, output[:, :self.num_history_tokens])
            all_outpus.append(output[:, -1])

        all_outputs = torch.stack(all_outpus, dim=1)
        return all_outputs.transpose(0, 1).contiguous(), transpose_hidden_states.transpose(0, 1).contiguous()
    

class TransformerMemory(Memory):
    def __init__(self, input_size, num_input_tokens, num_history_tokens, 
                 num_layers=1, d_model=256, hidden_dim=1024, num_heads=8, dropout=0.1, activation=None,
                 lnn_dt=0.02, lnn_tau=0.5):
        super(Memory, self).__init__()
        self.rnn = LNNStyleTransformer(d_model, 
                                       input_size, 
                                       num_input_tokens,
                                       num_history_tokens,
                                       num_layers, 
                                       num_heads, 
                                       hidden_dim, 
                                       dropout, 
                                       activation,
                                       lnn_dt,
                                       lnn_tau)
        self.hidden_states = None