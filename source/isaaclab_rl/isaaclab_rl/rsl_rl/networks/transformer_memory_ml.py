# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.utils import unpad_trajectories
from .memory import Memory
from .transformer import TransformerEncoder
from isaaclab_rl.rsl_rl.utils import TensorDict
    
class LNNStyleTransformerML(nn.Module):
    def __init__(self, 
                 d_model,
                 proprio_dim,
                 text_dim,
                 motion_dim,
                 num_input_tokens,
                 num_task_tokens,
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
        self.proprio_dim = proprio_dim
        self.text_dim = text_dim
        self.motion_dim = motion_dim

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size = d_model
        self.dropout = dropout
        self.activation = activation

        self.num_proprio_tokens = num_input_tokens
        self.num_task_tokens = num_task_tokens
        self.num_history_tokens = num_history_tokens

        self.lnn_dt = lnn_dt
        self.lnn_tau = lnn_tau
        self.lnn_tau_inv = 1 / lnn_tau
        self.lnn_bias = nn.Parameter(torch.zeros(1, d_model))

        self.proprio_proj = nn.Linear(proprio_dim, num_input_tokens * d_model)
        self.text_proj = nn.Linear(text_dim, num_task_tokens * d_model)
        self.motion_proj = nn.Linear(motion_dim, num_task_tokens * d_model)
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
        self.task_tokens_embeddings = nn.Parameter(torch.randn(1, d_model))
        self.proprio_tokens_embeddings = nn.Parameter(torch.randn(1, d_model))

        self._save_dict = dict()

    def encode_inputs(self, proprio, text=None, motion=None, blend_mask=None):
        proprio = self.proprio_proj(proprio)
        if text is not None:
            text = self.text_proj(text)
        if motion is not None:
            motion = self.motion_proj(motion)
        return proprio, text, motion, blend_mask
    
    def _lnn_update(self, hidden_states: torch.Tensor, hidden_states_vel: torch.Tensor):
        velocity = - (self.lnn_tau_inv + hidden_states_vel) * hidden_states + hidden_states_vel * self.lnn_bias.unsqueeze(0)
        hidden_states = hidden_states + velocity * self.lnn_dt
        return hidden_states.clip(min=-10, max=10)

    def forward(self, input: TensorDict, hidden_states: torch.Tensor | None, attn_mask=None,
                compute_align_loss=False, use_all_task_tokens=False):
        transpose_input = input.transpose(0, 1).contiguous()
        batch_size, seq_length = transpose_input.shape('proprio')[:2]
        if hidden_states is None:
            transpose_hidden_states = self.initial_history_tokens.clone().unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            transpose_hidden_states = hidden_states.transpose(0, 1).contiguous()

        proprio, text, motion, blend_mask = self.encode_inputs(**transpose_input)
        proprio = proprio.view(batch_size, seq_length, self.num_proprio_tokens, self.d_model)
        text = text.view(batch_size, seq_length, self.num_task_tokens, self.d_model)
        motion = motion.view(batch_size, seq_length, self.num_task_tokens, self.d_model)

        blend_mask = (blend_mask.view(batch_size, seq_length, 1, 1) + 1e-3).long()
        motion_mask = blend_mask == 0
        text_mask = blend_mask == 1

        if compute_align_loss:
            # duplicate batch
            proprio = torch.cat([proprio, proprio], dim=0)
            transpose_hidden_states = torch.cat([transpose_hidden_states, transpose_hidden_states], dim=0)
            assert text is not None and motion is not None, "text and motion must be provided if compute_align_loss is True"
            task_tokens = torch.cat([motion, text], dim=0)
            if 'hidden_align' not in self._save_dict:
                self._save_dict['hidden_align'] = 0.0
                self._save_dict['task_align'] = 0.0

        elif not use_all_task_tokens:
            self._save_dict.clear()
            assert (text is None or motion is None) or blend_mask is not None, "blend_mask must be provided if text or motion is both provided"
            task_tokens = motion * motion_mask + text * text_mask
        else:
            task_tokens = torch.cat([text, motion], dim=2)

        all_outpus = []
        for s in range(seq_length):
            step_proprio = proprio[:, s]
            step_task = task_tokens[:, s]
            step_proprio = step_proprio + self.proprio_tokens_embeddings.unsqueeze(0)
            step_task = step_task + self.task_tokens_embeddings.unsqueeze(0)
            transpose_hidden_states = transpose_hidden_states + self.history_tokens_embeddings.unsqueeze(0)

            step_input = torch.cat([transpose_hidden_states, 
                                    step_proprio, 
                                    step_task, 
                                    self.initial_cls_token.unsqueeze(0).repeat(step_proprio.shape[0], 1, 1)], dim=1)
            output = self.model(step_input, attn_mask, return_all_layers=True)
            last_output = output[-1]
            if compute_align_loss:
                hidden_align, task_align = 0.0, 0.0
                for layer_output in output:
                    motion_output, text_output = layer_output.chunk(2, dim=0)
                    task_token_start = self.num_history_tokens + self.num_proprio_tokens
                    hidden_align += F.mse_loss(text_output[:, -1], motion_output[:, -1])
                    task_align += F.mse_loss(text_output[:, task_token_start:task_token_start+self.num_task_tokens], 
                                            motion_output[:, task_token_start:task_token_start+self.num_task_tokens])
                self._save_dict['hidden_align'] += hidden_align / seq_length
                self._save_dict['task_align'] += task_align / seq_length

            transpose_hidden_states = self._lnn_update(transpose_hidden_states, last_output[:, :self.num_history_tokens])
            all_outpus.append(last_output[:, -1])

        all_outputs = torch.stack(all_outpus, dim=1)
        if compute_align_loss:
            motion_output, text_output = all_outputs.chunk(2, dim=0)
            motion_hidden_states, text_hidden_states = transpose_hidden_states.chunk(2, dim=0)

            motion_mask, text_mask = motion_mask.squeeze(-1), text_mask.squeeze(-1)
            all_outputs = motion_output * motion_mask + text_output * text_mask
            motion_mask, text_mask = motion_mask[:, -1:], text_mask[:, -1:]
            transpose_hidden_states = motion_hidden_states * motion_mask + text_hidden_states * text_mask
        
        return all_outputs.transpose(0, 1).contiguous(), transpose_hidden_states.transpose(0, 1).contiguous()
    

class TransformerMemoryML(Memory):
    def __init__(self, proprio_dim, 
                 text_dim, 
                 motion_dim,
                 num_input_tokens,
                 num_task_tokens,
                 num_history_tokens,
                 num_layers=1, 
                 d_model=256, 
                 hidden_dim=1024, 
                 num_heads=8, 
                 dropout=0.1, 
                 activation=None,
                 lnn_dt=0.02,
                 lnn_tau=0.5):
        super(Memory, self).__init__()
        self.rnn = LNNStyleTransformerML(d_model, 
                                       proprio_dim, 
                                       text_dim,
                                       motion_dim,
                                       num_input_tokens,
                                       num_task_tokens,
                                       num_history_tokens,
                                       num_layers, 
                                       num_heads, 
                                       hidden_dim, 
                                       dropout, 
                                       activation,
                                       lnn_dt,
                                       lnn_tau)
        self.hidden_states = None