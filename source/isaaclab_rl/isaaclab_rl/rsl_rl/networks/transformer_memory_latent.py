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
    
class LNNStyleTransformerLatent(nn.Module):
    def __init__(self, 
                 d_model,
                 proprio_dim,
                 latent_dim,
                 num_input_tokens,
                 num_latent_tokens,
                 num_history_tokens,
                 num_layers,
                 num_heads, 
                 hidden_dim, 
                 dropout, 
                 activation,
                 lnn_dt: float = 0.02,
                 lnn_tau: float = 0.5):
        super().__init__()
        self.input_size = proprio_dim + latent_dim

        self.d_model = d_model
        self.proprio_dim = proprio_dim
        self.latent_dim = latent_dim

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size = d_model
        self.dropout = dropout
        self.activation = activation

        self.num_proprio_tokens = num_input_tokens
        self.num_latent_tokens = num_latent_tokens
        self.num_history_tokens = num_history_tokens

        self.lnn_dt = lnn_dt
        self.lnn_tau = lnn_tau
        self.lnn_tau_inv = 1 / lnn_tau
        self.lnn_bias = nn.Parameter(torch.zeros(1, d_model))

        self.proprio_proj = nn.Linear(proprio_dim, num_input_tokens * d_model)
        self.condition_tokenizer = nn.Sequential(
            nn.Linear(latent_dim, num_latent_tokens * d_model),
            nn.GELU(approximate="tanh"),
        )
        self.latent_encoder = nn.Linear(d_model, 2 * d_model)
        self.latent_decoder = nn.Linear(num_latent_tokens * d_model, latent_dim)

        self.model = TransformerEncoder(d_model, 
                                       num_heads, 
                                       hidden_dim, 
                                       num_layers, 
                                       dropout, 
                                       is_causal=False, 
                                       activation=activation)
        
        self.initial_history_tokens = nn.Parameter(torch.zeros(num_history_tokens, d_model))
        self.history_tokens_embeddings = nn.Parameter(torch.randn(num_history_tokens, d_model))
        self.proprio_tokens_embeddings = nn.Parameter(torch.randn(num_input_tokens, d_model))
        self.latent_tokens_embeddings = nn.Parameter(torch.randn(num_latent_tokens, d_model))
        self.out_proj = nn.Sequential(
            nn.Linear(num_input_tokens * d_model, d_model),
            activation,
        )

        self._save_dict = dict()

    def encode_inputs(self, proprio: torch.Tensor, 
                      condition: torch.Tensor | None = None, 
                      latent: torch.Tensor | None = None):
        proprio = self.proprio_proj(proprio)
        if condition is not None:
            tokenized_condition = self.condition_tokenizer(condition)
        else:
            tokenized_condition = None
        return proprio, tokenized_condition, condition, latent
    
    def _lnn_update(self, hidden_states: torch.Tensor, hidden_states_vel: torch.Tensor):
        velocity = - (self.lnn_tau_inv + hidden_states_vel) * hidden_states + hidden_states_vel * self.lnn_bias.unsqueeze(0)
        hidden_states = hidden_states + velocity * self.lnn_dt
        return hidden_states.clip(min=-10, max=10)

    def forward(self, input: TensorDict, hidden_states: torch.Tensor | None,
                compute_latent_loss: bool = False, 
                compute_stable_loss: bool = False,
                no_encode_latent: bool = False, 
                masks: torch.Tensor | None = None):
        transpose_input = input.transpose(0, 1).contiguous()
        batch_size, seq_length = transpose_input.shape('proprio')[:2]
        if hidden_states is None:
            transpose_hidden_states = self.initial_history_tokens.clone().unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            transpose_hidden_states = hidden_states.transpose(0, 1).contiguous()

        if compute_stable_loss:
            assert not no_encode_latent, "no_encode_latent must be False when compute_stable_loss is True"
            assert 'condition' in transpose_input, "condition must be provided when compute_stable_loss is True"
            transpose_input['condition'].requires_grad_()

        proprio, tokenized_condition, condition_ori, latent = self.encode_inputs(proprio=transpose_input['proprio'], 
                                                                                 condition=transpose_input.get('condition', None), 
                                                                                 latent=transpose_input.get('latent', None))
        proprio = proprio.view(batch_size, seq_length, self.num_proprio_tokens, self.d_model)

        if masks is not None:
            num_reduce = masks.sum().item()
            masks = (~masks).transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
        else:
            num_reduce = batch_size * seq_length

        if latent is not None:
            assert tokenized_condition is None, "latent and condition cannot be provided at the same time"
            assert not compute_latent_loss, "input latent directly is not supported when compute_latent_loss is True"
            latent = latent.view(batch_size, seq_length, self.num_latent_tokens, self.d_model)
        else:
            assert tokenized_condition is not None, "condition must be provided when latent is not provided"
            tokenized_condition = tokenized_condition.view(batch_size, seq_length, self.num_latent_tokens, self.d_model)
            latent_mu, latent_logvar = self.latent_encoder(tokenized_condition).chunk(2, dim=-1)

            if compute_stable_loss:
                stable_loss = torch.autograd.grad(outputs=latent_mu, inputs=condition_ori, 
                                                    grad_outputs=torch.ones_like(latent_mu), 
                                                    create_graph=True)[0]
                if masks is not None:
                    stable_loss = stable_loss * masks[..., 0]
                stable_loss = stable_loss.square().mean(dim=-1)
                self._save_dict['stable_loss'] = stable_loss.sum() / num_reduce

            if not no_encode_latent:
                latent_std = torch.exp(0.5 * latent_logvar)
                latent = torch.randn_like(latent_mu) * latent_std + latent_mu
            else:
                latent = latent_mu

        if compute_latent_loss:
            assert not no_encode_latent, "latent loss is not supported when no_encode_latent is True"
            kl_loss: torch.Tensor = 0.5 * (latent_logvar.exp() + latent_mu ** 2 - 1 - latent_logvar)
            if masks is not None:
                kl_loss = kl_loss * masks
            self._save_dict['kl_loss'] = kl_loss.mean(dim=-1).mean(dim=-1).sum() / num_reduce
            self._save_dict['recons_loss'] = 0.0

        # add positional embeddings
        proprio = proprio + self.proprio_tokens_embeddings.unsqueeze(0).unsqueeze(0)
        latent_tokens = latent + self.latent_tokens_embeddings.unsqueeze(0).unsqueeze(0)

        all_outpus = []
        for s in range(seq_length):
            step_proprio = proprio[:, s]
            step_latent = latent_tokens[:, s]
            transpose_hidden_states = transpose_hidden_states + self.history_tokens_embeddings.unsqueeze(0)

            step_input = torch.cat([transpose_hidden_states, 
                                    step_latent,
                                    step_proprio,], dim=1)
            output = self.model(step_input, return_all_layers=True)
            last_output = output[-1]
            
            if compute_latent_loss:
                if masks is not None:
                    step_mask = masks[:, s, 0]
                else:
                    step_mask = 1.0
                recons_latent = last_output[:, self.num_history_tokens:-self.num_proprio_tokens]
                recons_latent = self.latent_decoder(recons_latent.reshape(batch_size, self.num_latent_tokens * self.d_model))
                recons_loss = F.mse_loss(recons_latent.view(batch_size, self.latent_dim) * step_mask, 
                                         condition_ori[:, s] * step_mask, reduction='none').mean(dim=-1).sum()
                self._save_dict['recons_loss'] += recons_loss / num_reduce

            transpose_hidden_states = self._lnn_update(transpose_hidden_states, last_output[:, :self.num_history_tokens])
            all_outpus.append(self.out_proj(
                last_output[:, -self.num_proprio_tokens:].flatten(start_dim=1)))

        all_outputs = torch.stack(all_outpus, dim=1)
        return all_outputs.transpose(0, 1).contiguous(), transpose_hidden_states.transpose(0, 1).contiguous()
    
    def forward_inference(self, proprio: torch.Tensor, 
                          condition: torch.Tensor | None = None,
                          latent: torch.Tensor | None = None,
                          hidden_states: torch.Tensor | None = None,
                          apply_vae_noise: bool = False):
        proprio = proprio.transpose(0, 1).contiguous()
        if condition is not None:
            condition = condition.transpose(0, 1).contiguous()
        if latent is not None:
            latent = latent.transpose(0, 1).contiguous()
        batch_size, seq_length = proprio.shape[0], proprio.shape[1]
        if hidden_states is None:
            transpose_hidden_states = self.initial_history_tokens.clone().unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            transpose_hidden_states = hidden_states.transpose(0, 1).contiguous()
        proprio, tokenized_condition, _, latent = self.encode_inputs(proprio=proprio, 
                                                                    condition=condition, 
                                                                    latent=latent)
        proprio = proprio.view(batch_size, seq_length, self.num_proprio_tokens, self.d_model)

        if latent is not None:
            assert tokenized_condition is None, "latent and condition cannot be provided at the same time"
            latent = latent.view(batch_size, seq_length, self.num_latent_tokens, self.d_model)
        else:
            assert tokenized_condition is not None, "condition must be provided when latent is not provided"
            tokenized_condition = tokenized_condition.view(batch_size, seq_length, self.num_latent_tokens, self.d_model)
            latent = self.latent_encoder(tokenized_condition)
            latent_mu, latent_logvar = latent.chunk(2, dim=-1)
            if apply_vae_noise:
                latent_std = torch.exp(0.5 * latent_logvar)
                latent = torch.randn_like(latent_mu) * latent_std + latent_mu
            else:
                latent = latent_mu
        
        proprio = proprio + self.proprio_tokens_embeddings.unsqueeze(0).unsqueeze(0)
        latent_tokens = latent + self.latent_tokens_embeddings.unsqueeze(0).unsqueeze(0)

        all_outpus = []
        for s in range(seq_length):
            step_proprio = proprio[:, s]
            step_latent = latent_tokens[:, s]
            transpose_hidden_states = transpose_hidden_states + self.history_tokens_embeddings.unsqueeze(0)

            step_input = torch.cat([transpose_hidden_states, 
                                    step_latent,
                                    step_proprio,], dim=1)
            output = self.model(step_input, return_all_layers=False)
            transpose_hidden_states = self._lnn_update(transpose_hidden_states, output[:, :self.num_history_tokens])
            all_outpus.append(self.out_proj(
                output[:, -self.num_proprio_tokens:].flatten(start_dim=1)))

        all_outputs = torch.stack(all_outpus, dim=1)
        return all_outputs.transpose(0, 1).contiguous(), transpose_hidden_states.transpose(0, 1).contiguous()
    

class TransformerMemoryLatent(Memory):
    def __init__(self, proprio_dim, 
                 latent_dim,
                 num_input_tokens,
                 num_latent_tokens,
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
        self.rnn = LNNStyleTransformerLatent(d_model, 
                                       proprio_dim, 
                                       latent_dim,
                                       num_input_tokens,
                                       num_latent_tokens,
                                       num_history_tokens,
                                       num_layers, 
                                       num_heads, 
                                       hidden_dim, 
                                       dropout, 
                                       activation,
                                       lnn_dt,
                                       lnn_tau)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None, *args, **kwargs):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode: needs saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states, *args, **kwargs, masks=masks)
            out = unpad_trajectories(out, masks)
        else:
            # inference/distillation mode: uses hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states, *args, **kwargs)
        return out