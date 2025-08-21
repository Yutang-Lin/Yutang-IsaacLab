# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from .transformer import TransformerEncoder
from isaaclab_rl.rsl_rl.utils import TensorDict
import torch.nn.functional as F

class TransformerPolicyLatent(nn.Module):
    def __init__(self, 
                 input_size,
                 condition_size,
                 output_size,
                 mlp_hidden_dims,
                 mlp_activation,
                 num_input_tokens,
                 num_latent_tokens,
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
        self.num_latent_tokens = num_latent_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size = d_model
        self.dropout = dropout
        self.activation = activation

        self.proprio_proj = nn.Linear(input_size, num_input_tokens * d_model)
        self.condition_tokenizer = nn.Sequential(
            nn.Linear(condition_size, num_latent_tokens * d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(num_latent_tokens * d_model, num_latent_tokens * d_model),
            nn.LayerNorm(num_latent_tokens * d_model),
            nn.GELU(approximate="tanh"),
        )
        self.latent_encoder = nn.Linear(d_model, 2 * d_model)
        self.latent_decoder = nn.Sequential(
            nn.Linear(num_latent_tokens * d_model, num_latent_tokens * d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(num_latent_tokens * d_model, condition_size),
        )
        self.model = TransformerEncoder(d_model, 
                                       num_heads, 
                                       hidden_dim, 
                                       num_layers, 
                                       dropout,
                                       is_causal=False, 
                                       activation=activation,
                                       enable_sdpa=enable_sdpa)
        out_proj_layers = [nn.Linear((num_input_tokens + num_latent_tokens) * d_model, mlp_hidden_dims[0]), mlp_activation]
        for i in range(1, len(mlp_hidden_dims)):
            out_proj_layers.append(nn.Linear(mlp_hidden_dims[i-1], mlp_hidden_dims[i]))
            out_proj_layers.append(mlp_activation)
        out_proj_layers.append(nn.Linear(mlp_hidden_dims[-1], output_size))
        self.output_proj = nn.Sequential(*out_proj_layers)

        self.proprio_tokens_embeddings = nn.Parameter(torch.randn(num_input_tokens, d_model))
        self.latent_tokens_embeddings = nn.Parameter(torch.randn(num_latent_tokens, d_model))
        self._save_dict = dict()

    def encode_inputs(self, proprio: torch.Tensor, condition: torch.Tensor | None = None,
                      latent: torch.Tensor | None = None):
        proprio = self.proprio_proj(proprio)
        if condition is not None:
            tokenized_condition = self.condition_tokenizer(condition)
        else:
            tokenized_condition = None
        return proprio, tokenized_condition, condition, latent

    def forward(self, input: TensorDict, 
                compute_latent_loss: bool = False, 
                compute_stable_loss: bool = False,
                apply_vae_noise: bool = True,
                return_latent: bool = False):
        if compute_stable_loss:
            assert input.get('condition', None) is not None, "condition must be provided when compute_stable_loss is True"
            input['condition'].requires_grad_()
        proprio, tokenized_condition, condition, latent = self.encode_inputs(proprio=input['proprio'], 
                                                                             condition=input.get('condition', None), 
                                                                             latent=input.get('latent', None))
        batch_size = proprio.shape[0]
        proprio = proprio.view(batch_size, self.num_input_tokens, self.d_model)

        if latent is not None:
            assert tokenized_condition is None, "latent and condition cannot be provided at the same time"
            latent = latent.view(batch_size, self.num_latent_tokens, self.d_model)
        else:
            assert tokenized_condition is not None, "condition must be provided when latent is not provided"
            tokenized_condition = tokenized_condition.view(batch_size, self.num_latent_tokens, self.d_model)
            latent_mu, latent_logvar = self.latent_encoder(tokenized_condition).chunk(2, dim=-1)

            if compute_stable_loss:
                stable_loss = torch.autograd.grad(outputs=latent_mu, inputs=condition, 
                                                    grad_outputs=torch.ones_like(latent_mu), 
                                                    create_graph=True)[0]
                self._save_dict['stable_loss'] = stable_loss.square().mean()

            if apply_vae_noise:
                latent_std = torch.exp(0.5 * latent_logvar)
                latent = torch.randn_like(latent_mu) * latent_std + latent_mu
            else:
                latent = latent_mu

            if compute_latent_loss:
                kl_loss: torch.Tensor = 0.5 * (latent_logvar.exp() + latent_mu ** 2 - 1 - latent_logvar)
                self._save_dict['kl_loss'] = kl_loss.mean()
        output = self.model(torch.cat([proprio + self.proprio_tokens_embeddings.unsqueeze(0), 
                                       latent + self.latent_tokens_embeddings.unsqueeze(0)], dim=1))

        if compute_latent_loss:
            output_latent = output[:, -self.num_latent_tokens:].flatten(start_dim=1)
            recons_latent = self.latent_decoder(output_latent)
            self._save_dict['recons_loss'] = F.mse_loss(recons_latent, condition)
        
        output = output.flatten(start_dim=1)
        if not return_latent:
            return self.output_proj(output)
        else:
            return self.output_proj(output), dict(latent_mu=latent_mu, latent_logvar=latent_logvar)
    
    def forward_inference(self, proprio: torch.Tensor, 
                          condition: torch.Tensor | None = None,
                          latent: torch.Tensor | None = None,
                          apply_vae_noise: bool = False):
        proprio, tokenized_condition, condition, latent = self.encode_inputs(proprio=proprio, 
                                                                             condition=condition, 
                                                                             latent=latent)
        batch_size = proprio.shape[0]
        if latent is not None:
            assert tokenized_condition is None, "latent and condition cannot be provided at the same time"
            latent = latent.view(batch_size, self.num_latent_tokens, self.d_model)
        else:
            assert tokenized_condition is not None, "condition must be provided when latent is not provided"
            tokenized_condition = tokenized_condition.view(batch_size, self.num_latent_tokens, self.d_model)
            latent_mu, latent_logvar = self.latent_encoder(tokenized_condition).chunk(2, dim=-1)
            if apply_vae_noise:
                latent_std = torch.exp(0.5 * latent_logvar)
                latent = torch.randn_like(latent_mu) * latent_std + latent_mu
            else:
                latent = latent_mu
        return self.output_proj(self.model(torch.cat([proprio + self.proprio_tokens_embeddings.unsqueeze(0), 
                                                      latent + self.latent_tokens_embeddings.unsqueeze(0)], dim=1)).flatten(start_dim=1))