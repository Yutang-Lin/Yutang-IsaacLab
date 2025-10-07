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
from transformers import PreTrainedModel, PretrainedConfig # type: ignore

class TransformerPolicyDDIMConfig(PretrainedConfig):
    def __init__(self,
                 *,
                 proprio_dim=0,
                 control_obs_dim=0,
                 action_dim=0,
                 control_obs_horizon=0,
                 mlp_hidden_dims=[],
                 mlp_activation="elu",
                 num_proprio_tokens=0,
                 num_action_tokens=0,
                 d_model=0,
                 num_layers=0,
                 num_heads=0, 
                 hidden_dim=0,
                 dropout=0.0,
                 activation="elu",
                 enable_sdpa: bool = True,
                 # DDIM specific parameters
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 num_timesteps: int = 1000,
                 eta: float = 0.0):
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
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps
        self.eta = eta

class TransformerPolicyDDIM(PreTrainedModel):
    def __init__(self, 
                 config: TransformerPolicyDDIMConfig):
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

        # DDIM noise scheduling
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.num_timesteps = config.num_timesteps
        self.eta = config.eta
        
        # Create beta schedule
        self.beta_schedule = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        
        # Compute alpha schedule
        self.alpha_schedule = 1.0 - self.beta_schedule
        self.alpha_bar_schedule = torch.cumprod(self.alpha_schedule, dim=0)
        
        # Register buffers for noise schedules
        self.register_buffer('beta', self.beta_schedule)
        self.register_buffer('alpha', self.alpha_schedule)
        self.register_buffer('alpha_bar', self.alpha_bar_schedule)
        self.beta: torch.Tensor
        self.alpha: torch.Tensor
        self.alpha_bar: torch.Tensor

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
                                       enable_sdpa=self.enable_sdpa,
                                       norm_first=True)
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

    def apply_noise(self, control: torch.Tensor, timestep: torch.Tensor):
        control = control.reshape(control.shape[0], self.control_obs_horizon, self.single_control_obs_dim)
        alpha_bar = self.alpha_bar[timestep.long()].unsqueeze(-1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        noise = torch.randn_like(control)
        return sqrt_alpha_bar * control + sqrt_one_minus_alpha_bar * noise

    def forward(self, proprio: torch.Tensor, control: torch.Tensor, timestep: torch.Tensor):
        batch_size = proprio.shape[0]
        proprio_tokens = self.proprio_proj(proprio).unflatten(1, (self.num_proprio_tokens, self.d_model)) + self.proprio_embedding
        control_obs_tokens = control.view(batch_size, self.control_obs_horizon, self.single_control_obs_dim)
        control_obs_tokens = self.control_obs_proj(control_obs_tokens) + self.control_obs_embedding + self.timestep_embedding(timestep.float())
        action_tokens = self.initial_action_tokens.repeat(batch_size, 1, 1) + self.action_embedding

        input = torch.cat([proprio_tokens, action_tokens, control_obs_tokens], dim=1)
        input = self.model(input, attn_mask=self.attn_mask.unsqueeze(0).repeat(batch_size, 1, 1))
        actions = self.action_output_proj(input[:, 
                                                self.num_proprio_tokens:self.num_proprio_tokens + self.num_action_tokens
                                            ].flatten(start_dim=1))
        clean_control = self.control_obs_output_proj(input[:, self.num_proprio_tokens + self.num_action_tokens:])
        return actions, clean_control
    
    @torch.no_grad()
    def _sample_one_step(self, x_t, pred_clean, 
                        time_step: torch.Tensor, 
                        prev_time_step: torch.Tensor, 
                        eta: float = 0.0):
        # get current and previous alpha_cumprod
        alpha_t = self.alpha_bar[time_step].unsqueeze(-1)
        alpha_t_prev = self.alpha_bar[prev_time_step].unsqueeze(-1)

        # predict noise using model
        epsilon_theta_t = (x_t - torch.sqrt(alpha_t) * pred_clean) / torch.sqrt(1 - alpha_t)

        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (
                torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                    (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                sigma_t * epsilon_t
        )
        return x_t_minus_one
    
    def _convert_timestep(self, t):
        if t.dtype == torch.long:
            return t
        else:
            return (t * (self.num_timesteps - 1)).long().clamp(min=0, max=self.num_timesteps - 1)
    
    def denoise(self, proprio: torch.Tensor, 
                control: torch.Tensor, 
                timestep: torch.Tensor, 
                timestep_target: torch.Tensor):
        actions, clean_control = self.forward(proprio, control, timestep)
        timestep = self._convert_timestep(timestep)
        timestep_target = self._convert_timestep(timestep_target)
        denoised_control = self._sample_one_step(control, 
                                                clean_control, 
                                                timestep, 
                                                timestep_target, 
                                                self.eta)
        return actions, denoised_control