# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class RslRlAmpCfg:
    """Configuration for the Adversarial Motion Prior (AMP) module."""

    input_dim: int = 1
    """The input dimension for the AMP module. Default is 1."""

    learning_rate: float = 1e-3
    """The learning rate for the AMP module. Default is 1e-3."""

    num_learning_epochs: int = 3
    """The number of learning epochs for the AMP module. Default is 1."""

    max_buffer_size: int = 49152
    """The maximum size of the buffer. Default is 49152."""
    
    sample_k: int = 2048
    """The number of samples to sample from the buffer. Default is 2048."""

    hidden_dims: list[int] = [256, 256, 256]
    """The hidden dimensions for the AMP discriminator network. Default is [256, 256, 256]."""

    activation: str = "relu"
    """The activation function for the AMP discriminator network. Default is "relu"."""

    use_transformer: bool = False
    """Whether to use a transformer for the AMP discriminator network. Default is False."""

    tf_d_model: int = 256
    """The dimension of the transformer for the AMP discriminator network. Default is 256."""

    tf_hidden_dim: int = 512
    """The hidden dimension for the transformer for the AMP discriminator network. Default is 256."""

    tf_num_layers: int = 2
    """The number of layers for the transformer for the AMP discriminator network. Default is 3."""

    tf_num_heads: int = 4
    """The number of heads for the transformer for the AMP discriminator network. Default is 4."""

    tf_dropout: float = 0.1
    """The dropout for the transformer for the AMP discriminator network. Default is 0.1."""

    tf_num_input_tokens: int = 4
    """The number of input tokens for the transformer for the AMP discriminator network. Default is 1."""

    tf_activation: str = "gelu"
    """The activation function for the transformer for the AMP discriminator network. Default is "gelu"."""

    offload_buffer: bool = False
    """Whether to offload the buffer for the AMP buffer to CPU. Default is False."""

    layer_norm: bool = False
    """Whether to use layer normalization for the AMP discriminator network. Default is False."""

    reward_scale: float = 1.0
    """The reward scale for the AMP discriminator network. Default is 1.0."""

    reward_factor: float = 1.0
    """The reward factor for the AMP discriminator network. Default is 1.0."""

    clip_obs_value: float = 100.0
    """The clipping value for the AMP discriminator network. Default is 100.0."""
   
    w_grad_penalty: float = 10.0
    """The weight for the gradient penalty for the AMP discriminator network. Default is 10.0."""

    max_grad_norm: float = 1.0
    """The maximum gradient norm for the AMP discriminator network. Default is 1.0."""
    