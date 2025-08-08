# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class RslRlAmpCfg:
    """Configuration for the Adversarial Motion Prior (AMP) module.

    """

    input_dim: int = 1
    """The input dimension for the AMP module. Default is 1."""

    learning_rate: float = 1e-3
    """The learning rate for the AMP module. Default is 1e-3."""

    num_learning_epochs: int = 3
    """The number of learning epochs for the AMP module. Default is 1."""

    hidden_dims: list[int] = [256, 256, 256]
    """The hidden dimensions for the AMP discriminator network. Default is [256, 256, 256]."""

    activation: str = "relu"
    """The activation function for the AMP discriminator network. Default is "relu"."""

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
    