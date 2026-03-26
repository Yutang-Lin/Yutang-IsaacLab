# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass


@configclass
class RslRlSmpCfg:
    """Configuration for the Score-Matching Motion Prior (SMP) reward module."""

    model_name: str = "kimodo-g1-rp"
    """Kimodo model name to load (e.g., 'kimodo-g1-rp', 'kimodo-g1-seed')."""

    sds_lambda: float = 1.0
    """Scaling factor inside the exp(-lambda * ||eps_hat - eps||^2) reward."""

    reward_scale: float = 1.0
    """Overall scale applied to the SMP reward before adding to total reward."""

    reward_baseline: float = 0.5
    """Baseline subtracted from raw SDS reward before scaling. Only rewards above this threshold contribute."""

    noise_timestep_range: tuple = (50, 950)
    """Range of diffusion timesteps to sample from (low, high) out of num_base_steps."""

    num_base_steps: int = 1000
    """Number of base diffusion steps in the noise schedule."""

    cfg_weight: tuple = (0.0, 1.0)
    """Classifier-free guidance weights (w_text, w_constraint) for separated CFG.
    w_text=0 means unconditional text; w_constraint controls how strongly the
    denoiser respects wrist inpainting constraints. Only used when wrist_conditioning=True."""

    smooth_kernel: int = 0
    """Gaussian smoothing kernel size for temporal smoothing of first-order features (0=disabled)."""

    smooth_sigma: float = 0.0
    """Gaussian smoothing sigma. Only used when smooth_kernel > 0."""

    wrist_conditioning: bool = False
    """If True, condition the denoiser on the rollout's wrist trajectory via inpainting."""

    rollout_fps: float = 50.0
    """Policy rollout frame rate in Hz. Used for finite-difference velocity computation."""
