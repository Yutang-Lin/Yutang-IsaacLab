# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal, Any

from isaaclab.utils import configclass

#########################
# Policy configurations #
#########################


@configclass
class RslRlDistillationStudentTeacherCfg:
    """Configuration for the distillation student-teacher networks."""

    class_name: str = "StudentTeacher"
    """The policy class name. Default is StudentTeacher."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the student policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    student_policy_cfg: Any = MISSING
    """The policy configuration for the student network."""

    teacher_policy_ckpt: str = MISSING
    """The checkpoint path for the teacher policy."""


@configclass
class RslRlDistillationStudentTeacherRecurrentCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for the distillation student-teacher recurrent networks."""

    class_name: str = "StudentTeacherRecurrent"
    """The policy class name. Default is StudentTeacherRecurrent."""


@configclass
class RslRlDistillationStudentCVAETrackerCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for the CVAE-based student tracking policy.

    Uses a Conditional VAE with low-rank posterior correction from motion_keybody observations.
    """

    class_name: str = "StudentCVAETracker"
    """The policy class name. Default is StudentCVAETracker."""


@configclass
class RslRlDistillationStudentCoDiTTrackerCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for CoDiT-Track distillation policy.

    Uses a Condition-Denoising Transformer with two-view corruption training,
    dual action heads (base + conditional), and future denoising auxiliary loss.
    """

    class_name: str = "StudentCoDiTTracker"
    """The policy class name. Default is StudentCoDiTTracker."""


@configclass
class RslRlDistillationStudentCoDiTMFTrackerCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for CoDiT-MF distillation policy.

    Uses MeanFlow velocity prediction with JVP self-consistency,
    contrastive feature regularization, and dual action heads.
    """

    class_name: str = "StudentCoDiTMFTracker"
    """The policy class name. Default is StudentCoDiTMFTracker."""


@configclass
class RslRlDistillationStudentCVAEBFMTrackerCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for CVAE-BFM foundation model distillation policy.

    Uses variable-interval 10-frame future conditioning with delta_t,
    per-frame pad masking, and binary keypoint masking.
    """

    class_name: str = "StudentCVAEBFMTracker"
    """The policy class name. Default is StudentCVAEBFMTracker."""


@configclass
class RslRlDistillationStudentVQVAEBFMTrackerCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for VQ-VAE BFM distillation policy."""

    class_name: str = "StudentVQVAEBFMTracker"
    """The policy class name. Default is StudentVQVAEBFMTracker."""


@configclass
class RslRlDistillationStudentFlowBFMTrackerCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for Flow-BFM distillation policy."""

    class_name: str = "StudentFlowBFMTracker"
    """The policy class name. Default is StudentFlowBFMTracker."""


@configclass
class RslRlDistillationStudentLFMBFMTrackerCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for LFM-BFM (Latent Flow Matching) distillation policy."""

    class_name: str = "StudentLFMBFMTracker"
    """The policy class name. Default is StudentLFMBFMTracker."""


@configclass
class RslRlDistillationStudentBCBFMTrackerCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for BC-BFM (naive transformer BC) distillation policy."""

    class_name: str = "StudentBCBFMTracker"
    """The policy class name. Default is StudentBCBFMTracker."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlDistillationAlgorithmCfg:
    """Configuration for the distillation algorithm."""

    class_name: str = "Distillation"
    """The algorithm class name. Default is Distillation."""

    num_learning_epochs: int = MISSING
    """The number of updates performed with each sample."""

    learning_rate: float = MISSING
    """The learning rate for the student policy."""

    weight_decay: float = 0.0
    """The weight decay for the student policy."""

    gradient_length: int = MISSING
    """The number of environment steps the gradient flows back."""

    max_grad_norm: None | float = None
    """The maximum norm the gradient is clipped to."""

@configclass
class RslRlFlowDAggerAlgorithmCfg:
    """Configuration for the distillation algorithm."""

    class_name: str = "FlowDAgger"
    """The algorithm class name. Default is FlowDAgger."""

    num_learning_epochs: int = MISSING
    """The number of updates performed with each sample."""

    learning_rate: float = MISSING
    """The learning rate for the student policy."""

    gradient_length: int = MISSING
    """The number of environment steps the gradient flows back."""

    max_grad_norm: None | float = None
    """The maximum norm the gradient is clipped to."""

    flow_state_horizon: None | int = None
    """The horizon of the flow state."""

    flow_state_normalizer: None | str = None
    """The normalizer for the flow state. It matches the name of the method in the environment."""

    allow_amp: bool = False
    """Whether to use automatic mixed precision."""