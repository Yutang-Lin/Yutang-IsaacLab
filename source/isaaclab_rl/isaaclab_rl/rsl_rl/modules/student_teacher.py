# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from .actor_critic import ActorCritic
from .actor_critic_moe import ActorCriticMoE


class StudentTeacher(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_policy_cfg,
        teacher_policy_ckpt,
        init_noise_std=0.1,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.loaded_teacher = False  # indicates if teacher has been loaded

        if isinstance(num_student_obs, tuple):
            num_student_obs, num_student_priv_obs = num_student_obs
        else:
            num_student_priv_obs = 1

        # student
        student_policy_class = eval(student_policy_cfg.pop("class_name"))
        self.student: ActorCritic = student_policy_class(num_student_obs, num_student_priv_obs, num_actions, **student_policy_cfg)

        # teacher
        teacher_policy_ckpt = torch.load(teacher_policy_ckpt, map_location="cpu", weights_only=False)
        if 'obs_norm_state_dict' in teacher_policy_ckpt:
            self.obs_norm_state_dict = teacher_policy_ckpt["obs_norm_state_dict"]
        else:
            self.obs_norm_state_dict = None

        teacher_policy_cfg = teacher_policy_ckpt["policy_cfg"]
        teacher_policy_class = eval(teacher_policy_cfg.pop("class_name"))
        teacher_policy_args = teacher_policy_cfg.pop("_args")
        assert num_teacher_obs == teacher_policy_args[0], f"Mismatch in number of teacher observations: num_teacher_obs: {num_teacher_obs}, teacher_policy_args[0]: {teacher_policy_args[0]}"
        assert num_actions == teacher_policy_args[2], f"Mismatch in number of actions: num_actions: {num_actions}, teacher_policy_args[2]: {teacher_policy_args[2]}"
        self.teacher = teacher_policy_class(*teacher_policy_args, **teacher_policy_cfg)
        self.teacher.load_state_dict(teacher_policy_ckpt["model_state_dict"], strict=True)
        if hasattr(self.teacher, "actor"):
            self.teacher = self.teacher.actor
        self.teacher.eval()

        self.loaded_teacher = True
        print(f"Student Model: {self.student}")
        print(f"Teacher Model: {self.teacher}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False # type: ignore

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def extra_loss(self):
        return self.student.extra_loss()
    
    def pre_train(self):
        self.student.pre_train()

    def after_train(self):
        self.student.after_train()

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.student.act_inference(observations)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        actions_mean = self.student.act_inference(observations)
        return actions_mean

    def evaluate(self, teacher_observations):
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters.
        """

        # ignore teacher parameters
        teacher_keys = [key for key in state_dict.keys() if "teacher" in key]
        rest_params = {key: value for key, value in state_dict.items() if key not in teacher_keys}
        super().load_state_dict(rest_params, strict=False)
        return True

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass
