# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

from isaaclab_rl.rsl_rl.modules import StudentTeacher
from isaaclab_rl.rsl_rl.networks import Memory
from isaaclab_rl.rsl_rl.utils import resolve_nn_activation

class StudentTeacherRecurrent(StudentTeacher):
    is_recurrent = True

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_policy_cfg,
        teacher_policy_ckpt,
        student_obs_meta,
        teacher_obs_meta,
        init_noise_std=0.1,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacherRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys()),
            )
        super().__init__(
            num_student_obs=num_student_obs,
            num_teacher_obs=num_teacher_obs,
            num_actions=num_actions,
            student_policy_cfg=student_policy_cfg,
            teacher_policy_ckpt=teacher_policy_ckpt,
            student_obs_meta=student_obs_meta,
            teacher_obs_meta=teacher_obs_meta,
            init_noise_std=init_noise_std,
        )

        self.student_recurrent = self.student.is_recurrent
        self.teacher_recurrent = self.teacher.is_recurrent
        assert self.student_recurrent or self.teacher_recurrent, "At least one of student_recurrent or teacher_recurrent must be True"

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is None:
            hidden_states = (None, None)
        if self.student_recurrent:
            self.student.memory_a.reset(dones, hidden_states[0])
        if self.teacher_recurrent:
            self.teacher.memory_a.reset(dones, hidden_states[1])

    def get_hidden_states(self):
        hidden_states = [None, None]
        if self.student_recurrent:
            hidden_states[0] = self.student.get_hidden_states()[0]
        if self.teacher_recurrent:
            hidden_states[1] = self.teacher.get_hidden_states()[0]
        return hidden_states

    def detach_hidden_states(self, dones=None):
        if self.student_recurrent:
            self.student.memory_a.detach_hidden_states(dones)
        if self.teacher_recurrent:
            self.teacher.memory_a.detach_hidden_states(dones)
