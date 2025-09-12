# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from .transformer_policy import TransformerPolicy

class TransformerPolicyResidual(TransformerPolicy):
    def __init__(self, input_size, output_size, base_policy_path: str, is_actor: bool = True, *args, **kwargs):
        residual_input_size = input_size + output_size
        super().__init__(residual_input_size, output_size, *args, **kwargs)

        base_policy_state_dict = torch.load(base_policy_path, map_location="cpu", weights_only=False)
        base_policy_state_dict = base_policy_state_dict["model_state_dict"]
        replace_prefix = "actor." if is_actor else "critic."
        base_policy_state_dict = {k.replace(replace_prefix, ""): v for k, v in base_policy_state_dict.items() if replace_prefix in k}
        self.base_policy = TransformerPolicy(input_size, output_size, *args, **kwargs)
        self.base_policy.load_state_dict(base_policy_state_dict, strict=True)
        self.base_policy.requires_grad_(False)
        self._init_weights()

    def _init_weights(self):
        self.output_proj[-1].weight.data.fill_(0.0)
        self.output_proj[-1].bias.data.fill_(0.0)

    @torch.no_grad()
    def forward_base(self, input: torch.Tensor):
        if self.base_policy.training:
            self.base_policy.eval()
        return self.base_policy(input)

    def forward(self, input: torch.Tensor):
        base_output = self.forward_base(input).detach()
        input = torch.cat([input, base_output], dim=-1)
        return super().forward(input) + base_output