# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCritic for BFM-Adapter: zero-initialized actor output."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic


class ActorCriticAdapter(ActorCritic):
    """ActorCritic with zero-initialized actor last layer.

    The actor outputs a residual that starts at zero (no correction).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        last_layer = self.actor[-1]
        assert isinstance(last_layer, nn.Linear)
        with torch.no_grad():
            last_layer.weight.zero_()
            last_layer.bias.zero_()
