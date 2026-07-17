# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Torch tests for trajectory-aware FB-CPR replay sampling."""

from __future__ import annotations

import importlib.util
import os

import torch


_STORAGE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "isaaclab_rl",
    "rsl_rl",
    "storage",
    "fb_cpr_storage.py",
)
_spec = importlib.util.spec_from_file_location("fb_cpr_storage", _STORAGE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FBCprReplayBuffer = _mod.FBCprReplayBuffer


def _make_buffer() -> FBCprReplayBuffer:
    return FBCprReplayBuffer(
        capacity=16,
        num_envs=1,
        obs_space={"state": (1,)},
        action_dim=1,
        z_dim=1,
        aux_reward_names=[],
        device="cpu",
        pin_memory=False,
    )


def test_flat_sampling_is_uniform_over_transitions():
    buffer = _make_buffer()
    # One short trajectory with one valid transition and one long trajectory
    # with six. A trajectory-uniform sampler would incorrectly split 50/50.
    buffer._start_idx = torch.tensor([[0, 0], [4, 0]])
    buffer._lengths = torch.tensor([3, 8])
    buffer._recompute_traj_info = False
    buffer._idx = 12
    buffer._obs["state"][:, 0, 0] = torch.arange(16)

    torch.manual_seed(7)
    sampled = buffer.sample_flat(70_000)["observation"]["state"][:, 0]
    counts = torch.bincount(sampled.long(), minlength=16)

    expected_rows = torch.tensor([0, 4, 5, 6, 7, 8, 9])
    expected = torch.full((7,), 10_000.0)
    assert torch.all((counts[expected_rows].float() - expected).abs() < 500)
    assert counts.sum() == 70_000


def test_sequence_sampling_is_uniform_over_valid_starts():
    buffer = _make_buffer()
    # At seq_length=2 these trajectories contribute one and five starts.
    buffer._start_idx = torch.tensor([[0, 0], [4, 0]])
    buffer._lengths = torch.tensor([4, 8])
    buffer._recompute_traj_info = False
    buffer._idx = 12
    buffer._obs["state"][:, 0, 0] = torch.arange(16)

    torch.manual_seed(11)
    sampled = buffer.sample(60_000, seq_length=2)["observation"]["state"]
    starts = sampled[::2, 0]
    counts = torch.bincount(starts.long(), minlength=16)

    expected_rows = torch.tensor([0, 4, 5, 6, 7, 8])
    expected = torch.full((6,), 5_000.0)
    assert torch.all((counts[expected_rows].float() - expected).abs() < 350)
    assert counts.sum() == 30_000
