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
    buffer._lengths = torch.tensor([2, 7])
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
    buffer._lengths = torch.tensor([3, 7])
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


def test_reset_markers_start_new_segments():
    buffer = _make_buffer()
    buffer._idx = 8
    buffer._truncated[3, 0] = True
    buffer._recompute_traj_info = True
    buffer._ensure_traj_info()

    assert buffer._start_idx.tolist() == [[0, 0], [3, 0]]
    assert buffer._lengths.tolist() == [3, 5]

    torch.manual_seed(13)
    buffer._obs["state"][:, 0, 0] = torch.arange(16)
    sampled = buffer.sample_flat(20_000)["observation"]["state"][:, 0].long()
    assert not bool((sampled == 2).any())  # 2 -> 3 crosses into the reset row
    assert bool((sampled == 3).any())      # 3 -> 4 is the first valid new-episode pair


def test_full_buffer_actor_window_never_wraps_into_future():
    buffer = FBCprReplayBuffer(
        capacity=8,
        num_envs=1,
        obs_space={"state": (1,)},
        action_dim=1,
        z_dim=1,
        aux_reward_names=[],
        device="cpu",
        pin_memory=False,
        actor_window_len=3,
    )
    buffer._is_full = True
    buffer._idx = 3  # row 3 is oldest; row 2 is newest
    buffer._obs["state"][:, 0, 0] = torch.arange(8)

    window = buffer._gather_actor_window(
        torch.tensor([3]),
        torch.tensor([0]),
    )
    assert window["valid"].tolist() == [[False, False, False, True]]


def test_wrapped_replay_segments_follow_chronological_order():
    buffer = FBCprReplayBuffer(
        capacity=8,
        num_envs=1,
        obs_space={"state": (1,)},
        action_dim=1,
        z_dim=1,
        aux_reward_names=[],
        device="cpu",
        pin_memory=False,
    )
    buffer._is_full = True
    buffer._idx = 3  # chronological rows: 3,4,5,6,7,0,1,2
    buffer._truncated[6, 0] = True
    buffer._truncated[1, 0] = True
    buffer._obs["state"][:, 0, 0] = torch.arange(8)
    buffer._recompute_traj_info = True
    buffer._ensure_traj_info()

    assert buffer._start_idx.tolist() == [[3, 0], [6, 0], [1, 0]]
    assert buffer._lengths.tolist() == [3, 3, 2]

    torch.manual_seed(17)
    sampled = buffer.sample_flat(30_000)["observation"]["state"][:, 0].long()
    assert set(sampled.unique().tolist()) == {1, 3, 4, 6, 7}
    assert not bool((sampled == 5).any())  # 5 -> 6 crosses a reset
    assert not bool((sampled == 0).any())  # 0 -> 1 crosses a reset
    assert not bool((sampled == 2).any())  # newest row has no stored successor
