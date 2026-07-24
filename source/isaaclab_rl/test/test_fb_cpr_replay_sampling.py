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
FBCprExpertBuffer = _mod.FBCprExpertBuffer


def _make_priority_expert_buffer() -> FBCprExpertBuffer:
    buffer = FBCprExpertBuffer.__new__(FBCprExpertBuffer)
    buffer.device = torch.device("cpu")
    buffer._motion_names = ["short", "medium", "long"]
    buffer._lengths = [10, 20, 40]
    buffer._lengths_t = torch.tensor(buffer._lengths)
    buffer._length_proportional_priors = True
    buffer._eval_priority_scores = torch.ones(3)
    buffer._tracking_failure_ema = torch.zeros(3)
    buffer._tracking_failure_bin_frames = 0
    buffer._init_tracking_failure_bins()
    buffer._failure_priority_scale = 0.0
    buffer._failure_priority_max_multiplier = 1.0
    buffer._priority_length_weights = torch.tensor([10.0, 20.0, 40.0])
    buffer._recompute_priorities()
    return buffer


def _make_buffer(
    sampling_mode: str = "uniform_transition",
) -> FBCprReplayBuffer:
    return FBCprReplayBuffer(
        capacity=16,
        num_envs=1,
        obs_space={"state": (1,)},
        action_dim=1,
        z_dim=1,
        aux_reward_names=[],
        device="cpu",
        pin_memory=False,
        replay_sampling_mode=sampling_mode,
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


def test_reset_boundary_transition_is_never_sampled():
    buffer = _make_buffer(sampling_mode="uniform_transition")

    for value, terminated, boundary in (
        (0.0, False, False),
        (1.0, False, False),   # failure-causing row
        (100.0, True, True),   # first observation after synthetic reset
        (101.0, False, False),
        (102.0, False, False),
    ):
        buffer.extend(
            {
                "observation": {"state": torch.tensor([[value]])},
                "action": torch.zeros(1, 1),
                "z": torch.zeros(1, 1),
                "terminated": torch.tensor([[terminated]]),
                "truncated": torch.tensor([[boundary]]),
                "aux_rewards": {},
            }
        )

    torch.manual_seed(29)
    batch = buffer.sample_flat(30_000)
    current = batch["observation"]["state"][:, 0]
    following = batch["next"]["observation"]["state"][:, 0]
    pairs = set(zip(current.tolist(), following.tolist()))

    assert pairs == {(0.0, 1.0), (100.0, 101.0), (101.0, 102.0)}
    assert not torch.any((current == 1.0) & (following == 100.0))


def test_expert_positive_context_moves_but_first_t_next_window_does_not():
    buffer = FBCprExpertBuffer.__new__(FBCprExpertBuffer)
    buffer.seq_length = 16
    buffer.device = torch.device("cpu")
    buffer._lengths_t = torch.tensor([80])
    buffer._priorities = torch.ones(1)
    buffer._motion_obs_starts = torch.tensor([0])
    frames = torch.arange(80, dtype=torch.float32).unsqueeze(-1)
    buffer._flat_state = frames
    buffer._flat_priv = frames.clone()
    buffer._flat_last_action = frames.clone()
    buffer._flat_history_actor = frames.clone()
    buffer.requires_terrain_t = torch.zeros(1, dtype=torch.bool)
    buffer._emit_anchored_pose = False

    widths = torch.tensor([1, 2, 4, 8, 16])
    batch = buffer.sample(
        batch_size=80,
        seq_length=16,
        mean_widths=widths,
    )
    obs = batch["observation"]["state"].view(5, 16)
    next_obs = batch["next"]["observation"]["state"].view(5, 16)

    expected_offsets = torch.tensor([-8, -7, -6, -4, 0])
    torch.testing.assert_close(
        obs[:, 0] - next_obs[:, 0],
        expected_offsets.float() - 1.0,
    )
    # The B input remains a contiguous window starting at current_time + 1.
    torch.testing.assert_close(
        next_obs[:, 1:] - next_obs[:, :-1],
        torch.ones(5, 15),
    )


def test_expert_chunk_prefetch_preserves_mean_width_schedule():
    buffer = FBCprExpertBuffer.__new__(FBCprExpertBuffer)
    buffer.seq_length = 16
    buffer.device = torch.device("cpu")
    buffer._lengths_t = torch.tensor([80])
    buffer._priorities = torch.ones(1)
    buffer._motion_obs_starts = torch.tensor([0])
    frames = torch.arange(80, dtype=torch.float32).unsqueeze(-1)
    buffer._flat_state = frames
    buffer._flat_priv = frames.clone()
    buffer._flat_last_action = frames.clone()
    buffer._flat_history_actor = frames.clone()
    buffer._emit_anchored_pose = False

    widths = torch.tensor([1, 2, 4, 8])
    chunks = buffer.sample_chunks(
        batch_size=32,
        num_chunks=2,
        target_device="cpu",
        mean_widths=widths,
    )

    assert len(chunks) == 2
    torch.testing.assert_close(chunks[0]["_mean_widths"], widths[:2])
    torch.testing.assert_close(chunks[1]["_mean_widths"], widths[2:])
    obs = torch.cat(
        [chunk["observation"]["state"] for chunk in chunks]
    ).view(4, 16)
    next_obs = torch.cat(
        [chunk["next"]["observation"]["state"] for chunk in chunks]
    ).view(4, 16)
    expected_offsets = torch.tensor([-8, -7, -6, -4])
    torch.testing.assert_close(
        obs[:, 0] - next_obs[:, 0],
        expected_offsets.float() - 1.0,
    )


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


def test_flat_legacy_sampling_is_uniform_over_trajectories():
    buffer = _make_buffer(sampling_mode="uniform_trajectory")
    # The trajectories have one and six valid transitions. Legacy sampling
    # assigns half the draws to each trajectory, then samples within it.
    buffer._start_idx = torch.tensor([[0, 0], [4, 0]])
    buffer._lengths = torch.tensor([2, 7])
    buffer._recompute_traj_info = False
    buffer._idx = 12
    buffer._obs["state"][:, 0, 0] = torch.arange(16)

    torch.manual_seed(19)
    sampled = buffer.sample_flat(70_000)["observation"]["state"][:, 0]
    short_count = (sampled == 0).sum()
    long_count = (sampled >= 4).sum()

    assert abs(int(short_count) - 35_000) < 600
    assert abs(int(long_count) - 35_000) < 600


def test_sequence_legacy_sampling_is_uniform_over_trajectories():
    buffer = _make_buffer(sampling_mode="uniform_trajectory")
    # At seq_length=2 these trajectories have one and five valid starts.
    buffer._start_idx = torch.tensor([[0, 0], [4, 0]])
    buffer._lengths = torch.tensor([3, 7])
    buffer._recompute_traj_info = False
    buffer._idx = 12
    buffer._obs["state"][:, 0, 0] = torch.arange(16)

    torch.manual_seed(23)
    sampled = buffer.sample(60_000, seq_length=2)["observation"]["state"]
    starts = sampled[::2, 0]
    short_count = (starts == 0).sum()
    long_count = (starts >= 4).sum()

    assert abs(int(short_count) - 15_000) < 450
    assert abs(int(long_count) - 15_000) < 450


def test_replay_sampling_mode_is_validated():
    try:
        _make_buffer(sampling_mode="not-a-mode")
    except ValueError as exc:
        assert "uniform_transition" in str(exc)
        assert "uniform_trajectory" in str(exc)
    else:
        raise AssertionError("invalid replay sampling mode was accepted")


def test_tracking_failure_priority_does_not_change_expert_sampling():
    buffer = _make_priority_expert_buffer()
    buffer._tracking_failure_bin_frames = 10
    buffer._init_tracking_failure_bins()
    buffer._tracking_bin_success_ema.fill_(1.0)
    expert_before = buffer._priorities.clone()

    buffer.update_tracking_bin_success_statistics(
        bin_ids=torch.tensor([0, 0, 1, 2]),
        succeeded=torch.tensor([True, False, False, False]),
        ema_decay=0.0,
        priority_scale=3.0,
        max_multiplier=4.0,
    )

    torch.testing.assert_close(buffer._priorities, expert_before)
    # Duplicate bin 0 aggregates to 50% success; both motion-1 bins fail.
    torch.testing.assert_close(
        buffer._tracking_failure_ema, torch.tensor([0.5, 1.0, 0.0])
    )
    assert buffer._tracking_priorities[1] > buffer._priorities[1]
    assert buffer._tracking_priorities[2] < buffer._priorities[2]


def test_tracking_failure_bins_cover_motion_segments_and_map_frames():
    buffer = _make_priority_expert_buffer()
    buffer._lengths = [120, 55]
    buffer._lengths_t = torch.tensor(buffer._lengths)
    buffer._motion_names = ["long", "short"]
    buffer._eval_priority_scores = torch.ones(2)
    buffer._tracking_failure_ema = torch.zeros(2)
    buffer._tracking_failure_bin_frames = 50
    buffer._init_tracking_failure_bins()

    assert buffer._tracking_bin_motion_ids.tolist() == [0, 0, 0, 1, 1]
    assert buffer._tracking_bin_starts.tolist() == [0, 50, 100, 0, 50]
    assert buffer._tracking_bin_ends.tolist() == [50, 100, 120, 50, 55]
    bin_ids = buffer.tracking_bin_ids(
        torch.tensor([0, 0, 0, 1, 1]),
        torch.tensor([0, 99, 119, 49, 54]),
    )
    assert bin_ids.tolist() == [0, 1, 2, 3, 4]


def test_tracking_bin_success_ema_aggregates_duplicate_attempts():
    buffer = _make_priority_expert_buffer()
    buffer._tracking_failure_bin_frames = 10
    buffer._init_tracking_failure_bins()

    buffer.update_tracking_bin_success_statistics(
        bin_ids=torch.tensor([0, 0, 1]),
        succeeded=torch.tensor([True, False, True]),
        ema_decay=0.0,
        priority_scale=3.0,
        max_multiplier=4.0,
    )

    torch.testing.assert_close(
        buffer._tracking_bin_success_ema[:2],
        torch.tensor([0.5, 1.0]),
    )
    # Motion-level diagnostics are the duration-weighted mean bin failure.
    assert 0.0 < float(buffer._tracking_failure_ema[0]) < 1.0


def test_tracking_failure_bin_sampling_prefers_low_success_bins():
    buffer = _make_priority_expert_buffer()
    buffer._lengths = [20]
    buffer._lengths_t = torch.tensor(buffer._lengths)
    buffer._motion_names = ["motion"]
    buffer._eval_priority_scores = torch.ones(1)
    buffer._tracking_failure_ema = torch.zeros(1)
    buffer._priority_length_weights = torch.ones(1)
    buffer._tracking_failure_bin_frames = 10
    buffer._init_tracking_failure_bins()
    buffer._tracking_bin_success_ema[:] = torch.tensor([1.0, 0.0])
    buffer._failure_priority_scale = 3.0
    buffer._failure_priority_max_multiplier = 4.0

    torch.manual_seed(19)
    sampled = buffer.sample_tracking_failure_bins(50_000)["bin_ids"]
    counts = torch.bincount(sampled, minlength=2).float()
    ratio = counts[1] / counts[0]
    assert 3.7 < float(ratio) < 4.3


def test_eval_priority_update_preserves_tracking_failure_ema():
    buffer = _make_priority_expert_buffer()
    buffer._tracking_failure_bin_frames = 10
    buffer._init_tracking_failure_bins()
    buffer._tracking_bin_success_ema.fill_(1.0)
    buffer.update_tracking_bin_success_statistics(
        bin_ids=torch.tensor([1, 2]),
        succeeded=torch.tensor([False, False]),
        ema_decay=0.0,
        priority_scale=3.0,
        max_multiplier=4.0,
    )

    buffer.update_priorities(torch.tensor([4.0, 2.0, 1.0]))

    torch.testing.assert_close(
        buffer._tracking_failure_ema, torch.tensor([0.0, 1.0, 0.0])
    )
    # Length weighting is composed exactly once with the new eval scores.
    expected_expert = torch.tensor([40.0, 40.0, 40.0])
    expected_expert /= expected_expert.sum()
    torch.testing.assert_close(buffer._priorities, expected_expert)
    assert buffer._tracking_priorities[1] > buffer._priorities[1]


def test_expert_priority_state_round_trip():
    source = _make_priority_expert_buffer()
    source._tracking_failure_bin_frames = 10
    source._init_tracking_failure_bins()
    source._tracking_bin_success_ema.fill_(1.0)
    source.update_priorities(torch.tensor([1.0, 2.0, 3.0]))
    source.update_tracking_bin_success_statistics(
        bin_ids=torch.tensor([3]),
        succeeded=torch.tensor([False]),
        ema_decay=0.5,
        priority_scale=2.0,
        max_multiplier=3.0,
    )
    restored = _make_priority_expert_buffer()
    restored._tracking_failure_bin_frames = 10
    restored._init_tracking_failure_bins()

    assert restored.load_priority_state_dict(source.priority_state_dict())
    torch.testing.assert_close(restored._priorities, source._priorities)
    torch.testing.assert_close(
        restored._tracking_priorities, source._tracking_priorities
    )
    torch.testing.assert_close(
        restored._tracking_failure_ema, source._tracking_failure_ema
    )


def test_tracking_bin_success_state_round_trip():
    source = _make_priority_expert_buffer()
    source._tracking_failure_bin_frames = 10
    source._init_tracking_failure_bins()
    source.update_tracking_bin_success_statistics(
        bin_ids=torch.tensor([0, 1]),
        succeeded=torch.tensor([True, False]),
        ema_decay=0.5,
        priority_scale=3.0,
        max_multiplier=4.0,
    )
    restored = _make_priority_expert_buffer()
    restored._tracking_failure_bin_frames = 10
    restored._init_tracking_failure_bins()

    assert restored.load_priority_state_dict(source.priority_state_dict())
    torch.testing.assert_close(
        restored._tracking_bin_success_ema,
        source._tracking_bin_success_ema,
    )


def test_expert_priority_state_rejects_different_equal_sized_shard():
    source = _make_priority_expert_buffer()
    restored = _make_priority_expert_buffer()
    restored._motion_names = ["other-a", "other-b", "other-c"]

    assert not restored.load_priority_state_dict(
        source.priority_state_dict()
    )


def _make_tracking_expert_buffer(
    circular_wrap: bool,
    requires_terrain: bool,
) -> FBCprExpertBuffer:
    buffer = FBCprExpertBuffer.__new__(FBCprExpertBuffer)
    buffer.seq_length = 8
    buffer.device = torch.device("cpu")
    buffer._expert_tracking_circular_wrap = circular_wrap
    buffer._lengths = [60]
    buffer._lengths_t = torch.tensor([60])
    buffer._priorities = torch.ones(1)
    buffer._motion_obs_starts = torch.tensor([0])
    frames = torch.arange(60, dtype=torch.float32).unsqueeze(-1)
    buffer._flat_state = frames
    buffer._flat_priv = frames.clone()
    buffer._flat_last_action = frames.clone()
    buffer._flat_history_actor = frames.clone()
    buffer.requires_terrain_t = torch.tensor([requires_terrain])
    buffer._emit_anchored_pose = False
    return buffer


def test_tracking_short_motion_requires_legacy_wrap():
    buffer = _make_tracking_expert_buffer(
        circular_wrap=False,
        requires_terrain=False,
    )
    try:
        buffer.sample_tracking_trajectories(num_trajs=1, traj_length=70)
    except RuntimeError as exc:
        assert "at least 71 frames" in str(exc)
    else:
        raise AssertionError("short tracking motion was accepted without legacy wrap")


def test_tracking_short_nonterrain_motion_wraps_circularly():
    buffer = _make_tracking_expert_buffer(
        circular_wrap=True,
        requires_terrain=False,
    )
    batch = buffer.sample_tracking_trajectories(num_trajs=1, traj_length=70)

    expected_obs = torch.arange(70) % 59
    expected_next = (torch.arange(70) + 1) % 59
    torch.testing.assert_close(
        batch["observation"]["state"][:, 0], expected_obs.float()
    )
    torch.testing.assert_close(
        batch["next_observation"]["state"][:, 0], expected_next.float()
    )


def test_tracking_short_terrain_motion_clamps_to_last_usable_frame():
    buffer = _make_tracking_expert_buffer(
        circular_wrap=True,
        requires_terrain=True,
    )
    batch = buffer.sample_tracking_trajectories(num_trajs=1, traj_length=70)

    expected_obs = torch.arange(70).clamp(max=58)
    expected_next = (torch.arange(70) + 1).clamp(max=58)
    torch.testing.assert_close(
        batch["observation"]["state"][:, 0], expected_obs.float()
    )
    torch.testing.assert_close(
        batch["next_observation"]["state"][:, 0], expected_next.float()
    )


def test_forced_tracking_bin_start_pads_without_crossing_motion_end():
    buffer = _make_tracking_expert_buffer(
        circular_wrap=False,
        requires_terrain=False,
    )
    batch = buffer.sample_tracking_trajectories(
        num_trajs=1,
        traj_length=10,
        motion_ids=torch.tensor([0]),
        starts=torch.tensor([55]),
        pad_to_motion_end=True,
    )

    torch.testing.assert_close(
        batch["observation"]["state"][:, 0],
        torch.tensor([55, 56, 57, 58, 59, 59, 59, 59, 59, 59]).float(),
    )
    torch.testing.assert_close(
        batch["next_observation"]["state"][:, 0],
        torch.tensor([56, 57, 58, 59, 59, 59, 59, 59, 59, 59]).float(),
    )


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


def test_multi_env_segments_are_grouped_and_length_correct():
    buffer = FBCprReplayBuffer(
        capacity=24,
        num_envs=3,
        obs_space={"state": (1,)},
        action_dim=1,
        z_dim=1,
        aux_reward_names=[],
        device="cpu",
        pin_memory=False,
    )
    buffer._idx = 8
    buffer._truncated[2, 1] = True
    buffer._truncated[5, 0] = True
    buffer._truncated[6, 1] = True
    buffer._recompute_traj_info = True
    buffer._ensure_traj_info()

    assert buffer._start_idx.tolist() == [
        [0, 0], [5, 0],
        [0, 1], [2, 1], [6, 1],
        [0, 2],
    ]
    assert buffer._lengths.tolist() == [5, 3, 2, 4, 2, 8]


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
