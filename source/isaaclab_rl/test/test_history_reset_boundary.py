# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Torch tests: replay ``history_reset`` vs ``truncated`` semantics.

``truncated`` is a SAMPLING boundary (no (t, t+1) pair may straddle it).
``history_reset`` marks rows that follow a real env reset (history rings
zeroed). The history recompose must zero frames reaching back past a reset,
but must keep frames across a boundary marked while the env state was
preserved (the conservative post-eval boundary).
"""

from __future__ import annotations

import importlib.util
import os

import torch


_STORAGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "isaaclab_rl", "rsl_rl", "storage",
    "fb_cpr_storage.py",
)
_spec = importlib.util.spec_from_file_location("fb_cpr_storage", _STORAGE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FBCprReplayBuffer = _mod.FBCprReplayBuffer

BLOCKS = [("act", 2), ("grav", 1)]
FRAME_DIM = 3
H = 3


def _make_buffer(time_cap: int = 32) -> FBCprReplayBuffer:
    return FBCprReplayBuffer(
        capacity=time_cap,
        num_envs=1,
        obs_space={"history_actor": (H * FRAME_DIM,)},
        action_dim=1,
        z_dim=1,
        aux_reward_names=[],
        device="cpu",
        pin_memory=False,
        history_recompose={"H": H, "blocks": BLOCKS},
    )


def _blob(value: float) -> torch.Tensor:
    """Full history blob whose every frame (incl. newest) equals ``value``."""
    return torch.full((1, H * FRAME_DIM), float(value))


def _extend(buf, t, *, truncated, history_reset=None):
    batch = {
        "observation": {"history_actor": _blob(t)},
        "action": torch.zeros(1, 1),
        "z": torch.zeros(1, 1),
        "terminated": torch.zeros(1, 1, dtype=torch.bool),
        "truncated": torch.tensor([[truncated]]),
    }
    if history_reset is not None:
        batch["history_reset"] = torch.tensor([[history_reset]])
    buf.extend(batch)


def _frames(buf, t):
    """Recomposed per-offset frame values at row t: [H] (newest lag first)."""
    out = buf._recompose_history(torch.tensor([t]), torch.tensor([0]))
    # Block-major layout: block 'act' occupies [0 : H*2], frame-major newest
    # first; take the first column of each frame of the 'act' block.
    return out[0, : H * 2].view(H, 2)[:, 0]


def _fill(buf):
    # Row 0: real reset. Rows 1..2 normal. Row 3: boundary WITHOUT reset (the
    # post-eval marker). Rows 4..6 normal.
    _extend(buf, 0, truncated=True, history_reset=True)
    _extend(buf, 1, truncated=False, history_reset=False)
    _extend(buf, 2, truncated=False, history_reset=False)
    _extend(buf, 3, truncated=True, history_reset=False)
    for t in (4, 5, 6):
        _extend(buf, t, truncated=False, history_reset=False)


def test_boundary_without_reset_keeps_history():
    buf = _make_buffer()
    _fill(buf)
    # Frames at row 4 reach rows 4, 3, 2 — across the non-reset boundary at 3.
    assert torch.equal(_frames(buf, 4), torch.tensor([4.0, 3.0, 2.0]))
    # Frames at row 1 reach rows 1, 0, -1: the real reset at row 0 zeroes lag>=1.
    assert torch.equal(_frames(buf, 1), torch.tensor([1.0, 0.0, 0.0]))
    # Row 2: rows 2, 1, 0 — row 0 is the reset row, so lag 2 (row 0) is zeroed
    # (its stored frame predates the episode).
    assert torch.equal(_frames(buf, 2), torch.tensor([2.0, 1.0, 0.0]))


def test_boundary_still_blocks_transition_sampling():
    buf = _make_buffer()
    _fill(buf)
    torch.manual_seed(0)
    seen = set()
    for _ in range(50):
        batch = buf.sample_flat(64)
        # Newest frame of the CURRENT row identifies t.
        cur = batch["observation"]["history_actor"][:, 0]
        seen.update(int(v) for v in cur.tolist())
    # Row 2 -> row 3 straddles the sampling boundary: never drawn. Row 3 -> 4
    # is inside the new segment: drawn. Row 6 has no successor.
    assert 2 not in seen
    assert 3 in seen
    assert 6 not in seen
    assert {0, 1, 4, 5} <= seen


def test_missing_history_reset_falls_back_to_truncated_on_extend():
    buf = _make_buffer()
    for t, trunc in enumerate([True, False, False, True, False]):
        _extend(buf, t, truncated=trunc)  # no history_reset key
    assert torch.equal(buf._history_reset[:5], buf._truncated[:5])
    # Legacy semantics: the boundary at row 3 is treated as a reset, so the
    # frame stored AT row 3 (pushed before that reset) and older are zeroed.
    assert torch.equal(_frames(buf, 4), torch.tensor([4.0, 0.0, 0.0]))


def test_legacy_state_dict_without_history_reset_loads():
    src = _make_buffer()
    _fill(src)
    sd = src.state_dict()
    assert "_history_reset" in sd
    del sd["_history_reset"]  # replay saved before the field existed
    dst = _make_buffer()
    dst.load_state_dict(sd)
    assert torch.equal(dst._history_reset, dst._truncated)
    assert torch.equal(_frames(dst, 4), torch.tensor([4.0, 0.0, 0.0]))


def test_state_dict_round_trips_history_reset():
    src = _make_buffer()
    _fill(src)
    dst = _make_buffer()
    dst.load_state_dict(src.state_dict())
    assert torch.equal(dst._history_reset, src._history_reset)
    assert torch.equal(_frames(dst, 4), torch.tensor([4.0, 3.0, 2.0]))
