# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay / expert / z buffers for the BFM-Zero FB-CPR-Aux agent (rsl_rl port).

This module re-implements BFM-Zero's ``DictBuffer`` (per-transition replay),
``TrajectoryDictBuffer`` (expert trajectory slicer) and ``ZBuffer`` (rolling
z reservoir) with the rsl_rl conventions used by this codebase. No tensordict
or torch._pytree dependencies -- everything is plain dicts of tensors.

All three classes expose a common dict-of-tensors sample format:

    {
        "observation":    {state, privileged_state, last_action, history_actor},
        "action":         [B, action_dim],
        "z":              [B, z_dim],
        "next": {
            "observation":   {state, privileged_state, last_action, history_actor},
            "terminated":    [B, 1]  (bool),
        },
        "aux_rewards": {name: [B, 1]}    # train replay only
    }

Samples stay on the buffer's own device; the caller is responsible for moving
the batch to the training device.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _space_shape(space) -> tuple[int, ...]:
    """Return the per-sample shape of a ``gymnasium.spaces.Box``-like object."""
    return tuple(int(s) for s in space.shape)


def _index_obs_dict(obs: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in obs.items()}


# ---------------------------------------------------------------------------
# 1. FBCprReplayBuffer -- circular dict-of-tensors replay
# ---------------------------------------------------------------------------

class FBCprReplayBuffer:
    """Flat circular replay buffer storing (obs, action, z, next_obs, next_terminated, aux_rewards).

    Transitions are appended as batches via ``add(batch_dict)``; when the write
    cursor reaches ``capacity`` it wraps, so older transitions get overwritten
    in FIFO order. Sampling is uniform over currently valid indices (``len(self)``).
    """

    def __init__(
        self,
        capacity: int,
        obs_space: Any,
        action_dim: int,
        z_dim: int,
        aux_reward_names: list[str],
        device: str | torch.device = "cpu",
        pin_memory: bool | None = None,
    ) -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        self.z_dim = int(z_dim)
        self.aux_reward_names = list(aux_reward_names)

        # Only pin memory when storing on CPU (pin_memory is a no-op otherwise
        # and raises on some GPU tensor constructors). Caller can force.
        if pin_memory is None:
            pin_memory = self.device.type == "cpu"
        self._pin_memory = bool(pin_memory)

        # obs_space is expected to be gymnasium Dict of Box entries; we only
        # peek at ``.spaces`` to get per-key shapes. We keep a list of ordered
        # keys to keep the dict layout stable across calls.
        if hasattr(obs_space, "spaces"):
            self._obs_shapes = {k: _space_shape(v) for k, v in obs_space.spaces.items()}
        else:
            # Allow passing an already-resolved dict[str, shape].
            self._obs_shapes = {k: tuple(v) for k, v in dict(obs_space).items()}

        # Pre-allocate storage.
        self._obs: dict[str, torch.Tensor] = {
            k: self._alloc((self.capacity, *shape))
            for k, shape in self._obs_shapes.items()
        }
        self._next_obs: dict[str, torch.Tensor] = {
            k: self._alloc((self.capacity, *shape))
            for k, shape in self._obs_shapes.items()
        }
        self._action = self._alloc((self.capacity, self.action_dim))
        self._z = self._alloc((self.capacity, self.z_dim))
        self._next_terminated = self._alloc((self.capacity, 1), dtype=torch.bool)
        self._aux_rewards: dict[str, torch.Tensor] = {
            name: self._alloc((self.capacity, 1)) for name in self.aux_reward_names
        }

        # Circular write cursor + "have we wrapped at least once" flag.
        self._idx = 0
        self._is_full = False

    # -- allocation helper -------------------------------------------------

    def _alloc(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        # NOTE: ``torch.zeros(...).pin_memory()`` is NOT reliably zero: the
        # pinned-memory allocator returns uninitialized pages and the
        # subsequent copy from the non-pinned source has (in some PyTorch
        # builds) produced garbage for large allocations. Allocate pinned
        # directly, then explicitly zero_() — cheap one-time cost at init
        # that prevents 1e37-valued "uninitialized read" NaN poisoning of
        # downstream BatchNorm running stats.
        if self._pin_memory and self.device.type == "cpu":
            t = torch.empty(shape, dtype=dtype, pin_memory=True)
        else:
            t = torch.empty(shape, dtype=dtype, device=self.device)
        t.zero_()
        return t

    # -- public API --------------------------------------------------------

    def __len__(self) -> int:
        return self.capacity if self._is_full else self._idx

    @property
    def full(self) -> bool:
        return self._is_full

    @torch.no_grad()
    def add(self, batch_dict: dict) -> None:
        """Append a batch of transitions, wrapping around on overflow.

        ``batch_dict`` must follow the same shape as ``sample()`` output: an
        ``observation`` dict, ``action``, ``z``, ``next.observation`` dict,
        ``next.terminated``, and (optionally) ``aux_rewards`` sub-dict.
        """
        obs = batch_dict["observation"]
        next_obs = batch_dict["next"]["observation"]
        # Use any obs key to read batch size.
        any_key = next(iter(self._obs_shapes))
        n = obs[any_key].shape[0]
        if n == 0:
            return
        if n > self.capacity:
            raise ValueError(
                f"Batch size {n} larger than replay capacity {self.capacity}; "
                "split into smaller chunks or increase capacity."
            )

        end = self._idx + n
        if end <= self.capacity:
            sl1 = slice(self._idx, end)
            self._write_slice(sl1, obs, next_obs, batch_dict, src_slice=slice(0, n))
            self._idx = end
            if self._idx == self.capacity:
                self._is_full = True
                self._idx = 0
        else:
            # Wrap: write [self._idx:capacity], then [0:remainder].
            first = self.capacity - self._idx
            sl1 = slice(self._idx, self.capacity)
            sl2 = slice(0, n - first)
            self._write_slice(sl1, obs, next_obs, batch_dict, src_slice=slice(0, first))
            self._write_slice(sl2, obs, next_obs, batch_dict, src_slice=slice(first, n))
            self._is_full = True
            self._idx = n - first

    def _write_slice(
        self,
        dst: slice,
        obs: dict,
        next_obs: dict,
        batch_dict: dict,
        src_slice: slice,
    ) -> None:
        # CRITICAL: when src is on GPU and dst is on pinned CPU, we MUST use
        # ``dst.copy_(src)`` (no ``non_blocking``) rather than the pattern
        # ``dst[...] = src.to(cpu, non_blocking=True)`` which creates an
        # intermediate non-pinned CPU tensor. That intermediate is filled by
        # cudaMemcpyAsync and the subsequent ``__setitem__`` on CPU does NOT
        # wait for the CUDA stream, so we end up copying uninitialized
        # memory (observed as ±8.51e+37 poisoning the replay buffer).
        # A blocking ``.copy_()`` GPU→pinned-CPU is already fast (PyTorch
        # uses cudaMemcpy, not cudaMemcpyAsync, when the destination is
        # pinned and non_blocking=False), and a single synchronize per
        # batch-add is negligible vs. the NaN blow-up it prevents.
        def _sync_copy(dst_view: torch.Tensor, src_t: torch.Tensor) -> None:
            if src_t.device == dst_view.device:
                dst_view.copy_(src_t)
            else:
                dst_view.copy_(src_t, non_blocking=False)

        for k in self._obs_shapes:
            _sync_copy(self._obs[k][dst], obs[k][src_slice])
            _sync_copy(self._next_obs[k][dst], next_obs[k][src_slice])
        _sync_copy(self._action[dst], batch_dict["action"][src_slice])
        _sync_copy(self._z[dst], batch_dict["z"][src_slice])
        term = batch_dict["next"]["terminated"][src_slice]
        if term.dtype != torch.bool:
            term = term.bool()
        if term.dim() == 1:
            term = term.unsqueeze(-1)
        _sync_copy(self._next_terminated[dst], term)
        aux = batch_dict.get("aux_rewards", {})
        for name in self.aux_reward_names:
            if name not in aux:
                continue
            v = aux[name][src_slice]
            if v.dim() == 1:
                v = v.unsqueeze(-1)
            _sync_copy(self._aux_rewards[name][dst], v)

    @torch.no_grad()
    def sample(self, batch_size: int) -> dict:
        n = len(self)
        if n == 0:
            raise RuntimeError("FBCprReplayBuffer.sample() called on empty buffer")
        idx = torch.randint(0, n, (batch_size,), device=self.device)
        return {
            "observation": _index_obs_dict(self._obs, idx),
            "action": self._action[idx],
            "z": self._z[idx],
            "next": {
                "observation": _index_obs_dict(self._next_obs, idx),
                "terminated": self._next_terminated[idx],
            },
            "aux_rewards": {name: self._aux_rewards[name][idx] for name in self.aux_reward_names},
        }

    def sample_chunks(self, batch_size: int, num_chunks: int, target_device: str | torch.device) -> list[dict]:
        """Sample ``num_chunks`` batches of size ``batch_size`` in ONE call.

        When the replay lives on CPU and training on GPU, this amortises the
        CPU→GPU transfer across all ``num_chunks`` updates — every leaf tensor
        is moved in a single async non_blocking copy, then sliced back into
        per-chunk views. Much faster than calling :meth:`sample` in a loop
        when ``num_chunks`` is large (e.g. BFM's 16 agent updates/iter).
        """
        n = len(self)
        if n == 0:
            raise RuntimeError("FBCprReplayBuffer.sample_chunks() called on empty buffer")
        total = int(batch_size) * int(num_chunks)
        idx = torch.randint(0, n, (total,), device=self.device)

        def _move(x: torch.Tensor) -> torch.Tensor:
            return x.to(target_device, non_blocking=True)

        # Gather + single async transfer per leaf tensor.
        obs_flat = {k: _move(self._obs[k][idx]) for k in self._obs_shapes}
        next_obs_flat = {k: _move(self._next_obs[k][idx]) for k in self._obs_shapes}
        action_flat = _move(self._action[idx])
        z_flat = _move(self._z[idx])
        term_flat = _move(self._next_terminated[idx])
        aux_flat = {name: _move(self._aux_rewards[name][idx]) for name in self.aux_reward_names}

        chunks: list[dict] = []
        for i in range(num_chunks):
            s = slice(i * batch_size, (i + 1) * batch_size)
            chunks.append({
                "observation": {k: obs_flat[k][s] for k in obs_flat},
                "action": action_flat[s],
                "z": z_flat[s],
                "next": {
                    "observation": {k: next_obs_flat[k][s] for k in next_obs_flat},
                    "terminated": term_flat[s],
                },
                "aux_rewards": {name: aux_flat[name][s] for name in self.aux_reward_names},
            })
        return chunks


# ---------------------------------------------------------------------------
# 2. FBCprExpertBuffer -- expert trajectory slicer loaded from .pt
# ---------------------------------------------------------------------------

class FBCprExpertBuffer:
    """Expert slicer that samples (s, s_next) windows of length ``seq_length``.

    Loads a ``.pt`` produced by ``scripts/precompute_bfm_expert_dataset.py``.
    Expected top-level keys: ``motions`` (dict name -> per-motion tensor dict),
    ``state_dim``, ``privileged_state_dim``, ``history_actor_dim``, etc.

    Each per-motion dict is expected to carry ``state`` [T, D_s],
    ``privileged_state`` [T, D_p], ``last_action`` [T, 29],
    ``history_actor`` [T, D_h].

    Sampling strategy (matches BFM ``TrajectoryDictBuffer.sample``): pick
    ``batch_size // seq_length`` trajectory start indices (optionally weighted
    by per-motion priorities), expand each into a length-``seq_length`` window,
    then flatten to a flat batch of shape ``(batch_size,)`` rows. The next-step
    observation is the same window shifted by +1 (so each motion needs at
    least ``seq_length + 1`` frames to be eligible).
    """

    def __init__(
        self,
        pt_path: str,
        seq_length: int,
        device: str | torch.device = "cpu",
        motion_ids: list[str] | None = None,
    ) -> None:
        self.seq_length = int(seq_length)
        self.device = torch.device(device)

        raw = torch.load(pt_path, weights_only=False, map_location="cpu")
        if not isinstance(raw, dict) or "motions" not in raw:
            raise ValueError(
                f"Expected a dict with 'motions' key from {pt_path}, got {type(raw)}"
            )

        all_motions = raw["motions"]
        if motion_ids is None:
            motion_ids = list(all_motions.keys())
        else:
            missing = [m for m in motion_ids if m not in all_motions]
            if missing:
                raise KeyError(f"Motions not in dataset: {missing}")

        self._motion_names: list[str] = list(motion_ids)
        self._states: list[torch.Tensor] = []
        self._privs: list[torch.Tensor] = []
        self._last_actions: list[torch.Tensor] = []
        self._history_actors: list[torch.Tensor] = []
        self._lengths: list[int] = []

        # RSI reset-state fields, populated below if present in the .pt.
        joint_pos_chunks: list[torch.Tensor] = []
        joint_vel_chunks: list[torch.Tensor] = []
        root_pos_chunks: list[torch.Tensor] = []
        root_quat_chunks: list[torch.Tensor] = []
        root_lin_vel_chunks: list[torch.Tensor] = []
        root_ang_vel_chunks: list[torch.Tensor] = []
        reset_fields_complete = True

        for name in self._motion_names:
            m = all_motions[name]
            self._states.append(m["state"].to(self.device).contiguous())
            self._privs.append(m["privileged_state"].to(self.device).contiguous())
            self._last_actions.append(m["last_action"].to(self.device).contiguous())
            self._history_actors.append(m["history_actor"].to(self.device).contiguous())
            self._lengths.append(int(m["state"].shape[0]))
            # RSI chunks — accumulate every motion or mark RSI unavailable.
            for key, bucket in (
                ("joint_pos", joint_pos_chunks),
                ("joint_vel", joint_vel_chunks),
                ("root_pos", root_pos_chunks),
                ("root_quat", root_quat_chunks),
                ("root_lin_vel", root_lin_vel_chunks),
                ("root_ang_vel", root_ang_vel_chunks),
            ):
                if key in m and isinstance(m[key], torch.Tensor):
                    bucket.append(m[key].to(self.device).contiguous())
                else:
                    reset_fields_complete = False

        self._lengths_t = torch.tensor(self._lengths, dtype=torch.long, device=self.device)

        # Per-motion RSI/tracking tensors, kept around so the tracking eval
        # can address individual motions (BFM's per-motion evaluation).
        self._per_motion_joint_pos: list[torch.Tensor] = list(joint_pos_chunks) if reset_fields_complete else []
        self._per_motion_joint_vel: list[torch.Tensor] = list(joint_vel_chunks) if reset_fields_complete else []
        self._per_motion_root_pos: list[torch.Tensor] = list(root_pos_chunks) if reset_fields_complete else []
        self._per_motion_root_quat: list[torch.Tensor] = list(root_quat_chunks) if reset_fields_complete else []
        self._per_motion_root_lin_vel: list[torch.Tensor] = list(root_lin_vel_chunks) if reset_fields_complete else []
        self._per_motion_root_ang_vel: list[torch.Tensor] = list(root_ang_vel_chunks) if reset_fields_complete else []

        # Infer dims from the first motion.
        self._state_dim = int(self._states[0].shape[-1])
        self._priv_dim = int(self._privs[0].shape[-1])
        self._action_dim = int(self._last_actions[0].shape[-1])
        self._history_actor_dim = int(self._history_actors[0].shape[-1])

        # RSI support: only enabled when every motion has all the needed fields.
        self.supports_reset_states = reset_fields_complete and len(joint_pos_chunks) == len(self._motion_names)
        if self.supports_reset_states:
            self.joint_pos_buffer = torch.cat(joint_pos_chunks, dim=0).contiguous()
            self.joint_vel_buffer = torch.cat(joint_vel_chunks, dim=0).contiguous()
            self.root_pos_buffer = torch.cat(root_pos_chunks, dim=0).contiguous()
            self.root_quat_buffer = torch.cat(root_quat_chunks, dim=0).contiguous()
            self.root_lin_vel_buffer = torch.cat(root_lin_vel_chunks, dim=0).contiguous()
            self.root_ang_vel_buffer = torch.cat(root_ang_vel_chunks, dim=0).contiguous()
            self.total_frames = int(self.joint_pos_buffer.shape[0])
            self.num_joints = int(self.joint_pos_buffer.shape[1])
            # Per-motion start offsets into the flat buffer (lazy RSI lookup).
            # ``_motion_starts[m]`` = first flat index, ``_motion_lengths_rsi[m]`` = frame count.
            _offs = [0]
            for chunk in joint_pos_chunks:
                _offs.append(_offs[-1] + int(chunk.shape[0]))
            self._motion_starts = torch.tensor(_offs[:-1], dtype=torch.long, device=self.device)
            self._motion_lengths_rsi = torch.tensor(
                [int(c.shape[0]) for c in joint_pos_chunks],
                dtype=torch.long, device=self.device,
            )
        else:
            self.joint_pos_buffer = None
            self.joint_vel_buffer = None
            self.root_pos_buffer = None
            self.root_quat_buffer = None
            self.root_lin_vel_buffer = None
            self.root_ang_vel_buffer = None
            self.total_frames = 0
            self.num_joints = 0

        # Uniform priority by default; updated via update_priorities().
        self._priorities = torch.ones(len(self._motion_names), dtype=torch.float32, device=self.device)
        self._priorities = self._priorities / self._priorities.sum()

    # -- properties --------------------------------------------------------

    @property
    def motion_ids(self) -> list[str]:
        return list(self._motion_names)

    @property
    def file_names(self) -> list[str]:
        return list(self._motion_names)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def priv_dim(self) -> int:
        return self._priv_dim

    @property
    def history_actor_dim(self) -> int:
        return self._history_actor_dim

    def __len__(self) -> int:
        return len(self._motion_names)

    def empty(self) -> bool:
        return len(self._motion_names) == 0

    @property
    def num_unique_motions(self) -> int:
        return len(self._motion_names)

    @property
    def lengths(self) -> list[int]:
        return list(self._lengths)

    @torch.no_grad()
    def get_motion_window(self, motion_id: int, num_frames: int) -> dict:
        """Return the first ``num_frames`` frames of motion ``motion_id`` as a dict.

        Truncates if the motion is shorter than ``num_frames`` (returns its
        actual length), so callers should read ``result['num_frames']``
        rather than assuming the full request.
        """
        assert 0 <= motion_id < self.num_unique_motions
        L = min(int(num_frames), self._lengths[motion_id])
        out = {
            "state": self._states[motion_id][:L],
            "privileged_state": self._privs[motion_id][:L],
            "last_action": self._last_actions[motion_id][:L],
            "history_actor": self._history_actors[motion_id][:L],
            "num_frames": L,
        }
        if self.supports_reset_states:
            out.update({
                "joint_pos": self._per_motion_joint_pos[motion_id][:L],
                "joint_vel": self._per_motion_joint_vel[motion_id][:L],
                "root_pos": self._per_motion_root_pos[motion_id][:L],
                "root_quat": self._per_motion_root_quat[motion_id][:L],
                "root_lin_vel": self._per_motion_root_lin_vel[motion_id][:L],
                "root_ang_vel": self._per_motion_root_ang_vel[motion_id][:L],
            })
        return out

    # -- RSI -------------------------------------------------------------- #

    @torch.no_grad()
    def sample_reset_states(self, batch_size: int) -> dict:
        """Sample expert frames for reference-state initialization (BFM-style RSI).

        Uniform over every frame in the flat buffer; returns world-frame root
        pose + velocities and joint pose + velocities so the env can write
        them into the simulator directly.
        """
        if not self.supports_reset_states:
            raise RuntimeError(
                "FBCprExpertBuffer does not have RSI fields. Re-run "
                "scripts/precompute_bfm_expert_dataset.py so that "
                "joint_pos / joint_vel / root_pos / root_quat / root_lin_vel "
                "/ root_ang_vel are populated for every motion."
            )
        # BFM's RSI samples in two stages: (a) motion_id ~ sampling_weights,
        # (b) start_time ~ Uniform[0, motion_length). Flat-uniform frame
        # sampling (old code) over-represents long motions and bypasses
        # priority feedback entirely. This restores BFM's semantics.
        motion_ids = torch.multinomial(
            self._priorities, num_samples=batch_size, replacement=True,
        )
        starts = self._motion_starts[motion_ids]
        lens = self._motion_lengths_rsi[motion_ids]
        # Uniform frame within each motion.
        rand = torch.rand(batch_size, device=self.device)
        offsets = (rand * lens.to(torch.float32)).floor().to(torch.long).clamp(max=lens - 1)
        frame = starts + offsets
        return {
            "joint_pos": self.joint_pos_buffer[frame],
            "joint_vel": self.joint_vel_buffer[frame],
            "root_pos": self.root_pos_buffer[frame],
            "root_quat": self.root_quat_buffer[frame],
            "root_lin_vel": self.root_lin_vel_buffer[frame],
            "root_ang_vel": self.root_ang_vel_buffer[frame],
        }

    # -- priority updates (stub-ish; accepted by agent) --------------------

    def update_priorities(self, priorities: torch.Tensor, idxs: torch.Tensor | None = None) -> None:
        """Update per-motion sampling weights.

        If ``idxs`` is None, expects ``priorities`` to have length equal to
        the number of motions. Otherwise scatters the new values into the
        given indices. Non-negative values are required; values are then
        renormalised to sum to 1.
        """
        priorities = priorities.to(self.device).float().clamp_min(0.0)
        if idxs is None:
            if priorities.numel() != len(self._motion_names):
                raise ValueError(
                    f"Expected priorities of length {len(self._motion_names)}, got {priorities.numel()}"
                )
            self._priorities = priorities
        else:
            idxs = idxs.to(self.device).long()
            self._priorities[idxs] = priorities
        s = self._priorities.sum()
        if s > 0:
            self._priorities = self._priorities / s
        else:
            self._priorities = torch.ones_like(self._priorities) / self._priorities.numel()

    # -- sampling ----------------------------------------------------------

    @torch.no_grad()
    def sample(self, batch_size: int, seq_length: int | None = None) -> dict:
        seq_length = int(seq_length) if seq_length is not None else self.seq_length
        if batch_size < seq_length or batch_size % seq_length != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a positive multiple of seq_length ({seq_length})"
            )
        num_slices = batch_size // seq_length

        # Eligibility: need at least seq_length + 1 frames per motion.
        eligible_mask = self._lengths_t >= (seq_length + 1)
        if not bool(eligible_mask.any().item()):
            raise RuntimeError(
                f"No motion has at least {seq_length + 1} frames; lower seq_length."
            )
        eligible_idx = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
        eligible_priors = self._priorities[eligible_idx]
        eligible_priors = eligible_priors / eligible_priors.sum().clamp_min(1e-12)
        eligible_lengths = self._lengths_t[eligible_idx]

        # Pick a motion per slice (weighted), then a frame start within it.
        sel = torch.multinomial(eligible_priors, num_slices, replacement=True)
        motion_picks = eligible_idx[sel]                    # [num_slices]
        motion_lens = eligible_lengths[sel]                 # [num_slices]
        # Legal starts: t in [0, T - seq_length - 1] inclusive.
        rand01 = torch.rand(num_slices, device=self.device)
        max_start = (motion_lens - seq_length - 1).clamp_min(0).to(torch.float32)
        starts = (rand01 * (max_start + 1.0)).floor().to(torch.long)

        # Build per-slice arange window and flatten.
        arange = torch.arange(seq_length, device=self.device).unsqueeze(0)     # [1, seq_length]
        # frame_idx_cur[t, s] and frame_idx_nxt[t, s] -> flattened [B].
        frame_cur = (starts.unsqueeze(1) + arange).reshape(-1)                  # [B]
        frame_nxt = frame_cur + 1
        motion_flat = motion_picks.unsqueeze(1).expand(-1, seq_length).reshape(-1)  # [B]

        state, priv, last_action, history = [], [], [], []
        state_nxt, priv_nxt, last_action_nxt, history_nxt = [], [], [], []

        # Batched gather per unique motion keeps this cheap for large datasets.
        # For typical num_slices ~ thousands and ~hundreds of motions this is
        # still fast enough; we just iterate over unique picks.
        unique_motions, inverse = torch.unique(motion_flat, return_inverse=True)
        # Zero-init (not empty_like) to guard against any gather miss leaving
        # uninitialized memory (1e37 NaN-bait) in the output. Cheap at
        # batch_size 1024 × 4 tensors.
        out_state = torch.zeros((motion_flat.shape[0], self._state_dim), device=self.device)
        out_priv = torch.zeros((motion_flat.shape[0], self._priv_dim), device=self.device)
        out_act = torch.zeros((motion_flat.shape[0], self._action_dim), device=self.device)
        out_hist = torch.zeros((motion_flat.shape[0], self._history_actor_dim), device=self.device)
        out_state_n = torch.zeros_like(out_state)
        out_priv_n = torch.zeros_like(out_priv)
        out_act_n = torch.zeros_like(out_act)
        out_hist_n = torch.zeros_like(out_hist)

        for u in unique_motions.tolist():
            mask = motion_flat == u
            fc = frame_cur[mask]
            fn = frame_nxt[mask]
            out_state[mask] = self._states[u][fc]
            out_priv[mask] = self._privs[u][fc]
            out_act[mask] = self._last_actions[u][fc]
            out_hist[mask] = self._history_actors[u][fc]
            out_state_n[mask] = self._states[u][fn]
            out_priv_n[mask] = self._privs[u][fn]
            out_act_n[mask] = self._last_actions[u][fn]
            out_hist_n[mask] = self._history_actors[u][fn]

        B = out_state.shape[0]
        terminated = torch.zeros((B, 1), dtype=torch.bool, device=self.device)
        z_dummy = torch.zeros((B, 0), device=self.device)   # filled by caller if needed
        # The agent signature expects a z entry; we provide zeros of a
        # caller-determined dim. We cannot know z_dim here, so we leave it
        # zero-width and let the caller broadcast / fill. The BFM reference
        # implementation uses the agent's own z-sampler to overwrite this
        # before the discriminator step.

        return {
            "observation": {
                "state": out_state,
                "privileged_state": out_priv,
                "last_action": out_act,
                "history_actor": out_hist,
            },
            "action": out_act,   # reconstructed "last_action" doubles as the demo action
            "z": z_dummy,
            "next": {
                "observation": {
                    "state": out_state_n,
                    "privileged_state": out_priv_n,
                    "last_action": out_act_n,
                    "history_actor": out_hist_n,
                },
                "terminated": terminated,
            },
        }

    @torch.no_grad()
    def sample_chunks(self, batch_size: int, num_chunks: int,
                      target_device: str | torch.device,
                      seq_length: int | None = None) -> list[dict]:
        """Sample ``num_chunks`` batches of size ``batch_size`` in ONE call.

        Each chunk preserves the ``[N x seq_length]`` ordering ``sample()``
        produces (the agent's ``encode_expert`` relies on it), because we
        concatenate batches along the row axis and slice contiguously.
        """
        seq_length = int(seq_length) if seq_length is not None else self.seq_length
        if batch_size % seq_length != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a positive multiple of seq_length ({seq_length})"
            )

        def _move(x: torch.Tensor) -> torch.Tensor:
            return x.to(target_device, non_blocking=True) if x.device != torch.device(target_device) else x

        # Build N*batch_size rows by calling sample(batch_size) num_chunks times
        # and concatenating. Each call preserves the N_i x seq_length layout,
        # so concatenating along dim 0 keeps chunk-i's rows contiguous.
        big_batches = [self.sample(batch_size, seq_length=seq_length) for _ in range(num_chunks)]

        def _stack_obs(key: str, sub: str | None = None) -> torch.Tensor:
            if sub is None:
                vals = [b["observation"][key] for b in big_batches]
            else:
                vals = [b[sub]["observation"][key] for b in big_batches]
            return _move(torch.cat(vals, dim=0))

        # Stack every leaf tensor and issue a single async transfer per leaf.
        obs_keys = ["state", "privileged_state", "last_action", "history_actor"]
        obs_flat = {k: _stack_obs(k) for k in obs_keys}
        next_obs_flat = {k: _stack_obs(k, sub="next") for k in obs_keys}
        action_flat = _move(torch.cat([b["action"] for b in big_batches], dim=0))
        z_flat = _move(torch.cat([b["z"] for b in big_batches], dim=0))
        term_flat = _move(torch.cat([b["next"]["terminated"] for b in big_batches], dim=0))

        chunks: list[dict] = []
        for i in range(num_chunks):
            s = slice(i * batch_size, (i + 1) * batch_size)
            chunks.append({
                "observation": {k: obs_flat[k][s] for k in obs_keys},
                "action": action_flat[s],
                "z": z_flat[s],
                "next": {
                    "observation": {k: next_obs_flat[k][s] for k in obs_keys},
                    "terminated": term_flat[s],
                },
            })
        return chunks


# ---------------------------------------------------------------------------
# 3. ZBuffer -- rolling reservoir of recent z vectors
# ---------------------------------------------------------------------------

class ZBuffer:
    """Circular buffer of recently-seen z vectors for mix-rollout sampling."""

    def __init__(self, capacity: int, z_dim: int, device: str | torch.device = "cpu") -> None:
        self.capacity = int(capacity)
        self.z_dim = int(z_dim)
        self.device = torch.device(device)
        self._storage = torch.zeros((self.capacity, self.z_dim), device=self.device)
        self._idx = 0
        self._is_full = False

    def __len__(self) -> int:
        return self.capacity if self._is_full else self._idx

    def empty(self) -> bool:
        return self._idx == 0 and not self._is_full

    @torch.no_grad()
    def add(self, z_batch: torch.Tensor) -> None:
        z_batch = z_batch.to(self.device, non_blocking=True)
        n = z_batch.shape[0]
        if n == 0:
            return
        if n >= self.capacity:
            # Keep the most recent ``capacity`` samples.
            self._storage.copy_(z_batch[-self.capacity:])
            self._is_full = True
            self._idx = 0
            return
        end = self._idx + n
        if end <= self.capacity:
            self._storage[self._idx:end] = z_batch
            self._idx = end
            if self._idx == self.capacity:
                self._is_full = True
                self._idx = 0
        else:
            first = self.capacity - self._idx
            self._storage[self._idx:] = z_batch[:first]
            self._storage[: n - first] = z_batch[first:]
            self._is_full = True
            self._idx = n - first

    @torch.no_grad()
    def sample(self, batch_size: int, device: str | torch.device | None = None) -> torch.Tensor:
        n = len(self)
        if n == 0:
            raise RuntimeError("ZBuffer.sample() called on empty buffer")
        idx = np.random.randint(0, n, size=batch_size)
        idx_t = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        out = self._storage[idx_t].clone()
        if device is not None:
            out = out.to(torch.device(device))
        return out
