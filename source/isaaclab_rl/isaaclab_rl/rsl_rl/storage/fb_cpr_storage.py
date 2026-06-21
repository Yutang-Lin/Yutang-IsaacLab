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

import math
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
    """Trajectory-aware 2D circular replay buffer matching BFM-Zero's
    ``TrajectoryDictBufferMultiDim``.

    Storage layout: ``[time_steps, num_envs, ...]`` — each ``extend()`` call
    appends one time-slice ``[1, num_envs, ...]`` across all parallel envs.
    Episode boundaries are marked by the ``truncated`` column; ``sample()``
    draws contiguous sub-sequences of length ``seq_length`` that never cross
    episode boundaries, exactly as BFM-Zero does.

    ``capacity`` is the total number of transitions (= ``time_steps * num_envs``).
    The time-axis length is ``capacity // num_envs``.
    """

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        obs_space: Any,
        action_dim: int,
        z_dim: int,
        aux_reward_names: list[str],
        device: str | torch.device = "cpu",
        pin_memory: bool | None = None,
        extra_field_shapes: dict[str, tuple[int, ...]] | None = None,
        actor_window_len: int = 0,
    ) -> None:
        # >0 -> sample() also returns an ``actor_window`` (the transformer actor's
        # per-timestep H+1 frame window + per-position obs + a valid mask), gathered
        # in-place from the 2D [time,env] storage. 0 -> off (MLP actor; no window).
        self.actor_window_len = int(actor_window_len)
        self.num_envs = int(num_envs)
        self.time_capacity = int(capacity) // self.num_envs
        self.capacity = self.time_capacity  # __len__ counts time-steps
        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        self.z_dim = int(z_dim)
        # BFM's train replay uses seq_length=1 (individual transitions).
        # seq_length=8 is only for the expert slicer. The train buffer's
        # trajectory awareness is for episode-boundary safety, not for
        # returning contiguous multi-frame windows.
        self.seq_length = 1
        self.aux_reward_names = list(aux_reward_names)

        if pin_memory is None:
            pin_memory = self.device.type == "cpu"
        self._pin_memory = bool(pin_memory)

        if hasattr(obs_space, "spaces"):
            self._obs_shapes = {k: _space_shape(v) for k, v in obs_space.spaces.items()}
        else:
            self._obs_shapes = {k: tuple(v) for k, v in dict(obs_space).items()}

        # 2D storage: [time_capacity, num_envs, ...]
        self._obs: dict[str, torch.Tensor] = {
            k: self._alloc((self.time_capacity, self.num_envs, *shape))
            for k, shape in self._obs_shapes.items()
        }
        self._action = self._alloc((self.time_capacity, self.num_envs, self.action_dim))
        self._z = self._alloc((self.time_capacity, self.num_envs, self.z_dim))
        self._terminated = self._alloc((self.time_capacity, self.num_envs, 1), dtype=torch.bool)
        self._truncated = self._alloc((self.time_capacity, self.num_envs, 1), dtype=torch.bool)
        self._aux_rewards: dict[str, torch.Tensor] = {
            name: self._alloc((self.time_capacity, self.num_envs, 1)) for name in self.aux_reward_names
        }
        # Optional extra per-timestep fields (e.g. world SE(2) pose for the
        # anchored variant). Gathered at BOTH current and next indices on
        # sample (like obs). Empty by default — existing tasks unaffected.
        self._extra_field_shapes: dict[str, tuple[int, ...]] = dict(extra_field_shapes or {})
        self._extras: dict[str, torch.Tensor] = {
            k: self._alloc((self.time_capacity, self.num_envs, *shape))
            for k, shape in self._extra_field_shapes.items()
        }

        self._idx = 0
        self._is_full = False
        self._recompute_traj_info = True
        # Cached trajectory start/stop/length info (recomputed lazily).
        self._start_idx: torch.Tensor | None = None
        self._lengths: torch.Tensor | None = None

    def _alloc(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self._pin_memory and self.device.type == "cpu":
            t = torch.empty(shape, dtype=dtype, pin_memory=True)
        else:
            t = torch.empty(shape, dtype=dtype, device=self.device)
        t.zero_()
        return t

    def __len__(self) -> int:
        return self.time_capacity if self._is_full else self._idx

    @property
    def full(self) -> bool:
        return self._is_full

    @property
    def total_transitions(self) -> int:
        return len(self) * self.num_envs

    # -- serialization ----------------------------------------------------

    @torch.no_grad()
    def state_dict(self) -> dict:
        return {
            "_obs": {k: v.clone() for k, v in self._obs.items()},
            "_action": self._action.clone(),
            "_z": self._z.clone(),
            "_terminated": self._terminated.clone(),
            "_truncated": self._truncated.clone(),
            "_aux_rewards": {k: v.clone() for k, v in self._aux_rewards.items()},
            "_extras": {k: v.clone() for k, v in self._extras.items()},
            "_idx": int(self._idx),
            "_is_full": bool(self._is_full),
            "time_capacity": self.time_capacity,
            "num_envs": self.num_envs,
            "action_dim": self.action_dim,
            "z_dim": self.z_dim,
            "aux_reward_names": list(self.aux_reward_names),
        }

    @torch.no_grad()
    def load_state_dict(self, sd: dict) -> None:
        if int(sd.get("time_capacity", sd.get("capacity", -1))) != self.time_capacity:
            raise ValueError("replay time_capacity mismatch")
        if int(sd.get("num_envs", -1)) != self.num_envs:
            raise ValueError("replay num_envs mismatch")
        for k in self._obs_shapes:
            if k in sd["_obs"]:
                self._obs[k].copy_(sd["_obs"][k].to(self._obs[k].device))
        self._action.copy_(sd["_action"].to(self._action.device))
        self._z.copy_(sd["_z"].to(self._z.device))
        self._terminated.copy_(sd["_terminated"].to(self._terminated.device))
        self._truncated.copy_(sd["_truncated"].to(self._truncated.device))
        for name in self.aux_reward_names:
            if name in sd["_aux_rewards"]:
                self._aux_rewards[name].copy_(
                    sd["_aux_rewards"][name].to(self._aux_rewards[name].device),
                )
        for name in self._extra_field_shapes:
            if name in sd.get("_extras", {}):
                self._extras[name].copy_(sd["_extras"][name].to(self._extras[name].device))
        self._idx = int(sd["_idx"])
        self._is_full = bool(sd["_is_full"])
        self._recompute_traj_info = True

    # -- extend (one time-step across all envs) ----------------------------

    @torch.no_grad()
    def extend(self, batch_dict: dict) -> None:
        """Append one time-step slice ``[1, num_envs, ...]``.

        ``batch_dict`` keys: ``observation`` (dict), ``action`` [num_envs, A],
        ``z`` [num_envs, Z], ``terminated`` [num_envs, 1], ``truncated``
        [num_envs, 1], and optionally ``aux_rewards`` sub-dict.
        """
        obs = batch_dict["observation"]
        t = self._idx

        def _copy(dst: torch.Tensor, src: torch.Tensor) -> None:
            if src.device == dst.device:
                dst.copy_(src)
            else:
                dst.copy_(src, non_blocking=False)

        for k in self._obs_shapes:
            _copy(self._obs[k][t], obs[k])
        _copy(self._action[t], batch_dict["action"])
        _copy(self._z[t], batch_dict["z"])
        term = batch_dict["terminated"]
        if term.dtype != torch.bool:
            term = term.bool()
        if term.dim() == 1:
            term = term.unsqueeze(-1)
        _copy(self._terminated[t], term)
        trunc = batch_dict["truncated"]
        if trunc.dtype != torch.bool:
            trunc = trunc.bool()
        if trunc.dim() == 1:
            trunc = trunc.unsqueeze(-1)
        _copy(self._truncated[t], trunc)
        aux = batch_dict.get("aux_rewards", {})
        for name in self.aux_reward_names:
            if name not in aux:
                continue
            v = aux[name]
            if v.dim() == 1:
                v = v.unsqueeze(-1)
            _copy(self._aux_rewards[name][t], v)
        extras = batch_dict.get("extras", {})
        for name in self._extra_field_shapes:
            if name not in extras:
                continue
            v = extras[name]
            if v.dim() == 1 and len(self._extra_field_shapes[name]) == 1:
                v = v.unsqueeze(-1)
            _copy(self._extras[name][t], v)

        self._idx = t + 1
        if self._idx >= self.time_capacity:
            self._is_full = True
            self._idx = 0
        self._recompute_traj_info = True

    # -- trajectory segmentation (BFM's find_start_stop_traj) ---------------

    def _ensure_traj_info(self) -> None:
        if not self._recompute_traj_info:
            return
        done = self._truncated[:len(self)].squeeze(-1)  # [T, E] bool
        T = done.shape[0]
        E = done.shape[1]
        starts_list: list[torch.Tensor] = []
        lengths_list: list[int] = []
        if self._is_full:
            cursor = (self._idx - 1) % self.time_capacity
            done_copy = done.clone()
            done_copy[cursor] = True
        else:
            done_copy = done.clone()
            done_copy[T - 1] = True
        for e in range(E):
            col = done_copy[:, e]
            ends = col.nonzero(as_tuple=False).squeeze(-1)
            if ends.numel() == 0:
                starts_list.append(torch.tensor([0, e], device=self.device))
                lengths_list.append(T)
                continue
            prev_end = -1
            for end_t in ends.tolist():
                start_t = (prev_end + 1) % T if prev_end >= 0 else 0
                if self._is_full and prev_end == -1:
                    start_t = (ends[-1].item() + 1) % T
                length = end_t - start_t + 1
                if length <= 0:
                    length += T
                starts_list.append(torch.tensor([start_t, e], device=self.device))
                lengths_list.append(length)
                prev_end = end_t
        self._start_idx = torch.stack(starts_list)  # [N_traj, 2]
        self._lengths = torch.tensor(lengths_list, device=self.device, dtype=torch.long)
        self._recompute_traj_info = False

    # -- sampling (BFM's get_idxs + _tensor_slices_from_startend) -----------

    @torch.no_grad()
    def sample(self, batch_size: int, seq_length: int | None = None) -> dict:
        seq_length = seq_length or self.seq_length
        if len(self) == 0:
            raise RuntimeError("FBCprReplayBuffer.sample() called on empty buffer")
        # Round batch down to a multiple of seq_length.
        self._ensure_traj_info()
        num_slices = max(1, batch_size // seq_length)
        # Episode "length" includes the truncated row (post-reset obs).
        # The last valid transition's next_obs is at position length-2 (0-indexed).
        # A window of seq_length obs frames needs next_obs at start+seq_length,
        # so we need start + seq_length <= length - 2, i.e. length >= seq_length + 2.
        min_len = seq_length + 2
        eligible = self._lengths >= min_len
        if not bool(eligible.any().item()):
            raise RuntimeError(
                f"No trajectories with length >= {min_len}; buffer too small or all episodes shorter."
            )
        eligible_idx = eligible.nonzero(as_tuple=False).squeeze(-1)
        eligible_lengths = self._lengths[eligible_idx]
        eligible_starts = self._start_idx[eligible_idx]

        traj_sel = torch.randint(eligible_idx.shape[0], (num_slices,), device=self.device)
        sel_lengths = eligible_lengths[traj_sel]
        sel_starts = eligible_starts[traj_sel]  # [num_slices, 2]

        # max start = length - seq_length - 2 so that last next_obs
        # at start + seq_length stays before the truncated row.
        end_point = (sel_lengths - seq_length - 2).clamp_min(0).to(torch.float32)
        relative_starts = (torch.rand(num_slices, device=self.device) * (end_point + 1.0)).floor().to(torch.long)

        time_starts = (sel_starts[:, 0] + relative_starts)  # [num_slices]
        env_ids = sel_starts[:, 1]  # [num_slices]

        arange = torch.arange(seq_length, device=self.device)  # [seq_length]
        # time indices: [num_slices, seq_length] -> flatten to [batch_size]
        time_idx = (time_starts.unsqueeze(1) + arange.unsqueeze(0)) % self.time_capacity
        time_idx = time_idx.reshape(-1)
        env_idx = env_ids.unsqueeze(1).expand(-1, seq_length).reshape(-1)
        # next-step indices (t+1)
        time_idx_next = (time_idx + 1) % self.time_capacity

        return self._gather(time_idx, env_idx, time_idx_next)

    def sample_flat(self, batch_size: int) -> dict:
        """Sample exactly ``batch_size`` i.i.d. transitions (no seq_length
        chunking). Returned tensors have shape ``[batch_size, ...]``. Used
        by the main training path (FB/actor/critic) which doesn't need
        the temporal sub-sequence structure that ``sample()`` produces.
        """
        if len(self) == 0:
            raise RuntimeError("FBCprReplayBuffer.sample_flat() called on empty buffer")
        self._ensure_traj_info()
        # Need length>=3: a valid NON-boundary pair (obs, next_obs) requires the
        # next-obs to be a pre-reset row (<= start+length-2), so rel in [0,L-3].
        eligible = self._lengths >= 3
        if not bool(eligible.any().item()):
            raise RuntimeError("No trajectories with length >= 3.")
        eligible_idx = eligible.nonzero(as_tuple=False).squeeze(-1)
        eligible_lengths = self._lengths[eligible_idx]
        eligible_starts = self._start_idx[eligible_idx]
        traj_sel = torch.randint(eligible_idx.shape[0], (batch_size,), device=self.device)
        sel_lengths = eligible_lengths[traj_sel]
        sel_starts = eligible_starts[traj_sel]
        # Pick a random transition index ``rel`` in [0, length-3] so that the
        # next-obs ``start+rel+1`` is at most ``start+length-2`` — the LAST
        # PRE-RESET obs. The final row of a segment (``start+length-1``) is the
        # truncated row = the POST-RESET spawn; sampling rel=length-2 would form
        # the cross-boundary transition (last-pre-reset -> fresh-spawn), a ~tens-
        # of-metres teleport in the (anchored) global pose. That timeout boundary
        # is ``truncated`` not ``terminated``, so the FB/critic discount (masks
        # ``terminated`` only) would NOT zero it -> B/F get trained on a garbage
        # one-step "reach 50 m" successor target. Excluding it here is the clean
        # fix (mirrors sample()'s seq-window bound). Needs length>=3 for any pair.
        end_point = (sel_lengths - 3).clamp_min(0).to(torch.float32)
        rel = (torch.rand(batch_size, device=self.device) * (end_point + 1.0)).floor().to(torch.long)
        time_idx = (sel_starts[:, 0] + rel) % self.time_capacity
        env_idx = sel_starts[:, 1]
        time_idx_next = (time_idx + 1) % self.time_capacity
        return self._gather(time_idx, env_idx, time_idx_next)

    def _gather(self, time_idx: torch.Tensor, env_idx: torch.Tensor,
                time_idx_next: torch.Tensor) -> dict:
        obs = {k: v[time_idx, env_idx] for k, v in self._obs.items()}
        next_obs = {k: v[time_idx_next, env_idx] for k, v in self._obs.items()}
        out = {
            "observation": obs,
            "action": self._action[time_idx, env_idx],
            "z": self._z[time_idx, env_idx],
            "next": {
                "observation": next_obs,
                "terminated": self._terminated[time_idx_next, env_idx],
            },
            "aux_rewards": {
                name: self._aux_rewards[name][time_idx, env_idx]
                for name in self.aux_reward_names
            },
        }
        if self._extras:
            out["extras"] = {
                k: v[time_idx, env_idx] for k, v in self._extras.items()
            }
            out["next"]["extras"] = {
                k: v[time_idx_next, env_idx] for k, v in self._extras.items()
            }
        if self.actor_window_len > 0:
            out["actor_window"] = self._gather_actor_window(time_idx, env_idx)
        return out

    @torch.no_grad()
    def _gather_actor_window(self, time_idx: torch.Tensor, env_idx: torch.Tensor) -> dict:
        """Gather the transformer actor's H+1 timestep window ending at the
        sampled ``time_idx`` (current step). Returns per-position obs windows and
        a ``valid`` mask. No extra storage — gathered from the 2D [time,env] buffer.

        Layout: offsets ``[-H, ..., 0]`` (oldest -> current). A past offset is
        INVALID (mask False) when an episode boundary (``_truncated``) lies
        strictly between it and the current step — i.e. that past frame belongs
        to a different episode and must be excluded from the parallel loss.
        ``valid[:, H]`` (current) is always True.
        """
        H = self.actor_window_len
        B = time_idx.shape[0]
        Tcap = self.time_capacity
        # offsets oldest..current: [-H, ..., 0]  -> shape [H+1]
        offs = torch.arange(-H, 1, device=self.device)
        # time index per (sample, pos): wrap circularly. [B, H+1]
        tw = (time_idx.unsqueeze(1) + offs.unsqueeze(0)) % Tcap
        ew = env_idx.unsqueeze(1).expand(-1, H + 1)
        # Per-position obs windows (state/priv/last_action/history_actor and any
        # other stored obs key), z, and the per-position truncated flag.
        obs_w = {k: v[tw, ew] for k, v in self._obs.items()}     # each [B, H+1, dim]
        z_w = self._z[tw, ew]                                    # [B, H+1, z_dim]
        trunc_w = self._truncated[tw, ew].squeeze(-1)            # [B, H+1] bool
        # Validity: a boundary at position p (truncated row = post-reset spawn of
        # a NEW episode) invalidates that position AND everything OLDER than it.
        # Walk from current (pos H) backwards: once we hit a truncated row, all
        # earlier positions are out-of-episode. truncated marks the FIRST frame of
        # a new episode, so positions <= that index are invalid.
        # boundary[:, p] = True if frame at pos p is a fresh-episode (truncated) row.
        valid = torch.ones(B, H + 1, dtype=torch.bool, device=self.device)
        # cumulative-OR of truncation scanning from oldest..current, but a
        # truncation at pos p means frames at pos < p (older) belong to a prior
        # episode; pos p itself is the new episode's first frame (valid, same
        # episode as current as long as no LATER truncation). So: invalid[p] =
        # any truncated at positions p+1..H (a later reset started a new episode
        # after p, cutting p off from current). Reverse-cumsum over p+1..H.
        if H > 0:
            later = trunc_w[:, 1:]                                # [B, H] (pos 1..H)
            # flip, cumsum-or, flip back -> "any truncation strictly after pos p"
            after_any = torch.flip(torch.cummax(torch.flip(later, dims=[1]), dim=1).values, dims=[1])
            # after_any[:, p] corresponds to "any trunc in pos (p+1)..H" for p in 0..H-1
            valid[:, :H] = ~after_any.bool()
        valid[:, H] = True
        return {
            "obs": obs_w,            # dict key -> [B, H+1, dim]
            "z": z_w,                # [B, H+1, z_dim]
            "valid": valid,          # [B, H+1] bool
        }

    def sample_chunks(self, batch_size: int, num_chunks: int, target_device: str | torch.device) -> list[dict]:
        """Sample ``num_chunks`` batches in ONE call, then transfer to ``target_device``.

        Train batch uses flat i.i.d. sampling (no seq_length chunking) so
        ``batch_size`` is honored exactly. The seq_length structure is
        only needed by the expert/disc path, not FB/actor/critic.
        """
        total = int(batch_size) * int(num_chunks)
        big = self.sample_flat(total)

        def _move(x: torch.Tensor) -> torch.Tensor:
            return x.to(target_device, non_blocking=True)

        obs_flat = {k: _move(v) for k, v in big["observation"].items()}
        next_obs_flat = {k: _move(v) for k, v in big["next"]["observation"].items()}
        action_flat = _move(big["action"])
        z_flat = _move(big["z"])
        term_flat = _move(big["next"]["terminated"])
        aux_flat = {name: _move(big["aux_rewards"][name]) for name in self.aux_reward_names}
        has_extras = "extras" in big
        if has_extras:
            extras_flat = {k: _move(v) for k, v in big["extras"].items()}
            next_extras_flat = {k: _move(v) for k, v in big["next"]["extras"].items()}
        has_window = "actor_window" in big
        if has_window:
            aw = big["actor_window"]
            aw_obs_flat = {k: _move(v) for k, v in aw["obs"].items()}
            aw_z_flat = _move(aw["z"])
            aw_valid_flat = _move(aw["valid"])

        chunks: list[dict] = []
        for i in range(num_chunks):
            s = slice(i * batch_size, (i + 1) * batch_size)
            chunk = {
                "observation": {k: obs_flat[k][s] for k in obs_flat},
                "action": action_flat[s],
                "z": z_flat[s],
                "next": {
                    "observation": {k: next_obs_flat[k][s] for k in next_obs_flat},
                    "terminated": term_flat[s],
                },
                "aux_rewards": {name: aux_flat[name][s] for name in self.aux_reward_names},
            }
            if has_extras:
                chunk["extras"] = {k: extras_flat[k][s] for k in extras_flat}
                chunk["next"]["extras"] = {k: next_extras_flat[k][s] for k in next_extras_flat}
            if has_window:
                chunk["actor_window"] = {
                    "obs": {k: aw_obs_flat[k][s] for k in aw_obs_flat},
                    "z": aw_z_flat[s],
                    "valid": aw_valid_flat[s],
                }
            chunks.append(chunk)
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
        length_proportional_priors: bool = True,
        distributed_shard: bool = False,
        shard_rank: int = 0,
        shard_world_size: int = 1,
        shard_seed: int = 0,
        keypoint_names: list[str] | None = None,
        emit_anchored_pose: bool = False,
        anchored_pose_clamp: float = 10.0,
        anchor_alpha_gt: float = 0.34,
        anchor_random_xy_range: float = 10.0,
        anchor_frame_body: bool = False,
        priv_include_heading_body: bool = False,
        history_len_override: int | None = None,
    ) -> None:
        """Expert motion buffer.

        ``history_len_override``: if set, compose the minimal-dataset
        ``history_actor`` at this many frames instead of the dataset's baked
        ``history_length``. Needed when the ENV history was deepened (e.g.
        BFM-0.5: H 4 -> 9) so the expert ``history_actor`` dim matches the env's
        (the shared obs-normalizer BatchNorm is sized from the env obs space).

        Args:
            length_proportional_priors: If True (default), initial
                ``_priorities`` are set proportional to each motion's
                frame count, so the expected per-transition draw
                probability is uniform across the dataset. With
                unbalanced clip lengths (e.g. a 7468-frame continuous
                motion alongside 8000-frame LAFAN clips) this matters a
                lot: under uniform motion-priors the transitions from a
                short clip are sampled ~clip-length-ratio more often per
                update. ``update_priorities()`` (from the tracking eval)
                overwrites this init.
        """
        self.seq_length = int(seq_length)
        self._length_proportional_priors = bool(length_proportional_priors)
        self.device = torch.device(device)
        # Anchored variant: emit an ``anchored_pose`` obs (A^-1 g). The anchor
        # is sampled from the SAME p_A as the policy (alpha at the frame's own
        # current pose, else random around it) so expert and policy z_spatial
        # share a distribution — otherwise the spatial discriminator shortcuts
        # on the z-region instead of judging the motion.
        self._emit_anchored_pose = bool(emit_anchored_pose)
        self._anchored_pose_clamp = float(anchored_pose_clamp)
        self._anchor_alpha_gt = float(anchor_alpha_gt)
        self._anchor_random_xy_range = float(anchor_random_xy_range)
        # Anchor-frame body pose: reframe the expert priv body-pose POS+ROT6D
        # from heading frame -> sub-trajectory-start anchor frame at sample time
        # (matches the env's anchor_frame_body=True and the algo per-row reframe).
        self._anchor_frame_body = bool(anchor_frame_body)
        # Append a heading-frame body (pos+rot6d) tail to composed priv so the
        # expert priv layout matches the env's _obs_max_local_self(
        # include_heading_body=True). MUST agree with the env or B sees a
        # layout mismatch between expert and rollout priv.
        self._priv_include_heading_body = bool(priv_include_heading_body)

        raw = torch.load(pt_path, weights_only=False, map_location="cpu")
        if not isinstance(raw, dict) or "motions" not in raw:
            raise ValueError(
                f"Expected a dict with 'motions' key from {pt_path}, got {type(raw)}"
            )

        # ------------------------------------------------------------- #
        # Minimal-format dataset: motion dicts only carry raw fields
        # (root_pos, root_quat, joint_pos, fps, ...). We derive the full
        # obs/RSI set at load time by calling the precompute pipeline's
        # ``_process_motion`` on each motion (per this rank's shard when
        # distributed). Avoids shipping huge precomputed buffers on disk.
        # ------------------------------------------------------------- #
        self._minimal: bool = bool(raw.get("minimal", False))
        self._minimal_derive_fn = None
        if self._minimal:
            # Lazy-import the precompute pipeline. It lives in the
            # Latent-Control repo under scripts/; the runner adds it to
            # sys.path via the cfg's ``expert_dataset_compose_script``
            # (with a sane default relative to this file's location).
            import importlib.util, os, sys
            script_rel = os.environ.get(
                "BFM_EXPERT_COMPOSE_SCRIPT",
                # Default: Latent-Control/scripts/precompute_bfm_expert_dataset.py
                # one sibling up from Yutang-IsaacLab.
                os.path.abspath(os.path.join(
                    os.path.dirname(__file__),
                    "..", "..", "..", "..", "..", "..",
                    "Latent-Control", "scripts",
                    "precompute_bfm_expert_dataset.py",
                )),
            )
            if not os.path.exists(script_rel):
                raise FileNotFoundError(
                    f"Minimal expert dataset {pt_path} requires the precompute "
                    f"script at {script_rel} (not found). Set BFM_EXPERT_COMPOSE_SCRIPT "
                    f"env var to its absolute path, or rebuild the dataset "
                    f"without --minimal."
                )
            spec = importlib.util.spec_from_file_location(
                "_bfm_precompute_loader", script_rel,
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_bfm_precompute_loader"] = mod
            spec.loader.exec_module(mod)
            self._minimal_mod = mod
            # Keypoint list for the load-time priv compose. A variant (e.g.
            # BFM-One) can pass a SHORTER list to drop redundant intermediate
            # links from the B-encode privileged_state — the raw motion data
            # is keypoint-agnostic, so the same minimal dataset is reused with
            # a different keypoint set (priv dim shrinks accordingly). The
            # dataset's stored ``keypoint_names`` is informational only; the
            # caller-provided override (or ``mod.KEYPOINT_NAMES``) wins.
            self._minimal_keypoint_names = list(
                keypoint_names if keypoint_names is not None else mod.KEYPOINT_NAMES
            )
            if keypoint_names is not None:
                print(
                    f"[FBCprExpertBuffer] using OVERRIDE keypoint list "
                    f"(K={len(self._minimal_keypoint_names)}, "
                    f"vs default K={len(mod.KEYPOINT_NAMES)}) for priv compose.",
                    flush=True,
                )
            urdf = raw.get("urdf_path") or mod.DEFAULT_URDF
            # Build the FK chain on the buffer's device so per-motion
            # compose runs entirely on GPU — ~20-40x faster than CPU on
            # a big dataset. When self.device is "cuda", pytorch_kinematics
            # moves its internal state to the same device.
            compose_device = str(self.device)
            self._minimal_chain = mod._build_chain(urdf, device=compose_device)
            self._minimal_default_q = torch.tensor(
                [float(x) for x in raw.get("default_dof_pos", mod.DEFAULT_DOF_POS)],
                dtype=torch.float32, device=self.device,
            )
            self._minimal_gravity = torch.tensor(
                list(raw.get("gravity", [0.0, 0.0, -1.0])),
                dtype=torch.float32, device=self.device,
            )
            self._minimal_history_len = (
                int(history_len_override) if history_len_override is not None
                else int(raw.get("history_length", 4))
            )
            if history_len_override is not None:
                print(f"[FBCprExpertBuffer] history_actor composed at H="
                      f"{self._minimal_history_len} (env override).", flush=True)
            self._minimal_action_scale = float(raw.get("action_scale", 0.25))
            self._minimal_action_clip = float(raw.get("action_clip", 5.0))
            self._minimal_resample_fps = float(raw.get("resample_fps", 0.0)) or None
            print(
                f"[FBCprExpertBuffer] minimal dataset detected; "
                f"load-time compose pipeline ready "
                f"(urdf={os.path.basename(urdf)}).",
                flush=True,
            )

        all_motions = raw["motions"]
        if motion_ids is None:
            motion_ids = list(all_motions.keys())
        else:
            missing = [m for m in motion_ids if m not in all_motions]
            if missing:
                raise KeyError(f"Motions not in dataset: {missing}")

        # Distributed shard: randomly permute the motion list with a seed
        # shared across ranks (so every rank gets a deterministic global
        # permutation), then take the contiguous slice belonging to this
        # rank. Result: each motion is owned by exactly ONE rank, and the
        # assignment is shuffled so locomotion/manipulation/stair/etc. are
        # spread across ranks rather than clumped by insertion order.
        self._shard_info: dict[str, int] = {
            "rank": int(shard_rank) if distributed_shard else 0,
            "world_size": int(shard_world_size) if distributed_shard else 1,
            "global_num_motions": len(motion_ids),
            "local_num_motions": len(motion_ids),
        }
        if distributed_shard and shard_world_size > 1:
            n_total = len(motion_ids)
            g = torch.Generator(device="cpu").manual_seed(int(shard_seed))
            perm = torch.randperm(n_total, generator=g).tolist()
            # Even split: rank r gets indices perm[r::W] for balanced load.
            local_idxs = [perm[i] for i in range(shard_rank, n_total, shard_world_size)]
            motion_ids = [motion_ids[i] for i in local_idxs]
            self._shard_info["local_num_motions"] = len(motion_ids)
            print(
                f"[FBCprExpertBuffer] distributed shard rank "
                f"{shard_rank}/{shard_world_size}: {len(motion_ids)}/{n_total} motions "
                f"(seed={shard_seed})",
                flush=True,
            )

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

        # Per-motion metadata for BFM-Terrain (and any other task that needs
        # to route RSI by motion source). ``motion_source_id`` defaults to 0
        # (= "regular" / flat-floor) if the .pt doesn't carry the field;
        # ``requires_terrain`` defaults to False. ``terrain_mesh_path`` may
        # be used by the env to build its shared terrain mesh.
        self._motion_source_id: list[int] = []
        self._requires_terrain: list[bool] = []
        self._terrain_id: list[int] = []
        self._terrain_mesh_paths: list[str] = []

        # Minimal path: batch-FK across all shard motions once, then
        # loop per-motion over the fast post-FK pipeline. FK is strongly
        # Python-launch-bound per call, so concat along T and one kernel
        # dispatch is ~30x faster than calling FK per motion. Velocity
        # FD must stay per-motion (it can't cross clip boundaries), but
        # that step is cheap.
        batched_fk_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if self._minimal:
            mod = self._minimal_mod
            dev = self.device
            # Collect raw tensors + lengths.
            raw_rp: list[torch.Tensor] = []
            raw_rq: list[torch.Tensor] = []
            raw_jp: list[torch.Tensor] = []
            lens: list[int] = []
            ordered_names: list[str] = list(self._motion_names)
            for name in ordered_names:
                m = all_motions[name]
                rp = m["root_pos"].to(dev, non_blocking=True)
                rq = mod._quat_normalize(m["root_quat"].to(dev, non_blocking=True))
                jp = m["joint_pos"].to(dev, non_blocking=True)
                raw_rp.append(rp); raw_rq.append(rq); raw_jp.append(jp)
                lens.append(int(rp.shape[0]))
            cat_rp = torch.cat(raw_rp, dim=0)
            cat_rq = torch.cat(raw_rq, dim=0)
            cat_jp = torch.cat(raw_jp, dim=0)
            # One batched FK call.
            print(
                f"[FBCprExpertBuffer] batch FK on {len(ordered_names)} motions, "
                f"{int(cat_rp.shape[0])} total frames ...",
                flush=True,
            )
            cat_wp, cat_wq = mod._world_fk(
                self._minimal_chain, cat_jp, cat_rp, cat_rq, self._minimal_keypoint_names,
            )
            # Split back into per-motion FK outputs.
            offs = 0
            for name, T in zip(ordered_names, lens):
                batched_fk_outputs[name] = (
                    cat_wp[offs:offs + T].contiguous(),
                    cat_wq[offs:offs + T].contiguous(),
                )
                offs += T
            # Cache raw pieces too so per-motion post-FK step avoids a
            # second copy.
            batched_raw = {
                name: (raw_rp[i], raw_rq[i], raw_jp[i])
                for i, name in enumerate(ordered_names)
            }
            # Free the concatenated scratch ASAP.
            del cat_rp, cat_rq, cat_jp, cat_wp, cat_wq, raw_rp, raw_rq, raw_jp

        # Progress bar for load-time compose (minimal datasets only).
        # Falls back to a silent no-op iterator when tqdm is unavailable
        # or the dataset is already precomputed.
        motion_iter = self._motion_names
        if self._minimal:
            try:
                from tqdm import tqdm
                shard_tag = ""
                if self._shard_info["world_size"] > 1:
                    shard_tag = (
                        f" [rank {self._shard_info['rank']}/"
                        f"{self._shard_info['world_size']}]"
                    )
                motion_iter = tqdm(
                    self._motion_names,
                    desc=f"[ExpertBuffer] compose{shard_tag}",
                    unit="motion",
                    dynamic_ncols=True,
                    mininterval=0.5,
                )
            except ImportError:
                pass

        for name in motion_iter:
            m = all_motions[name]
            # Minimal-dataset path: materialise state / priv / history /
            # velocities from the raw fields via _process_motion. We run
            # on CPU to keep torch JIT graph simple; the buffer moves to
            # GPU below. On a massive sharded dataset, this is the
            # dominant load-time cost (~5 ms per 500-frame motion).
            if self._minimal:
                mod = self._minimal_mod
                rp, rq, jp = batched_raw[name]
                wp_src, wq_src = batched_fk_outputs[name]
                src_fps_i = int(m["fps"])
                m = mod._process_motion_from_fk(
                    name=name,
                    root_pos=rp, root_quat=rq, joint_pos=jp,
                    world_pos_src=wp_src, world_quat_src=wq_src,
                    src_fps=src_fps_i, dt_min=1e-3,
                    keypoint_names=self._minimal_keypoint_names,
                    default_q=self._minimal_default_q,
                    gravity_world=self._minimal_gravity,
                    history_length=self._minimal_history_len,
                    action_scale=self._minimal_action_scale,
                    action_clip=self._minimal_action_clip,
                    resample_fps=self._minimal_resample_fps,
                    terrain_mesh=None,
                    terrain_z_precomputed=(
                        all_motions[name]["terrain_z"]
                        if "terrain_z" in all_motions[name] else None
                    ),
                    include_heading_body=self._priv_include_heading_body,
                )
                if m is None:
                    raise RuntimeError(
                        f"Minimal compose returned None for motion {name!r} "
                        f"(clip likely too short, T<3). Drop it from the dataset."
                    )
                # Carry over the original per-motion metadata tags.
                m["motion_source_id"] = int(all_motions[name].get("motion_source_id", 0))
                m["requires_terrain"] = bool(all_motions[name].get("requires_terrain", False))
                m["terrain_id"] = int(all_motions[name].get("terrain_id", -1))
                m["terrain_mesh_path"] = str(all_motions[name].get("terrain_mesh_path", ""))
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
            # Per-motion tags (default-safe for legacy .pt files).
            self._motion_source_id.append(int(m.get("motion_source_id", 0)))
            self._requires_terrain.append(bool(m.get("requires_terrain", False)))
            self._terrain_id.append(int(m.get("terrain_id", -1)))
            self._terrain_mesh_paths.append(str(m.get("terrain_mesh_path", "")))

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

        # Initial priors: length-proportional (default) so the per-
        # transition draw probability is equal across motions — otherwise
        # a short clip's transitions are seen (clip_length_ratio)x more
        # often per update than a long clip's. With a mix of 8k-frame
        # LAFAN clips and a 7468-frame continuous motion that matters.
        # ``update_priorities()`` (from the tracking eval) overwrites this.
        if (
            self._length_proportional_priors
            and len(self._lengths) > 0
            and sum(int(x) for x in self._lengths) > 0
        ):
            self._priorities = torch.tensor(
                [float(x) for x in self._lengths],
                dtype=torch.float32, device=self.device,
            )
            self._priorities = self._priorities / self._priorities.sum().clamp_min(1e-12)
        else:
            self._priorities = torch.ones(
                len(self._motion_names), dtype=torch.float32, device=self.device,
            )
            self._priorities = self._priorities / self._priorities.sum().clamp_min(1e-12)

        # --- Flat concatenated obs buffers for O(1) sample() --------------
        # Same trick as the RSI flat buffer: cat all motions along time,
        # keep per-motion ``_motion_obs_starts`` offsets, then sample()
        # can do ``flat[global_frame_idx]`` with ONE indexed read per leaf
        # instead of a Python for-loop over unique motions. Cost: 2× RAM
        # on the expert dataset (~2 GB here), which is fine on GPU.
        self._flat_state = torch.cat(self._states, dim=0).contiguous()
        self._flat_priv = torch.cat(self._privs, dim=0).contiguous()
        self._flat_last_action = torch.cat(self._last_actions, dim=0).contiguous()
        self._flat_history_actor = torch.cat(self._history_actors, dim=0).contiguous()
        _obs_offs = [0]
        for m in self._states:
            _obs_offs.append(_obs_offs[-1] + int(m.shape[0]))
        self._motion_obs_starts = torch.tensor(
            _obs_offs[:-1], dtype=torch.long, device=self.device,
        )  # [num_motions]; absolute start in flat_*.

        # Per-motion metadata as tensors (for device-efficient gather).
        self.motion_source_id_t = torch.tensor(
            self._motion_source_id, dtype=torch.long, device=self.device,
        )
        self.requires_terrain_t = torch.tensor(
            self._requires_terrain, dtype=torch.bool, device=self.device,
        )
        self.terrain_id_t = torch.tensor(
            self._terrain_id, dtype=torch.long, device=self.device,
        )
        # When RSI fields are available, expose per-RSI-frame metadata via
        # the flat buffer ordering. ``_motion_lengths_rsi`` gives the frame
        # count per motion; ``repeat_interleave`` maps that back to per-frame.
        if self.supports_reset_states:
            self.frame_motion_source_id = torch.repeat_interleave(
                self.motion_source_id_t, self._motion_lengths_rsi,
            )  # [total_frames]
            self.frame_requires_terrain = torch.repeat_interleave(
                self.requires_terrain_t, self._motion_lengths_rsi,
            )  # [total_frames]
        else:
            self.frame_motion_source_id = torch.zeros(0, dtype=torch.long, device=self.device)
            self.frame_requires_terrain = torch.zeros(0, dtype=torch.bool, device=self.device)

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
    def get_motion_window(self, motion_id: int, num_frames: int, start_frame: int = 0) -> dict:
        """Return ``num_frames`` frames of motion ``motion_id`` starting at
        ``start_frame`` (inclusive), as a dict.

        ``start_frame`` enables true Reference-State-Initialization (spawn at a
        random point within the motion, not always frame 0). It is clamped to
        ``[0, motion_len - 1]``; the returned window covers
        ``[start_frame : start_frame + num_frames]`` and is truncated at the
        motion end, so callers should read ``result['num_frames']`` (the actual
        REMAINING length from ``start_frame``) rather than assuming the request.
        """
        assert 0 <= motion_id < self.num_unique_motions
        mlen = self._lengths[motion_id]
        s = max(0, min(int(start_frame), mlen - 1))
        L = min(int(num_frames), mlen - s)
        e = s + L
        out = {
            "state": self._states[motion_id][s:e],
            "privileged_state": self._privs[motion_id][s:e],
            "last_action": self._last_actions[motion_id][s:e],
            "history_actor": self._history_actors[motion_id][s:e],
            "num_frames": L,
            "start_frame": s,
        }
        if self.supports_reset_states:
            out.update({
                "joint_pos": self._per_motion_joint_pos[motion_id][s:e],
                "joint_vel": self._per_motion_joint_vel[motion_id][s:e],
                "root_pos": self._per_motion_root_pos[motion_id][s:e],
                "root_quat": self._per_motion_root_quat[motion_id][s:e],
                "root_lin_vel": self._per_motion_root_lin_vel[motion_id][s:e],
                "root_ang_vel": self._per_motion_root_ang_vel[motion_id][s:e],
            })
        return out

    @torch.no_grad()
    def compute_body_pos(self, global_frames: torch.Tensor) -> torch.Tensor | None:
        """Run FK for the given flat-buffer frame indices, return [N, K, 3]."""
        if not self.supports_reset_states:
            return None
        mod = getattr(self, "_minimal_mod", None)
        chain = getattr(self, "_minimal_chain", None)
        if mod is None or chain is None:
            return None
        idx = global_frames.to(self.joint_pos_buffer.device)
        jp = self.joint_pos_buffer[idx]
        rp = self.root_pos_buffer[idx]
        rq = self.root_quat_buffer[idx]
        kp = getattr(self, "_minimal_keypoint_names", None) or mod.KEYPOINT_NAMES
        wp, _ = mod._world_fk(chain, jp, rp, rq, kp)
        return wp  # [N, K, 3]

    # -- RSI -------------------------------------------------------------- #

    @torch.no_grad()
    def get_reset_states_at(self, motion_ids: torch.Tensor, frame_offsets: torch.Tensor) -> dict:
        """Return RSI state for specific motion/frame pairs."""
        starts = self._motion_starts[motion_ids]
        lens = self._motion_lengths_rsi[motion_ids]
        offsets = frame_offsets.clamp(max=lens - 1)
        frame = starts + offsets
        return {
            "joint_pos": self.joint_pos_buffer[frame],
            "joint_vel": self.joint_vel_buffer[frame],
            "root_pos": self.root_pos_buffer[frame],
            "root_quat": self.root_quat_buffer[frame],
            "root_lin_vel": self.root_lin_vel_buffer[frame],
            "root_ang_vel": self.root_ang_vel_buffer[frame],
            "motion_source_id": self.frame_motion_source_id[frame],
            "requires_terrain": self.frame_requires_terrain[frame],
        }

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
            "motion_source_id": self.frame_motion_source_id[frame],
            "requires_terrain": self.frame_requires_terrain[frame],
        }

    def sample_tracking_trajectories(
        self, num_trajs: int, traj_length: int,
        anchor_canon_xy: torch.Tensor | None = None,
        anchor_canon_yaw: torch.Tensor | None = None,
    ) -> dict:
        """Sample contiguous expert sub-trajectories for tracking.

        Returns a dict with:
            - "observation": dict of obs tensors [B, ...]  (B = num_trajs * traj_length)
            - "next_observation": dict [B, ...]
            - "motion_ids": [num_trajs] — which motion each trajectory is from
            - "starts": [num_trajs] — start frame within each motion
            - "motion_lens": [num_trajs] — usable length of each motion
        """
        MIN_FRAMES = 50
        eligible_mask = self._lengths_t >= MIN_FRAMES
        if not bool(eligible_mask.any().item()):
            raise RuntimeError(f"No motion has at least {MIN_FRAMES} frames.")
        eligible_idx = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
        eligible_priors = self._priorities[eligible_idx]
        eligible_priors = eligible_priors / eligible_priors.sum().clamp_min(1e-12)
        eligible_lengths = self._lengths_t[eligible_idx]

        sel = torch.multinomial(eligible_priors, num_trajs, replacement=True)
        motion_picks = eligible_idx[sel]
        motion_lens = eligible_lengths[sel]
        rand01 = torch.rand(num_trajs, device=self.device)
        max_start = (motion_lens - traj_length - 1).clamp_min(0).to(torch.float32)
        starts = (rand01 * (max_start + 1.0)).floor().to(torch.long)

        arange = torch.arange(traj_length, device=self.device).unsqueeze(0)
        raw_frame = starts.unsqueeze(1) + arange
        usable_lens = (motion_lens - 1).clamp_min(1).unsqueeze(1)
        # Terrain motions: clamp to last frame (pad). Others: circular wrap.
        is_terrain = self.requires_terrain_t[motion_picks].unsqueeze(1)  # [N, 1]
        frame_cur_wrap = raw_frame % usable_lens
        frame_cur_clamp = raw_frame.clamp(max=usable_lens - 1)
        frame_cur = torch.where(is_terrain, frame_cur_clamp, frame_cur_wrap).reshape(-1)
        frame_nxt_wrap = (raw_frame + 1) % usable_lens
        frame_nxt_clamp = (raw_frame + 1).clamp(max=usable_lens - 1)
        frame_nxt = torch.where(is_terrain, frame_nxt_clamp, frame_nxt_wrap).reshape(-1)
        motion_flat = motion_picks.unsqueeze(1).expand(-1, traj_length).reshape(-1)

        global_cur = self._motion_obs_starts[motion_flat] + frame_cur
        global_nxt = self._motion_obs_starts[motion_flat] + frame_nxt

        obs = {
            "state": self._flat_state[global_cur],
            "privileged_state": self._flat_priv[global_cur],
            "last_action": self._flat_last_action[global_cur],
            "history_actor": self._flat_history_actor[global_cur],
        }
        next_obs = {
            "state": self._flat_state[global_nxt],
            "privileged_state": self._flat_priv[global_nxt],
            "last_action": self._flat_last_action[global_nxt],
            "history_actor": self._flat_history_actor[global_nxt],
        }
        # Anchored variant: attach expert anchored_pose A^-1 g, each row's pose
        # canonicalised to its OWN window-start frame (start_flat) so it
        # self-zeros like a rollout episode. The shared T-window mean then
        # windows the spatial z identically to the local z.
        #
        # CRITICAL — this is the TRACKING-z path (drives rollout tracking envs),
        # NOT the disc/FB augmentation path. The rollout-z's anchor MUST match
        # the anchor the env uses for the actor's ``anchored_pose`` obs, else
        # z = B(.) and the actor obs live in different SE(2) frames and the
        # implicit reward ⟨B(s),z⟩ never aligns -> the spatial command is noise
        # and the global tracking deviation can't drop (while the replay-path
        # Qfb, which is internally anchor-consistent, still rises).
        #
        # TWO-FRAME ANCHOR (sim <-> motion). At reset the robot inits at sim
        # pose A_init <-> motion pose A^m_init (RSI correspondence). The caller
        # samples ONE offset ``A_anchor`` (in the init-LOCAL frame) and uses it
        # on BOTH sides: the env sets its ``anchored_pose`` anchor to
        # A_init·A_anchor (sim space), and we encode z under A^m_init·A_anchor
        # (motion space). Because each side's init pose cancels in its own
        # local frame, both reduce to A_anchor and the obs/z frames coincide,
        # while A_anchor != 0 DISPLACES the spatial goal (filling the displaced-
        # goal coverage hole). ``anchor_canon_{xy,yaw}`` ARE that A_anchor,
        # expressed in the window-start canonical frame (the same frame
        # ``_canon_pose_at`` self-zeros to). A_anchor=0 reduces to the spawn-
        # anchored special case (== eval ``_build_global_track_z``).
        if self._emit_anchored_pose and getattr(self, "root_pos_buffer", None) is not None:
            start_flat = starts.unsqueeze(1).expand(-1, traj_length).reshape(-1)
            # Per-traj A_anchor (canonical frame), broadcast to each frame row.
            if anchor_canon_xy is not None:
                aA_xy = anchor_canon_xy.to(self.device).view(-1, 1, 2).expand(
                    -1, traj_length, -1).reshape(-1, 2)
                aA_yaw = anchor_canon_yaw.to(self.device).view(-1, 1).expand(
                    -1, traj_length).reshape(-1)
            else:
                # No offset supplied -> A_anchor = 0 (spawn-anchored).
                B_T = frame_cur.shape[0]
                aA_xy = torch.zeros(B_T, 2, device=self.device)
                aA_yaw = torch.zeros(B_T, device=self.device)
            gc_xy, gc_yaw = self._canon_pose_at(frame_cur, start_flat, motion_flat)
            gn_xy, gn_yaw = self._canon_pose_at(frame_nxt, start_flat, motion_flat)
            obs["anchored_pose"] = self.encode_anchored_pose(
                gc_xy, gc_yaw, aA_xy, aA_yaw, self._anchored_pose_clamp)
            next_obs["anchored_pose"] = self.encode_anchored_pose(
                gn_xy, gn_yaw, aA_xy, aA_yaw, self._anchored_pose_clamp)
            # Anchor-frame body pose: B was trained on priv body POS/ROT6D
            # reframed into the SAME anchor. The raw ``_flat_priv`` is heading-
            # frame, so reframe into the A_anchor frame: root pose in that frame
            # is (cr = A_anchor^-1·g_canon, dθ = g_canon_yaw - A_anchor_yaw) —
            # byte-identical to how the env / algo preamble reframe priv.
            if self._anchor_frame_body:
                R = self._anchored_pose_clamp
                K = self._priv_K_for_reframe(obs["privileged_state"])
                if K is not None:
                    cr_c, dth_c = self._cr_in_anchor(gc_xy, gc_yaw, aA_xy, aA_yaw)
                    cr_n, dth_n = self._cr_in_anchor(gn_xy, gn_yaw, aA_xy, aA_yaw)
                    obs["privileged_state"] = self.reframe_priv_body(
                        obs["privileged_state"], cr_c, dth_c, K, R, True)
                    next_obs["privileged_state"] = self.reframe_priv_body(
                        next_obs["privileged_state"], cr_n, dth_n, K, R, True)
        return {
            "observation": obs,
            "next_observation": next_obs,
            "motion_ids": motion_picks,
            "starts": starts,
            "motion_lens": motion_lens,
            "requires_terrain": self.requires_terrain_t[motion_picks],
        }

    # -- priority updates (stub-ish; accepted by agent) --------------------

    def update_priorities(self, priorities: torch.Tensor, idxs: torch.Tensor | None = None) -> None:
        """Update per-motion sampling weights.

        If ``idxs`` is None, expects ``priorities`` to have length equal to
        the number of motions. Otherwise scatters the new values into the
        given indices. Non-negative values are required; values are then
        renormalised to sum to 1.

        When ``length_proportional_priors`` is set, the incoming weights
        (e.g. MPJPE-derived) are multiplied by per-motion length BEFORE
        normalisation so the expected per-transition draw probability
        stays uniform across motions regardless of clip-length skew.
        """
        priorities = priorities.to(self.device).float().clamp_min(0.0)
        if idxs is None:
            if priorities.numel() != len(self._motion_names):
                raise ValueError(
                    f"Expected priorities of length {len(self._motion_names)}, got {priorities.numel()}"
                )
            if self._length_proportional_priors:
                lens = torch.tensor(
                    [float(x) for x in self._lengths],
                    dtype=priorities.dtype, device=self.device,
                )
                priorities = priorities * lens
            self._priorities = priorities
        else:
            idxs = idxs.to(self.device).long()
            if self._length_proportional_priors:
                lens = torch.tensor(
                    [float(self._lengths[int(i)]) for i in idxs.tolist()],
                    dtype=priorities.dtype, device=self.device,
                )
                priorities = priorities * lens
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
        # Round batch down to a multiple of seq_length.
        num_slices = max(1, batch_size // seq_length)

        # Eligibility: need at least 50 frames (~1s at 50Hz). Short motions
        # are circular-padded so they can be sampled for tracking trajectories
        # even if shorter than seq_length.
        MIN_FRAMES = 50
        eligible_mask = self._lengths_t >= MIN_FRAMES
        if not bool(eligible_mask.any().item()):
            raise RuntimeError(f"No motion has at least {MIN_FRAMES} frames.")
        eligible_idx = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
        eligible_priors = self._priorities[eligible_idx]
        eligible_priors = eligible_priors / eligible_priors.sum().clamp_min(1e-12)
        eligible_lengths = self._lengths_t[eligible_idx]

        # Pick a motion per slice (weighted), then a random start.
        sel = torch.multinomial(eligible_priors, num_slices, replacement=True)
        motion_picks = eligible_idx[sel]                    # [num_slices]
        motion_lens = eligible_lengths[sel]                 # [num_slices]
        # For motions longer than seq_length: start in [0, T-seq_length-1].
        # For shorter motions: start at 0, frames will wrap via modulo.
        rand01 = torch.rand(num_slices, device=self.device)
        max_start = (motion_lens - seq_length - 1).clamp_min(0).to(torch.float32)
        starts = (rand01 * (max_start + 1.0)).floor().to(torch.long)

        # Build per-slice arange window and flatten.
        arange = torch.arange(seq_length, device=self.device).unsqueeze(0)     # [1, seq_length]
        raw_frame = starts.unsqueeze(1) + arange                                # [num_slices, seq_length]
        # Circular wrap for short motions: frame % (T-1) keeps within
        # valid obs range [0, T-2] (T-1 is next_obs of last transition).
        usable_lens = (motion_lens - 1).clamp_min(1).unsqueeze(1)              # [num_slices, 1]
        frame_cur = (raw_frame % usable_lens).reshape(-1)                       # [B]
        frame_nxt = ((raw_frame + 1) % usable_lens).reshape(-1)                 # [B]
        motion_flat = motion_picks.unsqueeze(1).expand(-1, seq_length).reshape(-1)  # [B]

        # Fully-vectorized gather using the flat obs buffers (one big
        # indexed read per leaf tensor instead of a Python loop over
        # unique motions). Scales to O(1) in the number of motions —
        # matters a lot when the dataset is clip-diced (862 motions
        # instead of 40) and ``num_slices`` would otherwise touch ~128
        # distinct motions per ``sample()`` call, 16 times per iter.
        global_cur = self._motion_obs_starts[motion_flat] + frame_cur  # [B]
        global_nxt = self._motion_obs_starts[motion_flat] + frame_nxt  # [B]
        out_state = self._flat_state[global_cur]
        out_priv = self._flat_priv[global_cur]
        out_act = self._flat_last_action[global_cur]
        out_hist = self._flat_history_actor[global_cur]
        out_state_n = self._flat_state[global_nxt]
        out_priv_n = self._flat_priv[global_nxt]
        out_act_n = self._flat_last_action[global_nxt]
        out_hist_n = self._flat_history_actor[global_nxt]

        B = out_state.shape[0]
        terminated = torch.zeros((B, 1), dtype=torch.bool, device=self.device)
        z_dummy = torch.zeros((B, 0), device=self.device)   # filled by caller if needed
        # The agent signature expects a z entry; we provide zeros of a
        # caller-determined dim. We cannot know z_dim here, so we leave it
        # zero-width and let the caller broadcast / fill. The BFM reference
        # implementation uses the agent's own z-sampler to overwrite this
        # before the discriminator step.

        obs_dict = {
            "state": out_state,
            "privileged_state": out_priv,
            "last_action": out_act,
            "history_actor": out_hist,
        }
        next_obs_dict = {
            "state": out_state_n,
            "privileged_state": out_priv_n,
            "last_action": out_act_n,
            "history_actor": out_hist_n,
        }
        # Anchored variant: attach the expert's anchored pose A^-1 g. Each row's
        # pose is canonicalised to its OWN window-start frame (start_flat), then
        # an anchor A ~ p_A is applied — so expert windows self-zero like rollout
        # episodes. We also return the RAW canonical next-pose (x,y,yaw), which
        # the expert-z re-anchor needs to rebuild anchored_pose under an
        # arbitrary destination anchor A_i (the normalizer would drop it from the
        # obs dict, so it rides at the top level of ``next``).
        out_dict = {
            "observation": obs_dict,
            "action": out_act,   # reconstructed "last_action" doubles as the demo action
            "z": z_dummy,
            "next": {
                "observation": next_obs_dict,
                "terminated": terminated,
            },
        }
        if self._emit_anchored_pose and getattr(self, "root_pos_buffer", None) is not None:
            start_flat = starts.unsqueeze(1).expand(-1, seq_length).reshape(-1)
            obs_dict["anchored_pose"] = self._anchored_pose_at(frame_cur, motion_flat, start_flat)
            next_obs_dict["anchored_pose"] = self._anchored_pose_at(frame_nxt, motion_flat, start_flat)
            nxy, nyaw = self._canon_pose_at(frame_nxt, start_flat, motion_flat)
            out_dict["next"]["canon_pose"] = torch.cat([nxy, nyaw.unsqueeze(-1)], dim=-1)  # [B,3]
            # Also expose the CURRENT-frame canonical pose (sub-traj-start frame)
            # so the preamble can reframe the (heading-frame) expert priv body
            # block to the per-row anchor A_i. The priv itself stays heading-frame
            # here; reframing for BOTH train and expert happens in the preamble
            # under the same p_A, so the disc sees one consistent distribution.
            cxy, cyaw = self._canon_pose_at(frame_cur, start_flat, motion_flat)
            out_dict["canon_pose"] = torch.cat([cxy, cyaw.unsqueeze(-1)], dim=-1)  # [B,3]
        return out_dict

    def _canon_pose_at(self, frame_local: torch.Tensor, start_local: torch.Tensor,
                       motion_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonical (sub-trajectory-start-relative) SE(2) pose of an expert
        frame: ``g_canon = [Rot(-s_yaw)(g_xy - s_xy), wrap(g_yaw - s_yaw)]``
        where ``s`` is the window's OWN start frame (``start_local``). This makes
        each expert window self-zero exactly like a rollout episode (which zeros
        at its spawn pose) — so expert and policy poses live in one comparable
        frame, and cross-row anchor subtraction stays bounded by per-window
        travel rather than the motion's mocap-world origin.

        ``frame_local`` / ``start_local`` are motion-local frame indices; root
        buffers are indexed via ``_motion_starts`` (RSI flat space).
        Returns ``(g_xy_canon [N,2], g_yaw_canon [N])``.
        """
        base = self._motion_starts[motion_flat]
        lens = self._motion_lengths_rsi[motion_flat]
        fl = torch.minimum(frame_local, (lens - 1).clamp_min(0))
        sl = torch.minimum(start_local, (lens - 1).clamp_min(0))

        def _pose(idx_local):
            gi = base + idx_local
            xy = self.root_pos_buffer[gi][:, :2]
            q = self.root_quat_buffer[gi]
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                              1 - 2 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]))
            return xy, yaw

        g_xy, g_yaw = _pose(fl)
        s_xy, s_yaw = _pose(sl)
        d = g_xy - s_xy
        ca, sa = torch.cos(-s_yaw), torch.sin(-s_yaw)
        rel_x = ca * d[:, 0] - sa * d[:, 1]
        rel_y = sa * d[:, 0] + ca * d[:, 1]
        rel_xy = torch.stack([rel_x, rel_y], dim=-1)
        rel_yaw = torch.atan2(torch.sin(g_yaw - s_yaw), torch.cos(g_yaw - s_yaw))
        return rel_xy, rel_yaw

    @staticmethod
    def _signed_log_unit(v: torch.Tensor, R: float, s0: float = 1.0) -> torch.Tensor:
        """Signed-log range compression (byte-identical to
        ``AnchoredFBCprAux._signed_log_unit`` and the env encoder)."""
        return torch.sign(v) * torch.log1p(v.abs() / s0) / math.log1p(R / s0)

    @staticmethod
    def _cr_in_anchor(g_xy: torch.Tensor, g_yaw: torch.Tensor,
                      a_xy: torch.Tensor, a_yaw: torch.Tensor):
        """Root pose (cr_xy metres, dθ) expressed in anchor frame A: cr =
        Rot(-a_yaw)(g_xy - a_xy), dθ = wrap(g_yaw - a_yaw). Byte-identical to
        ``AnchoredFBCprAux._anchor_priv_pre_normalize._cr`` — the (cr, dθ) fed to
        ``reframe_priv_body``. Inputs/outputs in the canonical window frame."""
        d = g_xy - a_xy
        ca, sa = torch.cos(-a_yaw), torch.sin(-a_yaw)
        cr = torch.stack([ca * d[:, 0] - sa * d[:, 1],
                          sa * d[:, 0] + ca * d[:, 1]], dim=-1)
        dth = torch.atan2(torch.sin(g_yaw - a_yaw), torch.cos(g_yaw - a_yaw))
        return cr, dth

    def _priv_K_for_reframe(self, priv: torch.Tensor | None) -> int | None:
        """Keypoint count K from the priv DIMENSION (layout root_height_obs=True:
        dim = 1 + (K-1)*3 + K*6 + K*3 + K*3 = 15K - 2 -> K = (dim+2)/15). Matches
        ``AnchoredFBCprAux._priv_K`` so the tracking-z reframe uses the same K as
        training. Cached."""
        K = getattr(self, "_priv_K_reframe_cache", "unset")
        if K != "unset":
            return K
        K = None
        if priv is not None:
            dim = int(priv.shape[-1])
            if (dim + 2) % 15 == 0:
                K = (dim + 2) // 15
        self._priv_K_reframe_cache = K
        return K

    @staticmethod
    def reframe_priv_body(priv: torch.Tensor, cr_xy: torch.Tensor,
                          dtheta: torch.Tensor, K: int, R: float,
                          root_height_obs: bool = True) -> torch.Tensor:
        """Re-express the BODY-POSE block of a heading-frame ``privileged_state``
        into an anchor frame. POS (xy) + ROT6D only; height, lin/ang vel
        untouched. ``(cr_xy, dθ)`` is the ROOT pose in the target anchor frame
        (cr_xy metres, dθ = root_yaw - anchor_yaw). BYTE-IDENTICAL to the env's
        ``reframe_priv_body_anchor`` and the algo's reframer.

        Layout: ``[root_h(1)? | pos((K-1)*3, pelvis dropped) | rot6d(K*6) |
        vel(K*3) | ang(K*3)]``."""
        out = priv.clone()
        off = 1 if root_height_obs else 0
        npos = (K - 1) * 3
        nrot = K * 6
        c = torch.cos(dtheta).unsqueeze(-1)
        s = torch.sin(dtheta).unsqueeze(-1)
        crx = cr_xy[:, 0:1]
        cry = cr_xy[:, 1:2]
        pos = out[:, off:off + npos].view(-1, K - 1, 3)
        hx, hy = pos[..., 0], pos[..., 1]
        rx = c * hx - s * hy + crx
        ry = s * hx + c * hy + cry
        pos = torch.stack(
            [FBCprExpertBuffer._signed_log_unit(rx, R),
             FBCprExpertBuffer._signed_log_unit(ry, R), pos[..., 2]], dim=-1)
        out[:, off:off + npos] = pos.reshape(out.shape[0], -1)
        rs = off + npos
        rot = out[:, rs:rs + nrot].view(-1, K, 6).clone()
        for base in (0, 3):
            vx = rot[..., base + 0].clone()
            vy = rot[..., base + 1].clone()
            rot[..., base + 0] = c * vx - s * vy
            rot[..., base + 1] = s * vx + c * vy
        out[:, rs:rs + nrot] = rot.reshape(out.shape[0], -1)
        return out

    @staticmethod
    def encode_anchored_pose(g_xy: torch.Tensor, g_yaw: torch.Tensor,
                             a_xy: torch.Tensor, a_yaw: torch.Tensor,
                             clamp: float) -> torch.Tensor:
        """``A^-1 g -> [signed_log(px,R), signed_log(py,R), cosθ, sinθ]`` (in
        (-1,1)). ``clamp`` is the full-scale range R (m). Byte-identical to env
        ``_obs_anchored_pose`` and ``AnchoredFBCprAux._encode_anchored_pose``.
        ``g_*`` and ``a_*`` must be in the SAME frame (canonical sub-traj-start)."""
        d = g_xy - a_xy
        ca, sa = torch.cos(-a_yaw), torch.sin(-a_yaw)
        px = FBCprExpertBuffer._signed_log_unit(ca * d[:, 0] - sa * d[:, 1], clamp)
        py = FBCprExpertBuffer._signed_log_unit(sa * d[:, 0] + ca * d[:, 1], clamp)
        theta = g_yaw - a_yaw
        return torch.stack([px, py, torch.cos(theta), torch.sin(theta)], dim=-1)

    def _anchored_pose_at(self, frame_local: torch.Tensor, motion_flat: torch.Tensor,
                          start_local: torch.Tensor | None = None) -> torch.Tensor:
        """Encode A^-1 g for expert frames under an anchor A ~ p_A sampled from
        the SAME mixture as the policy (prob ``alpha`` at the frame's own pose,
        else random ±range xy / ±π yaw). Poses are first canonicalised to the
        window's own start frame (``_canon_pose_at``), so expert and policy share
        one frame. ``start_local`` defaults to the per-row window start; if None
        (legacy callers) we self-zero each frame (start == frame), which reduces
        to the old origin-invariant single-row behaviour.

        DISCRIMINATOR/FB-AUGMENTATION PATH ONLY (called from ``sample()``). The
        random p_A anchor randomises the spatial-z region so the discriminator
        cannot shortcut on it. The TRACKING-z path (``sample_tracking_trajectories``)
        does NOT use this — it encodes under ONE shared rollout anchor A_anchor
        (matching the env's actor ``anchored_pose`` obs), see there.

        Returns the anchored_pose obs ``[px, py, cosθ, sinθ]``.
        """
        if start_local is None:
            start_local = frame_local
        g_xy, g_yaw = self._canon_pose_at(frame_local, start_local, motion_flat)

        # Anchor A ~ p_A relative to THIS frame's own (canonical) pose.
        N = g_xy.shape[0]
        dev = g_xy.device
        a_xy = g_xy.clone()
        a_yaw = g_yaw.clone()
        is_rand = torch.rand(N, device=dev) >= self._anchor_alpha_gt
        r = self._anchor_random_xy_range
        rand_xy = g_xy + (torch.rand(N, 2, device=dev) * 2 - 1) * r
        rand_yaw = (torch.rand(N, device=dev) * 2 - 1) * math.pi
        a_xy = torch.where(is_rand.unsqueeze(-1), rand_xy, a_xy)
        a_yaw = torch.where(is_rand, rand_yaw, a_yaw)
        return self.encode_anchored_pose(g_xy, g_yaw, a_xy, a_yaw, self._anchored_pose_clamp)

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
        # Round each chunk's batch down to a multiple of seq_length.
        batch_size = max(seq_length, (batch_size // seq_length) * seq_length)

        def _move(x: torch.Tensor) -> torch.Tensor:
            return x.to(target_device, non_blocking=True) if x.device != torch.device(target_device) else x

        # Single fused sample for all chunks: one motion/frame draw, one
        # gather per leaf, then slice. Eliminates the num_chunks-sized
        # Python loop of sample() calls (the hot path when the dataset
        # is clip-diced — ~862 motions, with hundreds of unique picks
        # per batch). Keeps the N × seq_length contiguous layout.
        big = self.sample(batch_size * num_chunks, seq_length=seq_length)

        # Propagate EVERY obs key sample() produced — not a hardcoded subset.
        # In particular ``anchored_pose`` (emitted when emit_anchored_pose=True)
        # must survive to the disc / encode_expert so B_spatial sees a REAL
        # expert pose rather than a zero-filled constant.
        obs_keys = tuple(big["observation"].keys())
        obs_flat = {k: _move(big["observation"][k]) for k in obs_keys}
        next_obs_flat = {k: _move(big["next"]["observation"][k]) for k in obs_keys}
        action_flat = _move(big["action"])
        z_flat = _move(big["z"])
        term_flat = _move(big["next"]["terminated"])
        # Raw canonical next-pose (x,y,yaw), if the anchored sample emitted it —
        # the expert-z re-anchor needs it to rebuild anchored_pose under A_i.
        canon_flat = _move(big["next"]["canon_pose"]) if "canon_pose" in big["next"] else None
        # Current-frame canonical pose (top level) — preamble uses it to reframe
        # the expert priv body block to the per-row anchor A_i.
        canon_cur_flat = _move(big["canon_pose"]) if "canon_pose" in big else None

        chunks: list[dict] = []
        for i in range(num_chunks):
            s = slice(i * batch_size, (i + 1) * batch_size)
            nxt = {
                "observation": {k: next_obs_flat[k][s] for k in obs_keys},
                "terminated": term_flat[s],
            }
            if canon_flat is not None:
                nxt["canon_pose"] = canon_flat[s]
            chunk = {
                "observation": {k: obs_flat[k][s] for k in obs_keys},
                "action": action_flat[s],
                "z": z_flat[s],
                "next": nxt,
            }
            if canon_cur_flat is not None:
                chunk["canon_pose"] = canon_cur_flat[s]
            chunks.append(chunk)
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
