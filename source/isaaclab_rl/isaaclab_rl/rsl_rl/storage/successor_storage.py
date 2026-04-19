# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


class SuccessorStorage:
    """Replay-style rollout storage for the sparse-constraint successor tracking algorithm.

    Stores transitions with constraint set data (padded), snippets, and next-state information.
    Supports circular-buffer write and uniform-random mini-batch sampling.
    """

    class Transition:
        def __init__(self):
            self.observations: torch.Tensor = None  # type: ignore
            self.privileged_observations: torch.Tensor = None  # type: ignore
            self.actions: torch.Tensor = None  # type: ignore
            self.rewards: torch.Tensor = None  # type: ignore
            self.dones: torch.Tensor = None  # type: ignore
            # next step
            self.next_observations: torch.Tensor = None  # type: ignore
            self.next_privileged_observations: torch.Tensor = None  # type: ignore
            # constraint set (padded to max_constraints)
            self.constraint_keypoint_ids: torch.Tensor = None  # type: ignore
            self.constraint_targets: torch.Tensor = None  # type: ignore
            self.constraint_taus: torch.Tensor = None  # type: ignore
            self.constraint_weights: torch.Tensor = None  # type: ignore
            self.constraint_mask: torch.Tensor = None  # type: ignore
            # next constraint set
            self.next_constraint_keypoint_ids: torch.Tensor = None  # type: ignore
            self.next_constraint_targets: torch.Tensor = None  # type: ignore
            self.next_constraint_taus: torch.Tensor = None  # type: ignore
            self.next_constraint_weights: torch.Tensor = None  # type: ignore
            self.next_constraint_mask: torch.Tensor = None  # type: ignore
            # snippet for discriminator
            self.snippet: torch.Tensor = None  # type: ignore

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: list[int],
        privileged_obs_shape: list[int],
        actions_shape: list[int],
        max_constraints: int,
        target_dim: int,
        snippet_dim: int,
        device: str = "cpu",
        storage_device: str | None = None,
        sample_device: str | None = None,
    ):
        """
        Args:
            device: **Deprecated**. Used as both storage and sample device when
                ``storage_device`` / ``sample_device`` aren't given. Kept for
                backward compatibility.
            storage_device: Where the replay tensors live. Defaults to ``device``.
                Set to ``"cpu"`` to fit large replay buffers without consuming
                GPU memory — sampled batches are moved to ``sample_device``
                with non-blocking pinned transfers.
            sample_device: Where sampled mini-batches are served (i.e. the
                training device). Defaults to ``device``.
        """
        self.storage_device = storage_device if storage_device is not None else device
        self.sample_device = sample_device if sample_device is not None else device
        # Keep ``self.device`` for legacy callers but it is now just an alias
        # for storage_device.
        self.device = self.storage_device

        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.max_constraints = max_constraints
        self.target_dim = target_dim

        T = num_transitions_per_env
        N = num_envs
        M = max_constraints

        # Allocator helper. On CPU we pin memory so ``.to(device, non_blocking=True)``
        # from the sample path is genuinely async with the training stream.
        sd = self.storage_device
        pin = (isinstance(sd, str) and sd == "cpu") or (hasattr(sd, "type") and sd.type == "cpu")

        def _alloc(*shape, dtype=torch.float32):
            t = torch.zeros(*shape, dtype=dtype, device=sd)
            if pin:
                try:
                    t = t.pin_memory()
                except (RuntimeError, AssertionError):
                    # Pinning can fail under some fakesim / distributed setups;
                    # fall back silently to an unpinned CPU tensor.
                    pass
            return t

        # Core transition data
        self.observations = _alloc(T, N, *obs_shape)
        self.privileged_observations = _alloc(T, N, *privileged_obs_shape)
        self.actions = _alloc(T, N, *actions_shape)
        self.rewards = _alloc(T, N, 1)
        self.dones = _alloc(T, N, 1)

        # Next-step data
        self.next_observations = _alloc(T, N, *obs_shape)
        self.next_privileged_observations = _alloc(T, N, *privileged_obs_shape)

        # Constraint set data (current)
        self.constraint_keypoint_ids = _alloc(T, N, M, dtype=torch.long)
        self.constraint_targets = _alloc(T, N, M, target_dim)
        self.constraint_taus = _alloc(T, N, M)
        self.constraint_weights = _alloc(T, N, M)
        self.constraint_mask = _alloc(T, N, M)

        # Constraint set data (next)
        self.next_constraint_keypoint_ids = _alloc(T, N, M, dtype=torch.long)
        self.next_constraint_targets = _alloc(T, N, M, target_dim)
        self.next_constraint_taus = _alloc(T, N, M)
        self.next_constraint_weights = _alloc(T, N, M)
        self.next_constraint_mask = _alloc(T, N, M)

        # Snippet for discriminator
        self.snippets = _alloc(T, N, snippet_dim)

        self.step = 0
        self._full = False

    @property
    def size(self) -> int:
        if self._full:
            return self.num_transitions_per_env * self.num_envs
        return self.step * self.num_envs

    def add_transitions(self, transition: Transition):
        """Write one per-env transition into the circular buffer.

        ``copy_(src)`` handles cross-device copies automatically, so incoming
        transitions from GPU-side rollouts can be stored into a CPU buffer
        without an explicit ``.to()``.
        """
        idx = self.step % self.num_transitions_per_env

        self.observations[idx].copy_(transition.observations, non_blocking=True)
        self.privileged_observations[idx].copy_(transition.privileged_observations, non_blocking=True)
        self.actions[idx].copy_(transition.actions, non_blocking=True)
        self.rewards[idx].copy_(transition.rewards.view(-1, 1), non_blocking=True)
        self.dones[idx].copy_(transition.dones.view(-1, 1).float(), non_blocking=True)

        self.next_observations[idx].copy_(transition.next_observations, non_blocking=True)
        self.next_privileged_observations[idx].copy_(transition.next_privileged_observations, non_blocking=True)

        self.constraint_keypoint_ids[idx].copy_(transition.constraint_keypoint_ids, non_blocking=True)
        self.constraint_targets[idx].copy_(transition.constraint_targets, non_blocking=True)
        self.constraint_taus[idx].copy_(transition.constraint_taus, non_blocking=True)
        self.constraint_weights[idx].copy_(transition.constraint_weights, non_blocking=True)
        self.constraint_mask[idx].copy_(transition.constraint_mask, non_blocking=True)

        self.next_constraint_keypoint_ids[idx].copy_(transition.next_constraint_keypoint_ids, non_blocking=True)
        self.next_constraint_targets[idx].copy_(transition.next_constraint_targets, non_blocking=True)
        self.next_constraint_taus[idx].copy_(transition.next_constraint_taus, non_blocking=True)
        self.next_constraint_weights[idx].copy_(transition.next_constraint_weights, non_blocking=True)
        self.next_constraint_mask[idx].copy_(transition.next_constraint_mask, non_blocking=True)

        self.snippets[idx].copy_(transition.snippet, non_blocking=True)

        self.step += 1
        if self.step >= self.num_transitions_per_env:
            self._full = True

    def clear(self):
        self.step = 0
        self._full = False

    def _flatten_live_view(self):
        """Return flat views of every tensor into the populated region of the buffer."""
        max_t = self.num_transitions_per_env if self._full else self.step
        total = max_t * self.num_envs

        views = dict(
            obs=self.observations[:max_t].reshape(total, -1),
            priv=self.privileged_observations[:max_t].reshape(total, -1),
            act=self.actions[:max_t].reshape(total, -1),
            rew=self.rewards[:max_t].reshape(total, -1),
            done=self.dones[:max_t].reshape(total, -1),
            next_obs=self.next_observations[:max_t].reshape(total, -1),
            next_priv=self.next_privileged_observations[:max_t].reshape(total, -1),
            c_kid=self.constraint_keypoint_ids[:max_t].reshape(total, self.max_constraints),
            c_tgt=self.constraint_targets[:max_t].reshape(total, self.max_constraints, self.target_dim),
            c_tau=self.constraint_taus[:max_t].reshape(total, self.max_constraints),
            c_w=self.constraint_weights[:max_t].reshape(total, self.max_constraints),
            c_m=self.constraint_mask[:max_t].reshape(total, self.max_constraints),
            nc_kid=self.next_constraint_keypoint_ids[:max_t].reshape(total, self.max_constraints),
            nc_tgt=self.next_constraint_targets[:max_t].reshape(total, self.max_constraints, self.target_dim),
            nc_tau=self.next_constraint_taus[:max_t].reshape(total, self.max_constraints),
            nc_w=self.next_constraint_weights[:max_t].reshape(total, self.max_constraints),
            nc_m=self.next_constraint_mask[:max_t].reshape(total, self.max_constraints),
            snip=self.snippets[:max_t].reshape(total, -1),
        )
        return views, total

    def _move_to_sample_device(self, t: torch.Tensor) -> torch.Tensor:
        """Move a sampled tensor to the training device with non-blocking xfer."""
        if t.device == torch.device(self.sample_device) if isinstance(self.sample_device, str) else t.device == self.sample_device:
            return t
        return t.to(self.sample_device, non_blocking=True)

    def sample(self, batch_size: int):
        """Draw a single mini-batch uniformly at random from the populated region.

        Used for true off-policy replay — each call is an independent sample with
        replacement. Cheaper than re-permuting the whole buffer when we only want
        a handful of updates per iteration.

        Sampled tensors are moved to ``sample_device`` via a non-blocking copy;
        with pinned-memory CPU storage this overlaps with GPU compute.
        """
        views, total = self._flatten_live_view()
        # Generate indices on the storage device so the gather reads contiguous
        # memory there; the gathered results are then transferred once.
        idx = torch.randint(0, total, (batch_size,), device=self.storage_device)
        gathered = (
            views["obs"][idx],
            views["priv"][idx],
            views["act"][idx],
            views["next_obs"][idx],
            views["next_priv"][idx],
            views["done"][idx],
            views["rew"][idx],
            views["c_kid"][idx],
            views["c_tgt"][idx],
            views["c_tau"][idx],
            views["c_w"][idx],
            views["c_m"][idx],
            views["nc_kid"][idx],
            views["nc_tgt"][idx],
            views["nc_tau"][idx],
            views["nc_w"][idx],
            views["nc_m"][idx],
            views["snip"][idx],
        )
        return tuple(self._move_to_sample_device(t) for t in gathered)

    def mini_batch_generator(self, batch_size: int, num_epochs: int = 1):
        """Yield one full shuffled pass over the populated region, repeated num_epochs times.

        Yields tuples of:
            (obs, priv_obs, actions, next_obs, next_priv_obs, dones, rewards,
             c_key_ids, c_targets, c_taus, c_weights, c_mask,
             nc_key_ids, nc_targets, nc_taus, nc_weights, nc_mask,
             snippets)
        """
        views, total = self._flatten_live_view()
        obs_flat = views["obs"]
        priv_flat = views["priv"]
        act_flat = views["act"]
        done_flat = views["done"]
        rew_flat = views["rew"]
        next_obs_flat = views["next_obs"]
        next_priv_flat = views["next_priv"]
        c_kid_flat = views["c_kid"]
        c_tgt_flat = views["c_tgt"]
        c_tau_flat = views["c_tau"]
        c_w_flat = views["c_w"]
        c_m_flat = views["c_m"]
        nc_kid_flat = views["nc_kid"]
        nc_tgt_flat = views["nc_tgt"]
        nc_tau_flat = views["nc_tau"]
        nc_w_flat = views["nc_w"]
        nc_m_flat = views["nc_m"]
        snip_flat = views["snip"]

        for _ in range(num_epochs):
            perm = torch.randperm(total, device=self.storage_device)
            for start in range(0, total - batch_size + 1, batch_size):
                idx = perm[start: start + batch_size]
                gathered = (
                    obs_flat[idx],
                    priv_flat[idx],
                    act_flat[idx],
                    next_obs_flat[idx],
                    next_priv_flat[idx],
                    done_flat[idx],
                    rew_flat[idx],
                    c_kid_flat[idx],
                    c_tgt_flat[idx],
                    c_tau_flat[idx],
                    c_w_flat[idx],
                    c_m_flat[idx],
                    nc_kid_flat[idx],
                    nc_tgt_flat[idx],
                    nc_tau_flat[idx],
                    nc_w_flat[idx],
                    nc_m_flat[idx],
                    snip_flat[idx],
                )
                yield tuple(self._move_to_sample_device(t) for t in gathered)
