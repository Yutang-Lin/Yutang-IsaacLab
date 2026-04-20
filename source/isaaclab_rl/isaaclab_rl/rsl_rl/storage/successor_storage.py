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
            # Per-env root pose at the transition's sample time. Needed
            # by training-time expert-relabel: a target drawn in the
            # expert's own de-yawed frame must be re-expressed in the
            # stored step's frame before the critic reads it. Without
            # this, the critic's target and priv live in different
            # frames for relabeled rows.
            self.root_pos: torch.Tensor = None        # type: ignore  [N, 3]
            self.root_quat: torch.Tensor = None       # type: ignore  [N, 4]
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

        # Per-transition root pose at sample time. Used by the training
        # expert-relabel path to re-express the expert's world-anchored
        # target into the stored step's de-yawed root frame. Without
        # this, relabeled rows would feed the critic a target in a
        # frame that doesn't match the stored priv.
        self.root_pos = _alloc(T, N, 3)
        self.root_quat = _alloc(T, N, 4)

        # Snippet for discriminator
        self.snippets = _alloc(T, N, snippet_dim)

        self.step = 0
        self._full = False
        self._episode_length: int | None = None
        self._episode_phase_offset: int = 0

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

        if transition.root_pos is None or transition.root_quat is None:
            raise RuntimeError(
                "Transition missing root_pos / root_quat. The rollout path "
                "must populate these from the env's live body pose so the "
                "training expert-relabel can be frame-consistent with the "
                "stored priv."
            )
        self.root_pos[idx].copy_(transition.root_pos, non_blocking=True)
        self.root_quat[idx].copy_(transition.root_quat, non_blocking=True)

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
            root_pos=self.root_pos[:max_t].reshape(total, -1),
            root_quat=self.root_quat[:max_t].reshape(total, -1),
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
        with pinned-memory CPU storage this overlaps with GPU compute. The
        trailing two entries of the returned tuple are ``(t_idx, env_idx)``,
        the raw (time, env) coordinates of each mini-batch row on the
        **storage device** — they stay there so downstream helpers like
        :meth:`gather_next_priv_at` can do zero-copy indexing into the
        time-major buffers.
        """
        views, total = self._flatten_live_view()
        # Generate indices on the storage device so the gather reads contiguous
        # memory there; the gathered results are then transferred once.
        idx = torch.randint(0, total, (batch_size,), device=self.storage_device)
        max_t = self.num_transitions_per_env if self._full else self.step
        t_idx = idx // self.num_envs      # [B]  (on storage device)
        env_idx = idx % self.num_envs
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
            views["root_pos"][idx],
            views["root_quat"][idx],
        )
        moved = tuple(self._move_to_sample_device(t) for t in gathered)
        # ``t_idx`` / ``env_idx`` stay on the storage device so downstream
        # gathers read the circular buffer directly without a double hop.
        return moved + (t_idx, env_idx)

    def set_episode_alignment(self, episode_length: int | None, phase_offset: int = 0) -> None:
        """Inform the storage about a known fixed episode length.

        When every env resets in lock-step every ``episode_length`` env
        steps (as in the sparse-successor env — 500-step timeout, no
        hard terminations), the "safe future anchor" set becomes a
        trivial pattern:

            t_idx ≡ (t_idx mod L) in [phase_offset, L - horizon + phase_offset)

        where ``phase_offset`` is the offset (in buffer slots) of the
        first transition of a fresh episode. Setting this lets
        :meth:`sample_safe_future_anchors` skip a full O(T·N) done-scan
        and sample directly from the safe region.

        Args:
            episode_length: ``L`` in env-steps. ``None`` disables alignment
                and falls back to the general done-scan path.
            phase_offset: buffer slot at which episode 0's first
                transition was written. Usually 0 for a fresh run.
        """
        self._episode_length = int(episode_length) if episode_length is not None else None
        self._episode_phase_offset = int(phase_offset) % self.num_transitions_per_env

    def sample_safe_future_anchors(
        self,
        n: int,
        horizon: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample ``n`` anchor coordinates ``(t_idx, env_idx)`` whose full
        ``horizon`` lookahead window is populated **and** doesn't cross
        any episode boundary.

        Uses the fast aligned-block path when ``set_episode_alignment``
        has declared a fixed ``L`` (all envs reset in lock-step) —
        direct arithmetic on ``t_idx mod L`` suffices. Otherwise falls
        back to a per-(t, env) done-scan that works for general
        termination schedules.

        Returns ``None`` when no safe anchor exists (populated region
        is smaller than ``horizon + 1``, or every window crosses a
        reset). The caller treats ``None`` as "skip this source".
        """
        if not self._full and self.step < horizon + 1:
            return None

        H = int(horizon)
        max_t = self.num_transitions_per_env if self._full else self.step
        N = self.num_transitions_per_env
        n_envs = self.num_envs
        dev = self.storage_device

        L = getattr(self, "_episode_length", None)
        phase = getattr(self, "_episode_phase_offset", 0)

        if L is not None and L > H:
            # Aligned-block path: a slot is safe iff its horizon window
            # ``[t, t + H]`` stays strictly within one episode block.
            # In each L-step block, the last step (position ``L - 1``)
            # carries ``done=1`` for every env (hard timeout). We want
            # no step in the window to land on that boundary, so the
            # safe region is ``(t mod L) <= L - H - 2`` — equivalently
            # ``(t mod L) < L - H - 1``.
            if L - H - 1 <= 0:
                return None
            limit = L - H - 1
            if self._full:
                t_all = torch.arange(N, device=dev)                  # [N]
                within_block = ((t_all - phase).remainder(L)) < limit
                # Circular in-bounds check (distance to seam >= H).
                write_ptr = int(self.step % N)
                dist = (write_ptr - 1 - t_all).remainder(N)
                in_bounds = dist >= H
                safe_t = within_block & in_bounds                    # [N]
            else:
                t_all = torch.arange(max_t, device=dev)
                within_block = ((t_all - phase).remainder(L)) < limit
                in_bounds = (t_all + H) < max_t
                safe_t = within_block & in_bounds                    # [max_t]

            safe_t_idx = safe_t.nonzero(as_tuple=False).squeeze(-1)
            if safe_t_idx.numel() == 0:
                return None
            # Every env shares the same safe-t set (lock-step resets),
            # so sample t and env independently.
            pick_t = safe_t_idx[torch.randint(0, safe_t_idx.numel(), (n,), device=dev)]
            pick_env = torch.randint(0, n_envs, (n,), device=dev)
            if device is not None:
                pick_t = pick_t.to(device)
                pick_env = pick_env.to(device)
            return pick_t, pick_env

        # Fallback: general per-env done-scan. Works for any
        # termination schedule but is O(T · N_env) per call.
        dones_flat = self.dones[:max_t].squeeze(-1)                 # [max_t, N_env]
        cum = dones_flat.cumsum(dim=0)                              # [max_t, N_env]

        if self._full:
            write_ptr = int(self.step % N)
            t_all = torch.arange(N, device=dev)                      # [N]
            dist = (write_ptr - 1 - t_all).remainder(N)              # [N]
            in_bounds_per_t = dist >= H                              # [N]
            frame_end = (t_all + H) % N                              # [N]
            cum_end = cum[frame_end]                                 # [N, N_env]
            cum_start = cum[t_all]                                   # [N, N_env]
            no_reset_per_t = (cum_end - cum_start) <= 0.5            # [N, N_env]
            safe = in_bounds_per_t.unsqueeze(1) & no_reset_per_t     # [N, N_env]
        else:
            t_all = torch.arange(max_t, device=dev)                  # [max_t]
            in_bounds_per_t = (t_all + H) < max_t                    # [max_t]
            frame_end = (t_all + H).clamp(max=max_t - 1)
            cum_end = cum[frame_end]                                 # [max_t, N_env]
            cum_start = cum[t_all]                                   # [max_t, N_env]
            no_reset_per_t = (cum_end - cum_start) <= 0.5            # [max_t, N_env]
            safe = in_bounds_per_t.unsqueeze(1) & no_reset_per_t     # [max_t, N_env]

        flat_safe = safe.reshape(-1)
        num_safe = int(flat_safe.sum().item())
        if num_safe == 0:
            return None
        safe_idx = flat_safe.nonzero(as_tuple=False).squeeze(-1)
        pick = torch.randint(0, num_safe, (n,), device=dev)
        chosen = safe_idx[pick]
        t_idx = chosen // n_envs
        env_idx = chosen % n_envs
        if device is not None:
            t_idx = t_idx.to(device)
            env_idx = env_idx.to(device)
        return t_idx, env_idx

    def gather_next_priv_at(
        self,
        t_idx: torch.Tensor,
        env_idx: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather ``next_privileged_observations`` for a horizon of future
        frames per (t, env) anchor, plus a mask marking frames that stay on
        the same trajectory segment.

        For hindsight relabeling the training loop needs the env's realized
        future over the next ``horizon`` env-steps. We gather
        ``next_priv[t + h − 1, env]`` for h ∈ [0, horizon]. A frame is
        invalid if it is older than the anchor (circular overwrite) or if
        any ``done`` appears in ``(t, t + h)`` (trajectory reset).

        Args:
            t_idx, env_idx: [B] anchor coordinates (on storage device).
            horizon: H — how many future steps to gather.

        Returns:
            priv_window:  [B, H+1, priv_dim] on the sample device.
                          Index 0 = ``next_priv[t, env]`` (one step after
                          the anchor's action, i.e. the env's realized
                          next state). Index h = ``next_priv[t + h, env]``.
            valid_mask:   [B, H+1] bool on the sample device. True where
                          the frame is usable (no reset crossing + within
                          the populated circular region).
        """
        device = self.storage_device
        max_t = self.num_transitions_per_env if self._full else self.step
        N = self.num_transitions_per_env
        H = int(horizon)
        B = t_idx.shape[0]

        offsets = torch.arange(H + 1, device=device).unsqueeze(0)   # [1, H+1]

        if self._full:
            # Circular buffer: the write pointer is the seam between
            # newest and oldest. "Forward in time" wraps, but crossing
            # the seam lands on much-older data. Mark all frames that
            # cross the seam as invalid. ``write_ptr`` is where the next
            # transition will be written; slot ``write_ptr - 1`` is the
            # newest valid transition.
            write_ptr = int(self.step % N)
            # Distance forward (in ring steps) from anchor to seam:
            #   dist = (write_ptr - 1 - t_idx) mod N
            # Frames with offset > dist cross the seam.
            dist_to_seam = (write_ptr - 1 - t_idx.long()).remainder(N)   # [B]
            # Ring-advanced frame indices (wrapped).
            frame = (t_idx.unsqueeze(1) + offsets).remainder(N)          # [B, H+1]
            in_bounds = offsets <= dist_to_seam.unsqueeze(1)             # [B, H+1]
        else:
            # Linear write: simple [0, max_t) bounds check.
            frame = t_idx.unsqueeze(1) + offsets                         # [B, H+1]
            in_bounds = frame < max_t                                    # [B, H+1]

        # Reset-crossing mask: dones in the (t, t+h] half-open window
        # must all be zero. We compute per-env cumulative dones on the
        # populated slice, then probe at frame indices. Indices are
        # ring-wrapped when ``_full``; anchor gather uses the same
        # wrapping.
        dones_flat = self.dones[:max_t].squeeze(-1)                  # [max_t, N]
        cum = dones_flat.cumsum(dim=0)                               # [max_t, N]
        # Clamp frame to a valid storage index for the gather. Out-of-
        # bounds rows (already masked by ``in_bounds``) get zero-ish
        # values but never influence the final ``valid`` result.
        frame_c = frame.clamp(max=max_t - 1)
        env_exp = env_idx.unsqueeze(1).expand(B, H + 1)
        cum_at_frame = cum[frame_c, env_exp]                         # [B, H+1]
        cum_at_anchor = cum[t_idx, env_idx].unsqueeze(1)             # [B, 1]
        # When _full: the ring semantics mean cum can *decrease* going
        # forward if we cross the seam, but ``in_bounds`` already masks
        # those rows out. Within a valid run, the anchor's cum is <= the
        # frame's cum iff no reset was crossed.
        no_reset = (cum_at_frame - cum_at_anchor) <= 0.5
        # Anchor (h=0) is always "no reset" by definition.
        no_reset[:, 0] = True
        valid = in_bounds & no_reset                                 # [B, H+1]

        priv = self.next_privileged_observations[:max_t]             # [T, N, D]
        priv_window = priv[frame_c, env_exp]                         # [B, H+1, D]

        priv_window = self._move_to_sample_device(priv_window)
        valid = self._move_to_sample_device(valid)
        return priv_window, valid

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
        root_pos_flat = views["root_pos"]
        root_quat_flat = views["root_quat"]

        for _ in range(num_epochs):
            perm = torch.randperm(total, device=self.storage_device)
            for start in range(0, total - batch_size + 1, batch_size):
                idx = perm[start: start + batch_size]
                t_idx = idx // self.num_envs
                env_idx = idx % self.num_envs
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
                    root_pos_flat[idx],
                    root_quat_flat[idx],
                )
                moved = tuple(self._move_to_sample_device(t) for t in gathered)
                # ``t_idx`` / ``env_idx`` stay on the storage device so
                # downstream gathers hit the circular buffer directly.
                yield moved + (t_idx, env_idx)
