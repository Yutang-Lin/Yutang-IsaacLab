# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from isaaclab_rl.rsl_rl.modules.sparse_successor_policy import SparseSuccessorPolicy
from isaaclab_rl.rsl_rl.storage.successor_storage import SuccessorStorage
from isaaclab_rl.rsl_rl.storage.expert_motion_buffer import ExpertMotionBuffer
from isaaclab_rl.rsl_rl.utils import reduce_gradients


class SparseSuccessor:
    """Kernelized sparse-constraint successor tracking with GAN-style discriminator.

    Off-policy SAC-like algorithm that trains:
      1. Twin successor critics (per-query tracking ability)
      2. Style discriminator (expert vs policy)
      3. Twin style critics (naturalness value)
      4. Actor (maximize tracking + style)

    The constraint set is sampled from reference motions automatically during rollout.
    """

    policy: SparseSuccessorPolicy

    def __init__(
        self,
        policy: SparseSuccessorPolicy,
        # Learning rates
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_query: float = 3e-4,
        lr_disc: float = 1e-4,
        # RL hyperparameters
        gamma: float = 0.99,
        # Separate discount for the style critic. BFM-Zero uses 0.98 across the
        # board; we keep γ=0.99 for the successor/aux critics (the satisfaction
        # kernel is bounded in [0, 1], so a slightly higher discount is safe)
        # but drop the *style* discount to 0.98 because the log-odds style
        # reward ``log(D) − log(1−D)`` saturates at ±16 and a geometric
        # fixed point of ±1600 at γ=0.99 drives q_style to diverge whenever
        # the disc briefly wins. γ=0.98 halves the floor to ±800.
        gamma_style: float | None = None,
        target_tau: float = 0.005,
        lambda_style: float = 0.1,
        lambda_aux: float = 1.0,
        # Pessimism penalties (BFM-style). With twin critics, uncertainty is
        # proxied by |Q1 - Q2|; ``Q_pessimistic = mean(Q) - penalty * uncertainty``.
        # Zero reduces to the standard ``min(Q1, Q2)`` double-Q approximation —
        # setting these non-zero is strictly more conservative and stabilises
        # off-policy updates when the twin critics disagree.
        critic_pessimism_penalty: float = 0.5,
        actor_pessimism_penalty: float = 0.5,
        # Successor critic hyperparameters
        sigma_time: float = 2.0,
        beta: float | list[float] = 0.1,
        tau_max: int = 20,
        # Constraint sampling
        n_constraints_min: int = 1,
        n_constraints_max: int = 8,
        weight_range: tuple[float, float] = (0.5, 1.5),
        target_noise_std: float = 0.02,
        constraint_dropout_prob: float = 0.1,
        # Chunk-level rollout: hold the constraint-set content fixed until
        # its longest τ has counted all the way down (``max τ_i == 0``), at
        # which point a fresh C is sampled. ``constraint_horizon`` is an
        # *upper cap* on the dwell time so degenerate τ distributions can't
        # stall the actor on a dead set. Set to 0 to disable the cap.
        constraint_horizon: int = 0,
        # Fraction of rollout chunks that draw their fresh constraint set from
        # the expert motion buffer's keypoint positions (BFM-style expert
        # rollout). 0 disables entirely — the replay buffer then contains no
        # expert-anchored rollout trajectories.
        #
        # **Deprecated**: kept for backward-compatibility only. The rollout
        # source mixture now uses the 3-way ``rollout_{live,replay,expert}_fraction``
        # knobs below. If any of those are set, ``expert_chunk_fraction`` is
        # ignored.
        expert_chunk_fraction: float = 0.15,
        # 3-way per-env source mixture for fresh rollout chunks. Normalised
        # to sum to 1 at runtime; missing sources (e.g. no expert buffer,
        # empty replay) fold their mass into the remaining ones.
        #   - live:    env's current priv (single-frame, phase-1 leftover)
        #   - replay:  per-atom future-grounded from replay's realized future
        #   - expert:  per-atom future-grounded from expert motion buffer
        rollout_live_fraction: float = 0.2,
        rollout_replay_fraction: float = 0.3,
        rollout_expert_fraction: float = 0.5,
        # Training-time C-space relabeling ratios.
        relabel_ratio_stored: float = 0.4,
        relabel_ratio_hindsight: float = 0.3,
        relabel_ratio_expert: float = 0.3,
        # Snippet config
        snippet_length: int = 8,
        # Training
        num_learning_epochs: int = 1,
        mini_batch_size: int = 512,
        max_grad_norm: float = 1.0,
        updates_per_step: int = 1,
        grad_penalty_weight: float = 10.0,
        # Off-policy replay capacity per env. Decouples replay memory size from
        # the rollout length configured in the runner (``num_steps_per_env``).
        # When ``None``, replay equals rollout length (pure on-policy behaviour).
        replay_capacity_per_env: int | None = None,
        # Where the replay tensors live. Default ``cpu`` matches BFM-Zero's
        # ``buffer_device`` and lets us hold a large replay (e.g. 40s per env
        # × 4096 envs ≈ 8M transitions ≈ ~90 GB of floats) without consuming
        # GPU memory. Sampled batches are moved to the training device with
        # pinned-memory non-blocking transfers.
        replay_device: str | None = None,
        # Number of *total* env transitions collected with random actions
        # before training updates begin. Mirrors BFM-Zero's ``num_seed_steps``.
        # These transitions are still written to the replay buffer so the
        # critic cold-start sees a reasonable state-action distribution when
        # learning kicks in. Counted in env transitions, not iterations.
        num_seed_steps: int = 0,
        # Sparse-constraint tracking eval — runner-side frequency is in
        # env steps. Set eval_interval_env_steps=0 to disable.
        eval_interval_env_steps: int = 0,
        eval_num_samples_per_bucket: int = 512,
        # Number of gradient updates per training iteration (SAC/TD3-style).
        # When ``None``, fall back to the legacy behaviour of one full shuffled
        # pass × num_learning_epochs through the buffer. With replay enabled,
        # prefer setting this explicitly (e.g. BFM uses 16 updates per batch).
        num_updates_per_iter: int | None = None,
        # Expert dataset — if provided, the expert motion buffer is loaded
        # here so style_feature_dim is known before init_storage runs.
        expert_dataset_path: str | None = None,
        expert_dataset_device: str | None = None,
        # Misc
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        # Compatibility (unused)
        **kwargs,
    ):
        if kwargs:
            print(f"SparseSuccessor.__init__ got unexpected arguments: {list(kwargs.keys())}")

        self.device = device
        self.policy = policy
        self.policy.to(self.device)

        # Multi-GPU: standard DDP semantics — shared params, all-reduced gradients across ranks and nodes.
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Sync gradients whenever we have more than one rank
        self._sync_grads = self.is_multi_gpu and self.gpu_world_size > 1

        # Hyperparameters
        self.gamma = gamma
        self.gamma_style = float(gamma_style) if gamma_style is not None else float(gamma)
        self.target_tau = target_tau
        self.lambda_style = lambda_style
        self.lambda_aux = lambda_aux
        self.critic_pessimism_penalty = float(critic_pessimism_penalty)
        self.actor_pessimism_penalty = float(actor_pessimism_penalty)
        self.sigma_time = sigma_time
        # beta may be scalar or a per-keypoint list. Keep a tensor on device
        # shaped [num_keypoints] so satisfaction() can gather per query.
        if isinstance(beta, (list, tuple)):
            if len(beta) != self.policy.num_keypoints:
                raise ValueError(
                    f"Per-keypoint beta list has len {len(beta)} but policy has"
                    f" num_keypoints={self.policy.num_keypoints}"
                )
            self.beta = torch.as_tensor(list(beta), dtype=torch.float32, device=device)
        else:
            self.beta = torch.full((self.policy.num_keypoints,), float(beta),
                                   dtype=torch.float32, device=device)
        self.tau_max = tau_max
        self.n_constraints_min = n_constraints_min
        self.n_constraints_max = n_constraints_max
        self.weight_range = weight_range
        self.target_noise_std = target_noise_std
        self.constraint_dropout_prob = constraint_dropout_prob
        self.constraint_horizon = int(constraint_horizon)
        self.expert_chunk_fraction = float(expert_chunk_fraction)
        self.rollout_live_fraction = float(rollout_live_fraction)
        self.rollout_replay_fraction = float(rollout_replay_fraction)
        self.rollout_expert_fraction = float(rollout_expert_fraction)
        # Normalise relabel ratios so they sum to 1.
        total_r = float(relabel_ratio_stored + relabel_ratio_hindsight + relabel_ratio_expert)
        if total_r <= 0.0:
            self.relabel_ratio_stored = 1.0
            self.relabel_ratio_hindsight = 0.0
            self.relabel_ratio_expert = 0.0
        else:
            self.relabel_ratio_stored = float(relabel_ratio_stored) / total_r
            self.relabel_ratio_hindsight = float(relabel_ratio_hindsight) / total_r
            self.relabel_ratio_expert = float(relabel_ratio_expert) / total_r
        self.snippet_length = snippet_length
        self.num_learning_epochs = num_learning_epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm
        self.updates_per_step = updates_per_step
        self.grad_penalty_weight = grad_penalty_weight
        self.replay_capacity_per_env = replay_capacity_per_env
        self.replay_device = replay_device if replay_device is not None else self.device
        self.num_seed_steps = int(num_seed_steps)
        self.eval_interval_env_steps = int(eval_interval_env_steps)
        self.eval_num_samples_per_bucket = int(eval_num_samples_per_bucket)
        self.num_updates_per_iter = num_updates_per_iter
        self.learning_rate = lr_actor

        # Populated via set_expert_buffer() or below. Until then,
        # discriminator training is skipped and the actor never sees a style bonus.
        self.expert_buffer: ExpertMotionBuffer | None = None
        self.style_feature_dim: int | None = None

        # Diagnostic accumulator — reset at the start of every update() call
        # and flushed into the returned loss_dict. See ``update()`` for the
        # full list of tracked scalars (Loss/*, Scale/*, Critic/*, QueryTau/*,
        # QueryKeypoint/*, Disc/*, Aux/*, Action/*).
        self._diag: dict[str, float] = {}
        self._diag_count: int = 0

        # τ buckets used for per-query diagnostics. List of inclusive
        # (lo, hi) ranges covering [1, tau_max]. Buckets stay fixed across
        # configurations; if ``tau_max`` changes, missing buckets will just
        # report zero occupancy.
        self._tau_buckets: list[tuple[int, int]] = [
            (1, 3), (4, 6), (7, 10), (11, 15), (16, 20),
            (21, 30), (31, 40), (41, 50),
        ]
        if expert_dataset_path is not None:
            buf_device = expert_dataset_device if expert_dataset_device is not None else "cpu"
            buf = ExpertMotionBuffer(
                dataset_path=expert_dataset_path,
                snippet_length=snippet_length,
                device=buf_device,
            )
            # set_expert_buffer checks keypoint/snippet consistency.
            self.set_expert_buffer(buf)

        # Optimizers. The ``ConstraintSetEncoder`` now does pure weighted
        # linear aggregation of atomic embeddings — it has no learnable
        # parameters beyond the ones owned by ``query_encoder``. So we
        # optimize the query encoder directly; no separate ``opt_constraint``.
        self.opt_actor = optim.Adam(policy.actor.parameters(), lr=lr_actor)
        self.opt_query = optim.Adam(policy.query_encoder.parameters(), lr=lr_query)
        self.opt_U1 = optim.Adam(policy.successor_critic_1.parameters(), lr=lr_critic)
        self.opt_U2 = optim.Adam(policy.successor_critic_2.parameters(), lr=lr_critic)
        self.opt_disc = optim.Adam(policy.style_discriminator.parameters(), lr=lr_disc)
        self.opt_QS1 = optim.Adam(policy.style_critic_1.parameters(), lr=lr_critic)
        self.opt_QS2 = optim.Adam(policy.style_critic_2.parameters(), lr=lr_critic)
        self.opt_QA1 = optim.Adam(policy.aux_critic_1.parameters(), lr=lr_critic)
        self.opt_QA2 = optim.Adam(policy.aux_critic_2.parameters(), lr=lr_critic)

        # Combined optimizer for BaseRunner compatibility (save/load)
        self.optimizer = self.opt_actor
        self.critic_optimizer = self.opt_U1

        # Storage (created in init_storage)
        self.storage: SuccessorStorage = None  # type: ignore
        self.transition = SuccessorStorage.Transition()

        # Vectorized snippet ring buffer — allocated in init_storage once we
        # know (num_envs, style_feature_dim). Holds the last ``snippet_length``
        # style feature frames per env, ready to be flattened for the discriminator.
        self._snippet_ring: torch.Tensor | None = None          # [num_envs, snippet_length, style_dim]
        self._snippet_write_ptr: int = 0
        self._snippet_fill: torch.Tensor | None = None          # [num_envs] in [0, snippet_length]

        # Per-env constraint state
        self._env_constraints: dict[str, torch.Tensor] | None = None

        # RND compatibility
        self.rnd = None
        self.rnd_optimizer = None

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: list[int],
        critic_obs_shape: list[int],
        actions_shape: list[int],
        meta_tensors=None,
    ):
        # The per-frame style feature dim must be set before init_storage.
        # If the runner forgot to call set_expert_buffer first, fall back to
        # actor obs dim so the snippet tensors still allocate — but training
        # the discriminator will be skipped.
        if self.style_feature_dim is None:
            self.style_feature_dim = actor_obs_shape[0]
            print(
                f"[WARN] SparseSuccessor.init_storage called without an expert buffer;"
                f" defaulting style_feature_dim={self.style_feature_dim}."
                " Discriminator training will be disabled."
            )

        snippet_dim = self.style_feature_dim * self.snippet_length
        # Replay capacity: ``replay_capacity_per_env`` if explicitly configured
        # (off-policy), otherwise the runner's rollout length (legacy on-policy
        # semantics). The SuccessorStorage itself is already a circular buffer;
        # making its capacity larger than the rollout length is how we turn it
        # into a true replay buffer.
        capacity = self.replay_capacity_per_env or num_transitions_per_env
        if capacity < num_transitions_per_env:
            raise ValueError(
                f"replay_capacity_per_env={capacity} must be >= num_transitions_per_env="
                f"{num_transitions_per_env} (the runner rollout length); otherwise a single"
                " rollout overruns the buffer before any updates run."
            )
        self._replay_capacity_per_env = capacity
        self.storage = SuccessorStorage(
            num_envs=num_envs,
            num_transitions_per_env=capacity,
            obs_shape=actor_obs_shape,
            privileged_obs_shape=critic_obs_shape,
            actions_shape=actions_shape,
            max_constraints=self.policy.max_constraints,
            target_dim=self.policy.target_dim,
            snippet_dim=snippet_dim,
            storage_device=self.replay_device,
            sample_device=self.device,
        )

        # Vectorized snippet ring buffer on the algorithm device. We hold the
        # most recent ``snippet_length`` frames per env; ``_snippet_write_ptr``
        # marks the next slot. When rolled out, index order becomes
        # [ (ptr), (ptr+1), ..., (ptr - 1 mod L) ] — ascending time order.
        self._snippet_ring = torch.zeros(
            num_envs, self.snippet_length, self.style_feature_dim,
            device=self.device,
        )
        self._snippet_write_ptr = 0
        self._snippet_fill = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Initialize constraint state for all envs
        self._init_constraints(num_envs)

    def set_expert_buffer(self, buffer: ExpertMotionBuffer) -> None:
        """Attach an expert motion buffer, enabling style-discriminator training.

        Must be called before ``init_storage`` so that the snippet ring buffer
        is allocated with the correct ``style_feature_dim``.
        """
        self.expert_buffer = buffer
        self.style_feature_dim = buffer.style_feature_dim
        # Sanity-check keypoint layout. The policy uses num_keypoints from
        # its query encoder; the expert buffer must expose the same ordering.
        if buffer.num_keypoints != self.policy.num_keypoints:
            raise ValueError(
                f"Expert buffer has {buffer.num_keypoints} keypoints but the policy"
                f" was constructed with num_keypoints={self.policy.num_keypoints}."
            )
        if buffer.snippet_length != self.snippet_length:
            raise ValueError(
                f"Expert buffer snippet_length={buffer.snippet_length} does not match"
                f" algorithm snippet_length={self.snippet_length}."
            )

    def _init_constraints(self, num_envs: int):
        M = self.policy.max_constraints
        td = self.policy.target_dim
        self._env_constraints = {
            "keypoint_ids": torch.zeros(num_envs, M, dtype=torch.long, device=self.device),
            "targets": torch.zeros(num_envs, M, td, device=self.device),
            "taus": torch.zeros(num_envs, M, device=self.device),
            "weights": torch.zeros(num_envs, M, device=self.device),
            "mask": torch.zeros(num_envs, M, device=self.device),
        }
        # Remaining chunk lifetime per env (in env steps). When it hits zero,
        # a new constraint set is sampled for that env. Initialised to zero so
        # the very first call samples fresh constraints for every env.
        self._chunk_steps_left = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def _replace_constraints_for_envs(
        self,
        env_mask: torch.Tensor,
        new_constraints: dict[str, torch.Tensor],
    ) -> None:
        """Copy-in ``new_constraints`` only for envs where ``env_mask`` is True."""
        if not env_mask.any():
            return
        for key, new_val in new_constraints.items():
            dst = self._env_constraints[key]
            dst[env_mask] = new_val[env_mask]

    @torch.no_grad()
    def _advance_chunk(self, next_priv_obs: torch.Tensor, reset_mask: torch.Tensor | None) -> None:
        """Per-step chunk bookkeeping. Called from ``set_next_obs``.

        Per env we hold the sampled set ``(k, ξ, w)`` fixed and let every
        query's τ count down one step per env-step. A query whose τ reaches
        **0** is *expired*: its mask entry is cleared so it no longer
        contributes to the pooled ``z_C`` or to the satisfaction kernel. The
        whole set is resampled as soon as every active query has expired
        (i.e. no queries remain with ``mask > 0``), or immediately on env
        reset, or when the optional ``constraint_horizon`` cap trips.

        This replaces the previous fixed-horizon behaviour — chunks now
        live exactly as long as the latest-firing query asked for, which
        lines up the constraint lifetime with the satisfaction kernel's
        horizon expectations.
        """
        num_envs = next_priv_obs.shape[0]

        # --- 1) Decrement τ for every active query (mask>0). τ clamps at 0
        #        on the way down; a 0-τ query counts as "expired" and is
        #        masked out just below.
        taus = self._env_constraints["taus"]
        mask = self._env_constraints["mask"]
        valid = mask > 0
        if valid.any():
            new_taus = taus.clone()
            new_taus[valid] = (new_taus[valid] - 1).clamp(min=0.0)
            # Expire queries whose τ just hit 0.
            expired = valid & (new_taus == 0)
            if expired.any():
                mask = mask.clone()
                mask[expired] = 0.0
                self._env_constraints["mask"] = mask
            taus = new_taus
            self._env_constraints["taus"] = taus

        # Optional global cap — prevents a pathological set with very large
        # τ from stalling the actor indefinitely.
        if self.constraint_horizon > 0:
            self._chunk_steps_left = (self._chunk_steps_left - 1).clamp(min=0)

        # --- 2) Identify envs that need a fresh set: (a) every query has
        #        expired, (b) global cap elapsed, (c) env reset.
        active_any = (mask > 0).any(dim=-1)           # [N]
        needs_new = ~active_any
        if self.constraint_horizon > 0:
            needs_new = needs_new | (self._chunk_steps_left == 0)
        if reset_mask is not None and reset_mask.any():
            needs_new = needs_new | reset_mask.to(self.device, dtype=torch.bool)

        if needs_new.any():
            # Per-env 3-way mixture for fresh rollout chunks:
            #   - live/self (single-frame, phase-1 leftover): ``rollout_live_fraction``
            #   - replay-future (per-atom, real future from storage):
            #     ``rollout_replay_fraction``
            #   - expert-future (per-atom, from the expert buffer):
            #     ``rollout_expert_fraction``
            # Goal: push the rollout-task distribution toward
            # future-grounded sparse tracking. Live/self stays the smallest
            # share so the actor can't collapse into "hold current pose."
            n_new = int(needs_new.sum().item())

            # Initialise with a full live/self draw — we'll overlay replay
            # and expert rows below. ``random_C`` has ``num_envs`` rows;
            # we only mutate the rows under ``needs_new``.
            random_C = self.sample_constraint_set_vectorized(next_priv_obs, num_envs)

            # Decide per-new-env source. Replay-future needs at least
            # ``tau_max + 1`` steps of populated storage so the future
            # window has room; until then its probability folds into the
            # live/expert sources.
            p_live = float(self.rollout_live_fraction)
            replay_ready = (
                self.storage is not None
                and (
                    self.storage._full
                    or self.storage.step >= self.tau_max + 1
                )
            )
            p_replay = float(self.rollout_replay_fraction) if replay_ready else 0.0
            p_expert = float(self.rollout_expert_fraction) if self.expert_buffer is not None else 0.0
            total_p = p_live + p_replay + p_expert
            if total_p <= 0.0:
                p_live, p_replay, p_expert = 1.0, 0.0, 0.0
            else:
                p_live, p_replay, p_expert = p_live / total_p, p_replay / total_p, p_expert / total_p
            probs = torch.tensor([p_live, p_replay, p_expert], device=self.device)
            source = torch.multinomial(probs, n_new, replacement=True)  # [n_new] in {0,1,2}

            new_env_ids = needs_new.nonzero(as_tuple=True)[0]

            # --- Replay-future source ---
            replay_mask_of_new = source == 1
            if replay_mask_of_new.any():
                n_replay = int(replay_mask_of_new.sum().item())
                sdev = self.storage.storage_device
                max_t = (
                    self.storage.num_transitions_per_env
                    if self.storage._full else self.storage.step
                )
                # Anchor at a random (t, env) coordinate in the replay.
                # Constrain t so t + tau_max fits in the populated region.
                max_anchor = max(max_t - self.tau_max, 1)
                t_anchor = torch.randint(0, max_anchor, (n_replay,), device=sdev)
                env_anchor = torch.randint(0, self.storage.num_envs, (n_replay,), device=sdev)
                priv_window, valid = self.storage.gather_next_priv_at(
                    t_anchor, env_anchor, horizon=self.tau_max,
                )
                priv_window = priv_window.to(self.device)
                valid = valid.to(self.device)
                nk = self.policy.num_keypoints
                td = self.policy.target_dim
                priv_dim = priv_window.shape[-1]
                kp_offset = priv_dim - nk * td
                kp_window = priv_window[..., kp_offset:].reshape(
                    priv_window.shape[0], priv_window.shape[1], nk, td,
                )                                                         # [n_replay, H+1, K, 3]
                replay_C = self._sample_constraints_from_keypoint_future(
                    kp_window, valid_atom_mask=valid,
                )
                replay_env_ids = new_env_ids[replay_mask_of_new]
                for key, rv in replay_C.items():
                    random_C[key][replay_env_ids] = rv

            # --- Expert-future source ---
            expert_mask_of_new = source == 2
            if expert_mask_of_new.any() and self.expert_buffer is not None:
                n_expert = int(expert_mask_of_new.sum().item())
                expert_batch = self.expert_buffer.sample_with_future_window(
                    n_expert, horizon=self.tau_max,
                )
                expert_window = expert_batch["kp_window"].to(self.device)  # [E, H+1, K, 3]
                expert_C = self._sample_constraints_from_keypoint_future(expert_window)
                expert_env_ids = new_env_ids[expert_mask_of_new]
                for key, ev in expert_C.items():
                    random_C[key][expert_env_ids] = ev

            # Record source mix for diagnostics.
            self._diag_rollout_source_live = float((source == 0).float().mean().item())
            self._diag_rollout_source_replay = float((source == 1).float().mean().item())
            self._diag_rollout_source_expert = float((source == 2).float().mean().item())

            self._replace_constraints_for_envs(needs_new, random_C)

            # Reset the optional global cap. Pick ``constraint_horizon`` if
            # configured, otherwise fall back to the longest sampled τ so
            # the counter stays a meaningful upper bound.
            if self.constraint_horizon > 0:
                self._chunk_steps_left[needs_new] = self.constraint_horizon
            else:
                new_mask = self._env_constraints["mask"][needs_new] > 0
                new_taus = self._env_constraints["taus"][needs_new]
                effective_taus = new_taus * new_mask.float()
                max_tau_per_env = effective_taus.amax(dim=-1).long()
                self._chunk_steps_left[needs_new] = max_tau_per_env

        # Record a small diagnostic: fraction of envs entering a fresh chunk.
        if hasattr(self, "_diag"):
            self._diag_rollout_fresh_frac = float(needs_new.float().mean().item())

    # ------------------------------------------------------------------
    # Constraint sampling from privileged state
    # ------------------------------------------------------------------

    def _extract_keypoint_value(self, priv_state: torch.Tensor, keypoint_id: int) -> torch.Tensor:
        """Extract keypoint position from privileged state.

        Assumes privileged state contains keypoint positions packed as:
        [other_data..., kp0_x, kp0_y, kp0_z, kp1_x, kp1_y, kp1_z, ...]
        at the end of the observation vector. The environment must provide this.

        For now: keypoint_id * target_dim : (keypoint_id + 1) * target_dim from the end.
        """
        td = self.policy.target_dim
        nk = self.policy.num_keypoints
        offset = priv_state.shape[-1] - nk * td
        start = offset + keypoint_id * td
        end = start + td
        return priv_state[..., start:end]

    def sample_constraint_set(
        self,
        priv_state: torch.Tensor,
        num_envs: int,
    ) -> dict[str, torch.Tensor]:
        """Sample a random constraint set from the current privileged state (used as proxy for reference motion).

        Args:
            priv_state: [num_envs, priv_dim] current privileged observation (contains keypoint positions)
            num_envs: number of environments

        Returns:
            dict with keypoint_ids, targets, taus, weights, mask tensors all [num_envs, M]
        """
        M = self.policy.max_constraints
        td = self.policy.target_dim
        nk = self.policy.num_keypoints

        keypoint_ids = torch.zeros(num_envs, M, dtype=torch.long, device=self.device)
        targets = torch.zeros(num_envs, M, td, device=self.device)
        taus = torch.zeros(num_envs, M, device=self.device)
        weights = torch.zeros(num_envs, M, device=self.device)
        mask = torch.zeros(num_envs, M, device=self.device)

        for env_i in range(num_envs):
            n = random.randint(self.n_constraints_min, min(self.n_constraints_max, M))
            for j in range(n):
                k = random.randint(0, nk - 1)
                tau = random.randint(1, self.tau_max)
                xi = self._extract_keypoint_value(priv_state[env_i], k)
                # add small noise to target
                xi = xi + torch.randn_like(xi) * self.target_noise_std

                w_min, w_max = self.weight_range
                w = random.uniform(w_min, w_max)

                # random dropout
                if random.random() < self.constraint_dropout_prob:
                    continue

                keypoint_ids[env_i, j] = k
                targets[env_i, j] = xi
                taus[env_i, j] = tau
                weights[env_i, j] = w
                mask[env_i, j] = 1.0

        return {
            "keypoint_ids": keypoint_ids,
            "targets": targets,
            "taus": taus,
            "weights": weights,
            "mask": mask,
        }

    @torch.no_grad()
    def _relabel_constraint_sets(
        self,
        stored: dict[str, torch.Tensor],
        next_priv: torch.Tensor,
        t_idx: torch.Tensor | None = None,
        env_idx: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        """Build a C-space-relabeled constraint set for a training mini-batch.

        For each element of the batch, pick one of three sources according to
        ``relabel_ratio_{stored, hindsight, expert}``:

        - **stored**: use the constraint set that was actually stored with the
          transition (the same C used at rollout time for this (s, a, s') tuple).
        - **hindsight** (per-atom future-grounded): gather the env's realized
          future over the next ``tau_max`` steps from storage, then for each
          atomic query independently sample ``τ_i ∈ [1, tau_max]`` and take
          the target from the corresponding future frame. Atoms whose ``τ_i``
          lands on a reset-crossed frame are masked out.
        - **expert** (per-atom future-grounded): same per-atom semantics but
          anchored in the expert buffer — each atom draws from
          ``expert_kp_window[b, τ_i, k_i]``.

        Returns the relabeled constraint dict (same shape as ``stored``) plus
        counters for diagnostics.
        """
        B = stored["keypoint_ids"].shape[0]
        M = self.policy.max_constraints
        td = self.policy.target_dim
        nk = self.policy.num_keypoints

        # Decide the source per-element. categorical with 3 buckets.
        probs = torch.tensor(
            [self.relabel_ratio_stored, self.relabel_ratio_hindsight, self.relabel_ratio_expert],
            device=self.device, dtype=torch.float32,
        )
        # torch.multinomial needs a non-zero distribution; if expert is
        # unavailable, fold its mass into stored.
        if self.expert_buffer is None and probs[2] > 0.0:
            probs[0] = probs[0] + probs[2]
            probs[2] = 0.0
        # If the storage can't give us (t, env) coordinates, disable hindsight
        # (it would fall back to current-frame only, which is what we're
        # trying to move away from).
        if (t_idx is None or env_idx is None) and probs[1] > 0.0:
            probs[0] = probs[0] + probs[1]
            probs[1] = 0.0
        src = torch.multinomial(probs, B, replacement=True)  # [B] in {0,1,2}

        stored_mask = (src == 0)
        hind_mask = (src == 1)
        expert_mask = (src == 2)

        # Start from stored — we overlay hindsight / expert per src.
        relabeled = {k: v.clone() for k, v in stored.items()}

        # --- Hindsight source (per-atom future-grounded) ---
        if hind_mask.any() and t_idx is not None and env_idx is not None:
            hind_count = int(hind_mask.sum().item())
            idxs = hind_mask.nonzero(as_tuple=True)[0]
            # Gather the realized future priv window for these rows from
            # replay. ``priv_window[:, 0]`` = next_priv at the anchor step;
            # ``priv_window[:, h]`` = next_priv h steps after the anchor.
            priv_window, valid = self.storage.gather_next_priv_at(
                t_idx[idxs.to(t_idx.device)],
                env_idx[idxs.to(env_idx.device)],
                horizon=self.tau_max,
            )
            priv_window = priv_window.to(self.device)
            valid = valid.to(self.device)
            # Extract keypoint positions from the priv tail per frame.
            priv_dim = priv_window.shape[-1]
            kp_offset = priv_dim - nk * td
            kp_window = priv_window[..., kp_offset:].reshape(
                priv_window.shape[0], priv_window.shape[1], nk, td,
            )                                                               # [E, H+1, K, 3]
            hind_C = self._sample_constraints_from_keypoint_future(
                kp_window, valid_atom_mask=valid,
            )
            for key, val in hind_C.items():
                relabeled[key][idxs] = val

        # --- Expert source (per-atom future-grounded) ---
        if expert_mask.any() and self.expert_buffer is not None:
            expert_count = int(expert_mask.sum().item())
            expert_batch = self.expert_buffer.sample_with_future_window(
                expert_count, horizon=self.tau_max,
            )
            expert_window = expert_batch["kp_window"].to(self.device)        # [E, H+1, K, 3]
            expert_C = self._sample_constraints_from_keypoint_future(expert_window)
            idxs = expert_mask.nonzero(as_tuple=True)[0]
            for key, val in expert_C.items():
                relabeled[key][idxs] = val

        counts = {
            "stored": int(stored_mask.sum().item()),
            "hindsight": int(hind_mask.sum().item()),
            "expert": int(expert_mask.sum().item()),
        }
        return relabeled, counts

    def _sample_constraints_from_keypoint_future(
        self,
        kp_window: torch.Tensor,
        valid_atom_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-atom future-grounded constraint sampling.

        For each element b in the batch:
          - for each atomic slot i in ``[0, M)``:
              - sample ``k_i ∈ [0, K)``
              - sample ``τ_i ∈ [1, H]``
              - target ``ξ_i = kp_window[b, τ_i, k_i]``
          - ``n_active`` atoms get ``mask=1``, the rest are padded.

        This is the correct semantics for a sparse multi-time tracking
        query: one anchor time ``t``, many atoms with independent ``τ_i``,
        each atom pulling its target from the **corresponding future frame**
        ``t + τ_i`` rather than from a shared single frame.

        Args:
            kp_window: [B, H+1, K, 3] — index 0 is the anchor, index h>=1 is
                the keypoint position at ``t + h``. ``H`` must be >= ``tau_max``.
            valid_atom_mask: optional [B, H+1] boolean mask. When provided,
                any atom whose sampled ``τ_i`` lands on ``valid_atom_mask[b, τ_i] = 0``
                (e.g. the frame crossed a reset in replay) is masked out.

        Returns:
            dict of padded [B, M, *] constraint set tensors.
        """
        B, Hp1, K, td = kp_window.shape
        H = Hp1 - 1
        M = self.policy.max_constraints
        assert K == self.policy.num_keypoints, (
            f"kp_window has {K} keypoints but policy expects {self.policy.num_keypoints}"
        )
        assert td == self.policy.target_dim
        assert H >= 1, "future window must have at least one lookahead frame"

        max_tau = min(self.tau_max, H)

        # Number of active atoms per sample.
        n_per = torch.randint(
            self.n_constraints_min, min(self.n_constraints_max, M) + 1,
            (B,), device=self.device,
        )
        arange = torch.arange(M, device=self.device).unsqueeze(0).expand(B, -1)
        mask = (arange < n_per.unsqueeze(1)).float()

        # Per-atom keypoint + τ. τ is sampled independently per atom.
        keypoint_ids = torch.randint(0, K, (B, M), device=self.device)
        taus = torch.randint(1, max_tau + 1, (B, M), device=self.device).float()

        w_min, w_max = self.weight_range
        weights = torch.empty(B, M, device=self.device).uniform_(w_min, w_max)

        # Gather per-atom target from kp_window[b, τ_i, k_i].
        # Build a [B, M] pair of (τ_index, k_index) and gather.
        b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, M)  # [B, M]
        tau_idx = taus.long().clamp(0, H)                                       # [B, M]
        k_idx = keypoint_ids.clamp(0, K - 1).long()                             # [B, M]
        targets = kp_window[b_idx, tau_idx, k_idx]                              # [B, M, 3]
        targets = targets + torch.randn_like(targets) * self.target_noise_std

        # Apply dropout.
        dropout_mask = (torch.rand(B, M, device=self.device) > self.constraint_dropout_prob).float()
        mask = mask * dropout_mask

        # Mask atoms that landed on an invalid frame (e.g. reset crossing
        # in hindsight).
        if valid_atom_mask is not None:
            # valid_atom_mask: [B, H+1], True where that lookahead frame is
            # still on the same trajectory segment.
            v = valid_atom_mask[b_idx, tau_idx].float()                         # [B, M]
            mask = mask * v

        return {
            "keypoint_ids": keypoint_ids,
            "targets": targets,
            "taus": taus,
            "weights": weights,
            "mask": mask,
        }

    def _sample_constraints_from_keypoint_pos(
        self,
        keypoint_pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Single-frame (non-future-grounded) constraint sampling.

        **Kept for the rollout self-anchored branch only** (the 85% of
        rollout chunks whose target is the env's current body state). At
        rollout time the env's own future trajectory is not yet known, so
        we approximate the multi-time future-grounded construction with a
        single-frame anchor — every atom's target is drawn from the same
        ``keypoint_pos`` frame.

        All other paths (rollout expert-chunks, training-time hindsight /
        expert relabeling, eval) use the future-grounded variant
        :meth:`_sample_constraints_from_keypoint_future`. See the project
        note "phase-1 future-grounded constraint refactor" for context.

        Args:
            keypoint_pos: [B, K, 3]
        Returns:
            dict of padded [B, M, *] constraint set tensors.
        """
        B, K, td = keypoint_pos.shape
        M = self.policy.max_constraints
        assert K == self.policy.num_keypoints, (
            f"keypoint_pos has {K} keypoints but policy expects {self.policy.num_keypoints}"
        )
        assert td == self.policy.target_dim

        # Number of constraints per sample
        n_per = torch.randint(
            self.n_constraints_min, min(self.n_constraints_max, M) + 1,
            (B,), device=self.device,
        )
        arange = torch.arange(M, device=self.device).unsqueeze(0).expand(B, -1)
        mask = (arange < n_per.unsqueeze(1)).float()

        keypoint_ids = torch.randint(0, K, (B, M), device=self.device)
        taus = torch.randint(1, self.tau_max + 1, (B, M), device=self.device).float()
        w_min, w_max = self.weight_range
        weights = torch.empty(B, M, device=self.device).uniform_(w_min, w_max)

        kid_expanded = keypoint_ids.unsqueeze(-1).expand(-1, -1, td)
        targets = torch.gather(keypoint_pos, 1, kid_expanded)
        targets = targets + torch.randn_like(targets) * self.target_noise_std

        dropout_mask = (torch.rand(B, M, device=self.device) > self.constraint_dropout_prob).float()
        mask = mask * dropout_mask

        return {
            "keypoint_ids": keypoint_ids,
            "targets": targets,
            "taus": taus,
            "weights": weights,
            "mask": mask,
        }

    def sample_constraint_set_vectorized(
        self,
        priv_state: torch.Tensor,
        num_envs: int,
    ) -> dict[str, torch.Tensor]:
        """Single-frame rollout self-anchored constraint sampler.

        **Not future-grounded** — every atom's target comes from the same
        ``priv_state`` frame (the env's current body pose), with random
        τ_i ∈ [1, tau_max]. This is the phase-1 leftover: at rollout time
        we don't have the env's own future trajectory yet, so we can't
        do proper per-atom future grounding here.

        This path is invoked only for the rollout *self*-chunks (the 85%
        of fresh chunks that anchor on the env's current priv). Rollout
        *expert* chunks, training-time hindsight / expert relabeling, the
        discriminator positive branch, and live-eval all use the proper
        future-grounded :meth:`_sample_constraints_from_keypoint_future`.

        A planned phase-2 fix is to maintain a per-env priv ring buffer
        inside ``_advance_chunk`` and anchor new chunks ``tau_max`` steps
        in the past so its "future window" is already known — see
        project memory ``project_sparse_successor_phase1_future_grounded``.
        """
        M = self.policy.max_constraints
        td = self.policy.target_dim
        nk = self.policy.num_keypoints

        # Sample number of constraints per env
        n_per_env = torch.randint(
            self.n_constraints_min, min(self.n_constraints_max, M) + 1,
            (num_envs,), device=self.device,
        )
        # Build mask from n_per_env
        arange = torch.arange(M, device=self.device).unsqueeze(0).expand(num_envs, -1)
        mask = (arange < n_per_env.unsqueeze(1)).float()

        # Random keypoint ids
        keypoint_ids = torch.randint(0, nk, (num_envs, M), device=self.device)
        # Random taus
        taus = torch.randint(1, self.tau_max + 1, (num_envs, M), device=self.device).float()
        # Random weights
        w_min, w_max = self.weight_range
        weights = torch.empty(num_envs, M, device=self.device).uniform_(w_min, w_max)

        # Extract targets from priv_state based on keypoint_ids
        nk_total = nk
        offset = priv_state.shape[-1] - nk_total * td
        # Gather keypoint values: [num_envs, M, td]
        # Reshape priv to extract keypoint block
        kp_block = priv_state[:, offset:].reshape(num_envs, nk_total, td)  # [N, nk, td]
        # Gather using keypoint_ids
        kid_expanded = keypoint_ids.unsqueeze(-1).expand(-1, -1, td)  # [N, M, td]
        targets = torch.gather(kp_block, 1, kid_expanded.clamp(0, nk_total - 1))  # [N, M, td]
        # Add noise
        targets = targets + torch.randn_like(targets) * self.target_noise_std

        # Apply dropout: randomly zero out some constraints
        dropout_mask = (torch.rand(num_envs, M, device=self.device) > self.constraint_dropout_prob).float()
        mask = mask * dropout_mask

        return {
            "keypoint_ids": keypoint_ids,
            "targets": targets,
            "taus": taus,
            "weights": weights,
            "mask": mask,
        }

    # ------------------------------------------------------------------
    # Snippet building (vectorized)
    # ------------------------------------------------------------------

    def _push_snippet_frame(self, style_feature: torch.Tensor) -> None:
        """Write one per-env style-feature frame into the ring buffer.

        Args:
            style_feature: [num_envs, style_feature_dim]
        """
        self._snippet_ring[:, self._snippet_write_ptr] = style_feature.detach()
        self._snippet_write_ptr = (self._snippet_write_ptr + 1) % self.snippet_length
        self._snippet_fill.add_(1).clamp_(max=self.snippet_length)

    def _current_snippet_batch(self) -> torch.Tensor:
        """Return the current snippet for every env, flattened to [num_envs, L*style_dim].

        The oldest frame becomes index 0 and the newest becomes index L-1.
        Envs that have seen fewer than ``snippet_length`` frames have their
        initial slots repeated with the earliest observed frame (replicate
        padding).
        """
        L = self.snippet_length
        # Index order: start at write_ptr (oldest), wrap around to write_ptr-1.
        idx = torch.arange(L, device=self.device)
        roll = (self._snippet_write_ptr + idx) % L                         # [L]
        # Take all frames in ascending time order
        snippet = self._snippet_ring[:, roll]                              # [N, L, D]

        # Replicate-pad for envs that haven't filled yet. For each env, the
        # first (L - fill) slots of the rolled view are stale; overwrite them
        # with the oldest-valid frame (at slot index (L - fill)).
        fill = self._snippet_fill.clamp(max=L)                             # [N]
        # Build an [N, L] mask: True where a slot needs to be replaced.
        slot_idx = torch.arange(L, device=self.device).unsqueeze(0)        # [1, L]
        pad_mask = slot_idx < (L - fill).unsqueeze(1)                      # [N, L]
        # Oldest valid frame index per env = (L - fill), clamped to L-1
        oldest_idx = (L - fill).clamp(max=L - 1)                           # [N]
        oldest_frame = snippet[torch.arange(snippet.shape[0], device=self.device), oldest_idx]  # [N, D]
        # Broadcast the oldest-frame into the padded slots.
        snippet = torch.where(pad_mask.unsqueeze(-1), oldest_frame.unsqueeze(1), snippet)
        return snippet.reshape(snippet.shape[0], -1).contiguous()

    def _reset_snippet_buffer(self, env_ids: torch.Tensor) -> None:
        """Reset snippet fill count for environments that terminated.

        We don't zero the ring itself — the next ``_push_snippet_frame`` will
        overwrite slots as they come around; the fill counter controls the
        replicate-padding logic.
        """
        if env_ids is None:
            return
        ids = env_ids.reshape(-1)
        if ids.numel() == 0:
            return
        ids = ids.to(self._snippet_fill.device).long()
        self._snippet_fill[ids] = 0

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------

    @staticmethod
    def gaussian_time_kernel(tau: torch.Tensor, h: float, sigma_time: float) -> torch.Tensor:
        """K_tau(h) = exp(-(h - tau)^2 / (2 * sigma^2))"""
        return torch.exp(-((h - tau) ** 2) / (2 * sigma_time ** 2))

    def satisfaction(self, priv_state: torch.Tensor, keypoint_ids: torch.Tensor,
                     targets: torch.Tensor) -> torch.Tensor:
        """Compute satisfaction c(s, q) = exp(-||p - xi||^2 / beta(k)^2).

        Uses per-keypoint beta (gathered via ``keypoint_ids``) so that
        end-effectors with larger natural positional range can have a wider
        satisfaction kernel than stable bodies like the pelvis.

        Args:
            priv_state: [B, priv_dim]
            keypoint_ids: [B, N]
            targets: [B, N, target_dim]

        Returns:
            [B, N] satisfaction values in [0, 1]
        """
        B, N = keypoint_ids.shape
        td = self.policy.target_dim
        nk = self.policy.num_keypoints
        offset = priv_state.shape[-1] - nk * td

        kp_block = priv_state[:, offset:].reshape(B, nk, td)
        kid_expanded = keypoint_ids.unsqueeze(-1).expand(-1, -1, td)
        p = torch.gather(kp_block, 1, kid_expanded.clamp(0, nk - 1))  # [B, N, td]

        dist2 = ((p - targets) ** 2).sum(dim=-1)  # [B, N]
        # Per-query beta gathered from [num_keypoints]; the tensor is moved to
        # the right device once in __init__.
        beta_q = self.beta.to(keypoint_ids.device)[keypoint_ids]  # [B, N]
        return torch.exp(-dist2 / (beta_q ** 2))

    def compute_immediate_query_rewards(
        self,
        priv_state_next: torch.Tensor,
        keypoint_ids: torch.Tensor,
        targets: torch.Tensor,
        taus: torch.Tensor,
    ) -> torch.Tensor:
        """Compute K_tau(1) * c(s_{t+1}, q) for each query.

        Returns: [B, N]
        """
        kernel_val = self.gaussian_time_kernel(taus, h=1.0, sigma_time=self.sigma_time)  # [B, N]
        sat = self.satisfaction(priv_state_next, keypoint_ids, targets)  # [B, N]
        return kernel_val * sat

    # ------------------------------------------------------------------
    # Act / process_env_step / compute_returns
    # ------------------------------------------------------------------

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor, infos: dict | None = None) -> torch.Tensor:
        """Sample actions conditioned on current constraint set z_C."""
        num_envs = obs.shape[0]

        # Ensure constraints exist
        if self._env_constraints is None or self._env_constraints["mask"].shape[0] != num_envs:
            self._init_constraints(num_envs)

        # Encode constraint set
        with torch.no_grad():
            z_C = self.policy.encode_constraint_set(
                self._env_constraints["keypoint_ids"],
                self._env_constraints["targets"],
                self._env_constraints["taus"],
                self._env_constraints["weights"],
                self._env_constraints["mask"],
            )

        actions, _ = self.policy.actor.sample(obs, z_C)
        self.transition.actions = actions.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs

        # Store current constraint set into transition
        self.transition.constraint_keypoint_ids = self._env_constraints["keypoint_ids"].clone()
        self.transition.constraint_targets = self._env_constraints["targets"].clone()
        self.transition.constraint_taus = self._env_constraints["taus"].clone()
        self.transition.constraint_weights = self._env_constraints["weights"].clone()
        self.transition.constraint_mask = self._env_constraints["mask"].clone()

        return actions.detach()

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
        style_features: torch.Tensor | None = None,
    ):
        """Record reward/done and push one frame into the snippet ring.

        Args:
            rewards: [num_envs]
            dones: [num_envs]
            infos: runner-provided info dict (unused here).
            style_features: [num_envs, style_feature_dim] — per-env BFM-style
                features for the frame that just completed. If ``None``, the
                snippet ring is not updated for this step; the snippet at
                commit time will keep its previous content. Pass ``None`` only
                when the env side cannot produce features (e.g. during smoke
                tests).
        """
        num_envs = rewards.shape[0]

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones.clone()

        if style_features is not None:
            assert style_features.shape == (num_envs, self.style_feature_dim), (
                f"style_features shape {tuple(style_features.shape)} != "
                f"({num_envs}, {self.style_feature_dim})"
            )
            self._push_snippet_frame(style_features)

        self.transition.snippet = self._current_snippet_batch()

        # Reset snippet fill counters for done envs BEFORE next push.
        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._reset_snippet_buffer(done_ids)

    def set_next_obs(self, next_obs: torch.Tensor, next_priv_obs: torch.Tensor):
        """Called by the runner after env.step to commit the transition.

        Chunk-level constraint update:
        - If this env is mid-chunk (counter > 0) and didn't reset, we keep
          (k, ξ, w, m) fixed and only decrement τ by 1.
        - If the chunk ended or the env reset, sample a fresh C from
          ``next_priv_obs`` (optionally expert-anchored for a small fraction
          of envs).

        The transition's ``next_constraint_*`` fields are always set to the
        post-step constraint state, which is what the next ``act()`` call
        will consume. That way replay and bootstrap stay consistent — the
        stored ``next_C`` IS the C used when action a' was sampled.
        """
        self.transition.next_observations = next_obs
        self.transition.next_privileged_observations = next_priv_obs

        # Reset mask from the transition's dones (set by process_env_step).
        reset_mask = None
        if self.transition.dones is not None:
            reset_mask = self.transition.dones > 0

        # Advance the chunk state in-place on self._env_constraints.
        self._advance_chunk(next_priv_obs, reset_mask)

        # Mirror the new _env_constraints into the transition's next-C fields.
        self.transition.next_constraint_keypoint_ids = self._env_constraints["keypoint_ids"].clone()
        self.transition.next_constraint_targets = self._env_constraints["targets"].clone()
        self.transition.next_constraint_taus = self._env_constraints["taus"].clone()
        self.transition.next_constraint_weights = self._env_constraints["weights"].clone()
        self.transition.next_constraint_mask = self._env_constraints["mask"].clone()

        # Commit transition to storage
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_critic_obs: torch.Tensor, **kwargs):
        """No-op for off-policy algorithm."""
        pass

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:  # noqa: C901
        """Run one full update cycle: discriminator, successor critics, style critics, actor."""
        if self.storage.size < self.mini_batch_size:
            return {}

        self._diag_reset()
        loss_dict: dict[str, float] = {}
        total_disc_loss = 0.0
        total_U_loss = 0.0
        total_QS_loss = 0.0
        total_QA_loss = 0.0
        total_actor_loss = 0.0
        num_updates = 0

        # Build an iterator over mini-batches. Two modes:
        #   • ``num_updates_per_iter`` set → off-policy replay: draw that many
        #     independent samples from the whole replay buffer.
        #   • otherwise → legacy behaviour: one shuffled pass × num_learning_epochs.
        if self.num_updates_per_iter is not None and self.num_updates_per_iter > 0:
            def _mini_batch_iter():
                for _ in range(self.num_updates_per_iter):
                    yield self.storage.sample(self.mini_batch_size)
            mini_batch_iter = _mini_batch_iter()
        else:
            mini_batch_iter = self.storage.mini_batch_generator(
                self.mini_batch_size, self.num_learning_epochs
            )

        for mini_batch in mini_batch_iter:
            (
                obs, priv, actions, next_obs, next_priv, dones, rewards_batch,
                c_kid, c_tgt, c_tau, c_w, c_m,
                nc_kid, nc_tgt, nc_tau, nc_w, nc_m,
                snippets,
                t_idx_batch, env_idx_batch,
            ) = mini_batch

            dones_flat = dones.squeeze(-1)  # [B]

            # ----------------------------------------------------------
            # C-space relabeling: mix stored / hindsight / expert sources.
            # We use the SAME relabeled C on both the (s, a) and (s', a')
            # sides of every TD target in this batch — no mismatch between
            # c_* and nc_*. The bootstrap's ``q^-`` is constructed below by
            # decrementing τ on the same relabeled C.
            # ----------------------------------------------------------
            stored = {
                "keypoint_ids": c_kid,
                "targets": c_tgt,
                "taus": c_tau,
                "weights": c_w,
                "mask": c_m,
            }
            relabeled, relabel_counts = self._relabel_constraint_sets(
                stored, next_priv,
                t_idx=t_idx_batch, env_idx=env_idx_batch,
            )
            c_kid = relabeled["keypoint_ids"]
            c_tgt = relabeled["targets"]
            c_tau = relabeled["taus"]
            c_w = relabeled["weights"]
            c_m = relabeled["mask"]
            # Share with the next-side so z_C_next is anchored on the same C.
            nc_kid, nc_tgt, nc_tau, nc_w, nc_m = c_kid, c_tgt, c_tau, c_w, c_m

            # ----------------------------------------------------------
            # Encode constraints
            # ----------------------------------------------------------
            z_C = self.policy.encode_constraint_set(c_kid, c_tgt, c_tau, c_w, c_m)
            # z_C_next shares the relabeled set — see note above.
            z_C_next = z_C

            q_emb = self.policy.encode_single_queries(c_kid, c_tgt, c_tau)           # [B, N, D]

            # z_C norm diagnostics (group 2 / scale). Logged against rollout
            # z_C so you can watch it for drift even with unit-sphere projection
            # disabled.
            with torch.no_grad():
                z_norm = z_C.norm(dim=-1)
                self._diag_add("Scale/z_C_norm_mean", z_norm.mean().item())
                self._diag_add("Scale/z_C_norm_std", z_norm.std().item())

            # ----------------------------------------------------------
            # 1) Train discriminator
            # ----------------------------------------------------------
            # Real GAN training: positives are precomputed expert snippets
            # (with z_C sampled from the expert's own keypoint positions so
            # the (snippet, z) pair is mutually consistent), negatives are
            # the rollout snippet / z_C pair for the same mini-batch.
            eps = 1e-6
            disc_loss = None
            grad_penalty_val = 0.0
            if self.expert_buffer is not None:
                # Per-atom future-grounded expert z: snippet + kp_window, each
                # atomic constraint samples its own τ_i from [1, tau_max] and
                # draws its target from the corresponding future frame. Note
                # scheme (B): we do NOT require the snippet horizon L to
                # cover every τ_i — the disc only sees the snippet as
                # evidence of style, while the z_C it's paired with carries
                # multi-time query information that the snippet's 8-frame
                # window cannot fully verify. This is the intended trade-off.
                expert_batch = self.expert_buffer.sample_with_future_window(
                    snippets.shape[0], horizon=self.tau_max,
                )
                expert_snippets = expert_batch["snippet"].to(self.device)    # [B, L*style_dim]
                expert_window = expert_batch["kp_window"].to(self.device)    # [B, H+1, K, 3]
                expert_constraints = self._sample_constraints_from_keypoint_future(expert_window)
                with torch.no_grad():
                    expert_z = self.policy.encode_constraint_set(
                        expert_constraints["keypoint_ids"],
                        expert_constraints["targets"],
                        expert_constraints["taus"],
                        expert_constraints["weights"],
                        expert_constraints["mask"],
                    )
                    # Track expert z_C norm so any rollout-vs-expert magnitude
                    # drift surfaces clearly.
                    self._diag_add("Scale/expert_z_norm_mean", expert_z.norm(dim=-1).mean().item())

                pos_out = self.policy.style_discriminator(expert_snippets, expert_z)
                neg_out = self.policy.style_discriminator(snippets, z_C.detach())
                disc_loss = -(torch.log(pos_out + eps).mean()
                              + torch.log(1.0 - neg_out + eps).mean())

                # WGAN-style gradient penalty on the interpolated pair.
                alpha_gp = torch.rand(snippets.shape[0], 1, device=self.device)
                interp_snippet = (alpha_gp * expert_snippets
                                  + (1 - alpha_gp) * snippets).detach().requires_grad_(True)
                interp_z = (alpha_gp * expert_z
                            + (1 - alpha_gp) * z_C.detach()).detach().requires_grad_(True)
                interp_out = self.policy.style_discriminator(interp_snippet, interp_z)
                grad = torch.autograd.grad(
                    outputs=interp_out.sum(), inputs=[interp_snippet, interp_z],
                    create_graph=True, retain_graph=True,
                )
                grad_cat = torch.cat([g.reshape(g.shape[0], -1) for g in grad], dim=-1)
                grad_penalty = ((grad_cat.norm(dim=-1) - 1) ** 2).mean() * self.grad_penalty_weight
                disc_loss = disc_loss + grad_penalty
                grad_penalty_val = grad_penalty.item()

                self.opt_disc.zero_grad()
                disc_loss.backward()
                if self._sync_grads:
                    reduce_gradients(self.policy.style_discriminator)
                nn.utils.clip_grad_norm_(
                    self.policy.style_discriminator.parameters(), self.max_grad_norm
                )
                self.opt_disc.step()

                # Discriminator diagnostics (group 5)
                with torch.no_grad():
                    self._diag_add("Disc/pos_mean", pos_out.mean().item())
                    self._diag_add("Disc/pos_std", pos_out.std().item())
                    self._diag_add("Disc/neg_mean", neg_out.mean().item())
                    self._diag_add("Disc/neg_std", neg_out.std().item())
                    self._diag_add("Disc/gap_mean", (pos_out.mean() - neg_out.mean()).item())
                    self._diag_add("Disc/grad_penalty", grad_penalty_val)

            # ----------------------------------------------------------
            # 2) Train twin successor critics
            # ----------------------------------------------------------
            with torch.no_grad():
                next_action, _ = self.policy.actor.sample(next_obs, z_C_next)

                # q^- by decrementing tau
                taus_minus_1 = (c_tau - 1).clamp(min=1)
                q_minus_emb = self.policy.encode_single_queries(c_kid, c_tgt, taus_minus_1)

                targ_U1 = self.policy.successor_critic_1_target(next_obs, next_priv, next_action, q_minus_emb)
                targ_U2 = self.policy.successor_critic_2_target(next_obs, next_priv, next_action, q_minus_emb)
                targ_U = self._pessimistic_q(targ_U1, targ_U2, self.critic_pessimism_penalty)

                immediate = self.compute_immediate_query_rewards(next_priv, c_kid, c_tgt, c_tau)

                tau_gt_1 = (c_tau > 1).float()
                y_U = immediate + self.gamma * tau_gt_1 * (1.0 - dones_flat.unsqueeze(-1)) * targ_U

            pred_U1 = self.policy.successor_critic_1(obs, priv, actions, q_emb)
            pred_U2 = self.policy.successor_critic_2(obs, priv, actions, q_emb)

            mask_float = c_m.float()
            weights = c_w
            mask_sum = mask_float.sum() + 1e-6

            loss_U1 = (((pred_U1 - y_U) ** 2) * weights * mask_float).sum() / mask_sum
            loss_U2 = (((pred_U2 - y_U) ** 2) * weights * mask_float).sum() / mask_sum
            loss_U = loss_U1 + loss_U2

            self.opt_query.zero_grad()
            self.opt_U1.zero_grad()
            self.opt_U2.zero_grad()
            loss_U.backward()
            if self._sync_grads:
                reduce_gradients(self.policy.query_encoder)
                reduce_gradients(self.policy.successor_critic_1)
                reduce_gradients(self.policy.successor_critic_2)
            nn.utils.clip_grad_norm_(self.policy.successor_critic_1.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.policy.successor_critic_2.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.policy.query_encoder.parameters(), self.max_grad_norm)
            self.opt_U1.step()
            self.opt_U2.step()
            self.opt_query.step()

            # Successor critic diagnostics (groups 3 & 4)
            with torch.no_grad():
                # Use averaged twin predictions for scalar diagnostics.
                pred_U_combined = 0.5 * (pred_U1 + pred_U2)
                # Bootstrap term matches the target construction: γ·1[τ>1]·(1-done)·targ_U
                bootstrap_term = (
                    self.gamma * tau_gt_1
                    * (1.0 - dones_flat.unsqueeze(-1)) * targ_U
                )
                self._log_query_buckets(
                    pred_U=pred_U_combined,
                    target_U=y_U,
                    immediate=immediate,
                    bootstrap_term=bootstrap_term,
                    taus=c_tau,
                    kids=c_kid,
                    mask=mask_float,
                )

            # ----------------------------------------------------------
            # 3) Train style critics — only when the discriminator is real.
            # Without an expert buffer, the style reward would be meaningless,
            # so we skip the whole style branch instead of training on noise.
            # ----------------------------------------------------------
            loss_QS1 = None
            loss_QS2 = None
            if self.expert_buffer is not None:
                with torch.no_grad():
                    next_action_style, _ = self.policy.actor.sample(next_obs, z_C_next)
                    # Clamp D in (eps, 1-eps) to avoid inf logits — same form as
                    # BFM's Discriminator.compute_reward.
                    D_clamped = self.policy.style_discriminator(snippets, z_C).clamp(eps, 1.0 - eps)
                    r_style = torch.log(D_clamped) - torch.log(1.0 - D_clamped)

                    self._last_r_style_mean = r_style.mean().item()
                    self._last_r_style_std = r_style.std().item()

                    q_style_next = self._pessimistic_q(
                        self.policy.style_critic_1_target(next_obs, next_priv, next_action_style, z_C_next),
                        self.policy.style_critic_2_target(next_obs, next_priv, next_action_style, z_C_next),
                        self.critic_pessimism_penalty,
                    )
                    y_style = r_style + self.gamma_style * (1.0 - dones_flat) * q_style_next

                pred_QS1 = self.policy.style_critic_1(obs, priv, actions, z_C.detach())
                pred_QS2 = self.policy.style_critic_2(obs, priv, actions, z_C.detach())

                loss_QS1 = ((pred_QS1 - y_style) ** 2).mean()
                loss_QS2 = ((pred_QS2 - y_style) ** 2).mean()

                self.opt_QS1.zero_grad()
                loss_QS1.backward()
                if self._sync_grads:
                    reduce_gradients(self.policy.style_critic_1)
                nn.utils.clip_grad_norm_(self.policy.style_critic_1.parameters(), self.max_grad_norm)
                self.opt_QS1.step()

                self.opt_QS2.zero_grad()
                loss_QS2.backward()
                if self._sync_grads:
                    reduce_gradients(self.policy.style_critic_2)
                nn.utils.clip_grad_norm_(self.policy.style_critic_2.parameters(), self.max_grad_norm)
                self.opt_QS2.step()

                # Style diagnostics (group 5)
                with torch.no_grad():
                    self._diag_add("Loss/style_critic_1", loss_QS1.item())
                    self._diag_add("Loss/style_critic_2", loss_QS2.item())
                    self._diag_add("Style/r_style_mean", r_style.mean().item())
                    self._diag_add("Style/r_style_std", r_style.std().item())
                    self._diag_add("Style/q_style_target_mean", y_style.mean().item())
                    pred_qs = 0.5 * (pred_QS1 + pred_QS2)
                    self._diag_add("Style/q_style_pred_mean", pred_qs.mean().item())

            # ----------------------------------------------------------
            # 3.5) Train aux critics against env-level shaping rewards
            # (BFM-style ``aux_critic``). Uses a running reward normalizer so
            # the Q scale stays O(1) regardless of raw reward magnitude.
            # ----------------------------------------------------------
            loss_QA1 = None
            loss_QA2 = None
            q_aux_for_actor = None
            if self.lambda_aux > 0.0:
                rewards_flat = rewards_batch.squeeze(-1)  # [B]
                # Update the running stats from this batch, then normalize.
                self.policy.aux_reward_normalizer.update(rewards_flat)
                r_env_norm = self.policy.aux_reward_normalizer.normalize(rewards_flat)

                self._last_r_env_raw_mean = rewards_flat.mean().item()
                self._last_r_env_mean = r_env_norm.mean().item()
                self._last_r_env_std = r_env_norm.std().item()

                with torch.no_grad():
                    next_action_aux, _ = self.policy.actor.sample(next_obs, z_C_next)
                    q_aux_next = self._pessimistic_q(
                        self.policy.aux_critic_1_target(next_obs, next_priv, next_action_aux, z_C_next),
                        self.policy.aux_critic_2_target(next_obs, next_priv, next_action_aux, z_C_next),
                        self.critic_pessimism_penalty,
                    )
                    y_aux = r_env_norm + self.gamma * (1.0 - dones_flat) * q_aux_next

                pred_QA1 = self.policy.aux_critic_1(obs, priv, actions, z_C.detach())
                pred_QA2 = self.policy.aux_critic_2(obs, priv, actions, z_C.detach())
                loss_QA1 = ((pred_QA1 - y_aux) ** 2).mean()
                loss_QA2 = ((pred_QA2 - y_aux) ** 2).mean()

                self.opt_QA1.zero_grad()
                loss_QA1.backward()
                if self._sync_grads:
                    reduce_gradients(self.policy.aux_critic_1)
                nn.utils.clip_grad_norm_(self.policy.aux_critic_1.parameters(), self.max_grad_norm)
                self.opt_QA1.step()

                self.opt_QA2.zero_grad()
                loss_QA2.backward()
                if self._sync_grads:
                    reduce_gradients(self.policy.aux_critic_2)
                nn.utils.clip_grad_norm_(self.policy.aux_critic_2.parameters(), self.max_grad_norm)
                self.opt_QA2.step()

                # Aux diagnostics (group 6)
                with torch.no_grad():
                    self._diag_add("Loss/aux_critic_1", loss_QA1.item())
                    self._diag_add("Loss/aux_critic_2", loss_QA2.item())
                    self._diag_add("Aux/r_env_raw_mean", rewards_flat.mean().item())
                    self._diag_add("Aux/r_env_raw_std", rewards_flat.std().item())
                    self._diag_add("Aux/r_env_norm_mean", r_env_norm.mean().item())
                    self._diag_add("Aux/r_env_norm_std", r_env_norm.std().item())
                    self._diag_add("Aux/q_aux_target_mean", y_aux.mean().item())
                    pred_qa = 0.5 * (pred_QA1 + pred_QA2)
                    self._diag_add("Aux/q_aux_pred_mean", pred_qa.mean().item())

            # Detach the query path so the query/constraint encoders are shaped
            # purely by the successor critics' TD loss, not by actor exploitation
            # (mirrors BFM, where backward_map is never called in the actor update).
            z_C_actor = z_C.detach()
            q_emb_actor = q_emb.detach()

            new_action, _ = self.policy.actor.sample(obs, z_C_actor)

            qU1 = self.policy.successor_critic_1(obs, priv, new_action, q_emb_actor)
            qU2 = self.policy.successor_critic_2(obs, priv, new_action, q_emb_actor)

            q_track_1 = (qU1 * weights * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-6)
            q_track_2 = (qU2 * weights * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-6)
            q_track = self._pessimistic_q(q_track_1, q_track_2, self.actor_pessimism_penalty)

            q_total = q_track

            if self.expert_buffer is not None and self.lambda_style > 0.0:
                q_style = self._pessimistic_q(
                    self.policy.style_critic_1(obs, priv, new_action, z_C_actor),
                    self.policy.style_critic_2(obs, priv, new_action, z_C_actor),
                    self.actor_pessimism_penalty,
                )
                q_total = q_total + self.lambda_style * q_style
            else:
                q_style = None

            if self.lambda_aux > 0.0:
                q_aux = self._pessimistic_q(
                    self.policy.aux_critic_1(obs, priv, new_action, z_C_actor),
                    self.policy.aux_critic_2(obs, priv, new_action, z_C_actor),
                    self.actor_pessimism_penalty,
                )
                q_total = q_total + self.lambda_aux * q_aux
            else:
                q_aux = None

            loss_actor = -q_total.mean()

            # Actor-objective decomposition (group 2) + action stats (group 8).
            with torch.no_grad():
                self._diag_add("Scale/q_track_mean", q_track.mean().item())
                self._diag_add("Scale/q_track_std", q_track.std().item())
                self._diag_add("Scale/q_total_mean", q_total.mean().item())
                if q_style is not None:
                    self._diag_add("Scale/q_style_mean", q_style.mean().item())
                    self._diag_add("Scale/q_style_std", q_style.std().item())
                    self._diag_add(
                        "Scale/lambda_style_times_q_style_mean",
                        (self.lambda_style * q_style).mean().item(),
                    )
                if q_aux is not None:
                    self._diag_add("Scale/q_aux_mean", q_aux.mean().item())
                    self._diag_add("Scale/q_aux_std", q_aux.std().item())
                    self._diag_add(
                        "Scale/lambda_aux_times_q_aux_mean",
                        (self.lambda_aux * q_aux).mean().item(),
                    )

                # Action stats (group 8) — note new_action already went through
                # tanh + truncated-normal clamp so it lives in [action_low, action_high].
                lo = self.policy.actor.action_low
                hi = self.policy.actor.action_high
                clip_eps = 1e-3
                clip_frac = (
                    (new_action <= lo + clip_eps) | (new_action >= hi - clip_eps)
                ).float().mean().item()
                # mu: the deterministic mean before adding Gaussian noise.
                mu = self.policy.actor.forward(obs, z_C_actor)
                self._diag_add("Action/action_mean", new_action.mean().item())
                self._diag_add("Action/action_std", new_action.std().item())
                self._diag_add("Action/action_abs_mean", new_action.abs().mean().item())
                self._diag_add("Action/action_clip_fraction", clip_frac)
                self._diag_add("Action/mu_mean", mu.mean().item())
                self._diag_add("Action/mu_std", mu.std().item())

            self.opt_actor.zero_grad()
            loss_actor.backward()
            if self._sync_grads:
                reduce_gradients(self.policy.actor)
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            self.opt_actor.step()

            # ----------------------------------------------------------
            # 5) Soft update targets
            # ----------------------------------------------------------
            self._soft_update(self.policy.successor_critic_1_target, self.policy.successor_critic_1)
            self._soft_update(self.policy.successor_critic_2_target, self.policy.successor_critic_2)
            if self.expert_buffer is not None:
                self._soft_update(self.policy.style_critic_1_target, self.policy.style_critic_1)
                self._soft_update(self.policy.style_critic_2_target, self.policy.style_critic_2)
            if self.lambda_aux > 0.0:
                self._soft_update(self.policy.aux_critic_1_target, self.policy.aux_critic_1)
                self._soft_update(self.policy.aux_critic_2_target, self.policy.aux_critic_2)

            # Per-mini-batch accounting
            B_size = relabeled["keypoint_ids"].shape[0]
            self._diag_add("Relabel/stored_fraction", relabel_counts["stored"] / max(B_size, 1))
            self._diag_add("Relabel/hindsight_fraction", relabel_counts["hindsight"] / max(B_size, 1))
            self._diag_add("Relabel/expert_fraction", relabel_counts["expert"] / max(B_size, 1))

            if disc_loss is not None:
                total_disc_loss += disc_loss.item()
                self._diag_add("Loss/discriminator", disc_loss.item())
            total_U_loss += (loss_U1.item() + loss_U2.item()) / 2
            self._diag_add("Loss/successor_critic_1", loss_U1.item())
            self._diag_add("Loss/successor_critic_2", loss_U2.item())
            if loss_QS1 is not None:
                total_QS_loss += (loss_QS1.item() + loss_QS2.item()) / 2
            if loss_QA1 is not None:
                total_QA_loss += (loss_QA1.item() + loss_QA2.item()) / 2
            total_actor_loss += loss_actor.item()
            self._diag_add("Loss/actor", loss_actor.item())
            num_updates += 1
            self._diag_bump()

        if num_updates > 0:
            # Keep the top-level short names for the terminal printout — the
            # full per-group detail is attached from self._diag below.
            if self.expert_buffer is not None:
                loss_dict["discriminator"] = total_disc_loss / num_updates
                loss_dict["style_critic"] = total_QS_loss / num_updates
            if self.lambda_aux > 0.0:
                loss_dict["aux_critic"] = total_QA_loss / num_updates
            loss_dict["successor_critic"] = total_U_loss / num_updates
            loss_dict["actor"] = total_actor_loss / num_updates

            # Replay stats (group 7)
            if self.storage is not None:
                size = self.storage.size
                capacity = self.storage.num_transitions_per_env * self.storage.num_envs
                self._diag_add("Replay/size", float(size))
                self._diag_add("Replay/fill_ratio", size / max(capacity, 1))

            # Rollout chunk source mixture (recorded at the last
            # ``_advance_chunk`` call with fresh envs). These scalars are
            # per-env fractions, summing to ~1 across the three sources.
            for key in (
                "_diag_rollout_source_live",
                "_diag_rollout_source_replay",
                "_diag_rollout_source_expert",
            ):
                if hasattr(self, key):
                    tag = "Rollout/" + key.replace("_diag_rollout_source_", "source_")
                    self._diag_add(tag, getattr(self, key))
            if hasattr(self, "_diag_rollout_fresh_frac"):
                self._diag_add("Rollout/fresh_fraction", self._diag_rollout_fresh_frac)

            # Merge the full diagnostic dict (mean across mini-batches).
            loss_dict.update(self._diag)

        # Off-policy replay: do NOT clear storage — the circular buffer
        # keeps accumulating transitions so subsequent updates can reuse them.
        return loss_dict

    # ------------------------------------------------------------------
    # Evaluation — BFM-style live tracking rollout
    # ------------------------------------------------------------------

    def snapshot_state(self) -> dict:
        """Save the parts of the algorithm state that an eval rollout disturbs.

        Used in tandem with ``env.snapshot_state`` by
        ``evaluate_live_tracking`` so the live training rollout is not
        interrupted when the eval runs. Covers:
          - constraint set per env + chunk-step counter (rollout constraints)
          - snippet ring + fill counter + write pointer (style discriminator)

        The storage/replay is *not* snapshotted — eval doesn't add
        transitions to it (we use ``clear_transition=True`` in the rollout).
        The actor's aux reward normalizer is also left alone; it is only
        touched during training updates, not during eval.
        """
        snap: dict = {}
        if self._env_constraints is not None:
            snap["env_constraints"] = {
                k: v.clone() for k, v in self._env_constraints.items()
            }
            snap["chunk_steps_left"] = self._chunk_steps_left.clone()
        if self._snippet_ring is not None:
            snap["snippet_ring"] = self._snippet_ring.clone()
            snap["snippet_write_ptr"] = int(self._snippet_write_ptr)
            snap["snippet_fill"] = self._snippet_fill.clone()
        return snap

    def restore_state(self, snap: dict) -> None:
        """Restore the state captured by ``snapshot_state``."""
        if "env_constraints" in snap and self._env_constraints is not None:
            for k, v in snap["env_constraints"].items():
                self._env_constraints[k].copy_(v)
            self._chunk_steps_left.copy_(snap["chunk_steps_left"])
        if "snippet_ring" in snap and self._snippet_ring is not None:
            self._snippet_ring.copy_(snap["snippet_ring"])
            self._snippet_write_ptr = int(snap["snippet_write_ptr"])
            self._snippet_fill.copy_(snap["snippet_fill"])

    @torch.inference_mode()
    def evaluate_live_tracking(
        self,
        env,
        obs_normalizer=None,
        privileged_obs_normalizer=None,
        privileged_obs_type: str = "critic",
        horizon: int | None = None,
        action_clip_range: tuple[float, float] = (-1.0, 1.0),
    ) -> dict[str, float]:
        """BFM-style live eval rollout with save/restore around the env.

        Protocol:
          1. Snapshot the full env + algorithm state.
          2. Reset all envs to get a clean starting point.
          3. Sample one fresh constraint set per env and **freeze** it for
             the whole horizon (no chunk re-sampling, no random dropouts
             applied on top — the same C is used for every step).
          4. Roll out deterministically (``actor.act_inference``) for at
             most ``horizon = tau_max`` steps. At each query's ``τ = step+1``
             we record the realized keypoint position from the next priv
             observation and compare it against the target.
          5. Restore the env + algorithm state before returning.

        Returns a dict of ``Eval/*`` scalars: per-τ-bucket error, per-keypoint
        error, and the global mean — all in β-normalised units so values
        scale with the training satisfaction kernel.
        """
        out: dict[str, float] = {}
        if self.policy is None:
            return out

        env_u = env.unwrapped if hasattr(env, "unwrapped") else env
        if not hasattr(env_u, "snapshot_state") or not hasattr(env_u, "restore_state"):
            # Env hasn't implemented the snapshot API yet — skip with a diag.
            out["Eval/no_snapshot_api"] = 1.0
            return out

        device = self.device
        td = self.policy.target_dim
        nk = self.policy.num_keypoints
        num_envs = env.num_envs

        H = int(horizon) if horizon is not None else int(self.tau_max)
        if H <= 0:
            return out

        # 1. Snapshot both sides.
        env_snap = env_u.snapshot_state()
        alg_snap = self.snapshot_state()

        # Flag the env so eval-mode resets don't pollute the training reset
        # statistics (consume_reset_stats).
        if hasattr(env_u, "set_eval_mode"):
            env_u.set_eval_mode(True)

        try:
            # 2. Clean reset.
            obs, extras = env.reset()
            obs = obs.to(device)
            priv_obs = extras["observations"].get(privileged_obs_type, obs).to(device)
            if obs_normalizer is not None:
                obs = obs_normalizer(obs)
            if privileged_obs_normalizer is not None:
                priv_obs = privileged_obs_normalizer(priv_obs)

            # 3. Freeze a fresh C per env, per-atom future-grounded from
            #    the expert buffer when available (closest match to how
            #    a user would query the policy at test time). Without an
            #    expert buffer, fall back to the single-frame self-priv
            #    sampler — not future-grounded, but at least the eval can
            #    still run. See phase-1 note on rollout self-source for
            #    why we don't use the env's own future here either.
            if self.expert_buffer is not None:
                expert_batch = self.expert_buffer.sample_with_future_window(
                    num_envs, horizon=self.tau_max,
                )
                kp_window = expert_batch["kp_window"].to(device)
                C = self._sample_constraints_from_keypoint_future(kp_window)
            else:
                C = self.sample_constraint_set_vectorized(priv_obs, num_envs)
            # In eval we want the *declared* per-query τ to be respected
            # exactly, so drop the random dropout that the sampler applies.
            mask = (C["mask"] > 0).float()
            C["mask"] = mask

            kid = C["keypoint_ids"].long()        # [N, M]
            tgt = C["targets"]                    # [N, M, td]
            tau = C["taus"].long()                # [N, M]
            mask_b = mask.bool()                  # [N, M]

            # Encode z_C once — it's frozen for the whole horizon.
            z_C = self.policy.encode_constraint_set(
                C["keypoint_ids"], C["targets"], C["taus"], C["weights"], mask
            )

            beta = self.beta.to(device)

            # Pre-allocate accumulators per bucket and per keypoint. We
            # aggregate raw per-query (N, M, H) errors at the end rather
            # than growing Python lists inside the loop.
            err_hist = torch.zeros(num_envs, kid.shape[1], H, device=device)
            seen_hist = torch.zeros(num_envs, kid.shape[1], H, device=device)

            # Track per-env liveness: if an env resets mid-eval, exclude its
            # remaining steps. BFM resets everything cleanly at the start so
            # this is primarily defensive.
            alive = torch.ones(num_envs, dtype=torch.bool, device=device)

            for step in range(H):
                actions = self.policy.actor.act_inference(obs, z_C)
                actions = actions.clamp(*action_clip_range)

                next_obs, _rewards, dones, infos = env.step(actions.to(env.device))
                next_obs = next_obs.to(device)
                dones = dones.to(device).bool()
                priv_next = infos["observations"].get(privileged_obs_type, next_obs).to(device)
                if obs_normalizer is not None:
                    next_obs = obs_normalizer(next_obs)
                if privileged_obs_normalizer is not None:
                    priv_next = privileged_obs_normalizer(priv_next)

                # Extract realized keypoint positions from the priv tail.
                priv_dim = priv_next.shape[-1]
                kp_offset = priv_dim - nk * td
                kp_block = priv_next[:, kp_offset:].reshape(num_envs, nk, td)   # [N, nk, td]

                # For every query q=(k, ξ, τ), if τ == step+1 *and* the env
                # is still alive, record the realized keypoint for that q.
                this_step = (tau == (step + 1)) & mask_b & alive.unsqueeze(-1)   # [N, M]
                if this_step.any():
                    # Gather realized keypoint per query.
                    kid_exp = kid.unsqueeze(-1).expand(-1, -1, td).clamp(0, nk - 1)
                    realized = torch.gather(
                        kp_block.unsqueeze(1).expand(-1, kid.shape[1], -1, -1),
                        -2,
                        kid_exp.unsqueeze(-2),
                    ).squeeze(-2)                                           # [N, M, td]
                    err = (realized - tgt).norm(dim=-1)                     # [N, M]
                    beta_q = beta[kid.clamp(0, nk - 1)]                     # [N, M]
                    err_norm = err / beta_q
                    err_hist[..., step][this_step] = err_norm[this_step]
                    seen_hist[..., step][this_step] = 1.0

                # Mark envs that reset this step as "not alive" going fwd.
                if dones.any():
                    alive = alive & (~dones)

                obs = next_obs

            # Aggregate per-τ-bucket and per-keypoint scalars. Each query
            # contributes to *exactly one* τ (its declared value), which
            # matches how the training kernel reasons about credit.
            err_flat = err_hist.sum(dim=-1)          # [N, M]  (per-query error)
            seen_flat = seen_hist.sum(dim=-1)        # [N, M]  (1 if recorded, else 0)
            valid = seen_flat > 0.5
            if valid.any():
                # Global mean
                out["Eval/error_norm_mean"] = float(err_flat[valid].mean().item())
                out["Eval/satisfaction_mean"] = float(
                    torch.exp(-err_flat[valid] ** 2).mean().item()
                )

                # Per τ bucket
                taus_valid = tau[valid].float()
                errs_valid = err_flat[valid]
                for lo, hi in self._tau_buckets:
                    in_b = (taus_valid >= lo) & (taus_valid <= hi)
                    if in_b.any():
                        name = f"tau_{lo:02d}_{hi:02d}"
                        out[f"Eval/error_norm_{name}"] = float(errs_valid[in_b].mean().item())
                        out[f"Eval/satisfaction_{name}"] = float(
                            torch.exp(-errs_valid[in_b] ** 2).mean().item()
                        )
                        out[f"Eval/count_{name}"] = float(in_b.sum().item())

                # Per keypoint
                kid_valid = kid[valid]
                for k in range(nk):
                    in_k = kid_valid == k
                    if in_k.any():
                        out[f"Eval/error_norm_keypoint_{k:02d}"] = float(
                            errs_valid[in_k].mean().item()
                        )

                out["Eval/num_scored_queries"] = float(valid.sum().item())
            else:
                out["Eval/num_scored_queries"] = 0.0

        finally:
            # 5. Always restore, even if the rollout raised.
            self.restore_state(alg_snap)
            env_u.restore_state(env_snap)
            if hasattr(env_u, "set_eval_mode"):
                env_u.set_eval_mode(False)

        return out

    @torch.no_grad()
    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - self.target_tau).add_(sp.data, alpha=self.target_tau)

    # ------------------------------------------------------------------
    # Diagnostic accumulation
    # ------------------------------------------------------------------

    def _diag_reset(self) -> None:
        self._diag = {}
        self._diag_count = 0

    def _diag_add(self, key: str, value: float) -> None:
        """Add one scalar observation; mean is computed at flush time."""
        if value is None:
            return
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return
        prev = self._diag.get(key)
        if prev is None:
            self._diag[key] = v
        else:
            # Running mean across calls within the same update() invocation.
            n = self._diag_count if self._diag_count > 0 else 1
            self._diag[key] = prev + (v - prev) / (n + 1)

    def _diag_bump(self) -> None:
        """Call once per mini-batch after all _diag_add calls for that batch."""
        self._diag_count += 1

    def _log_query_buckets(
        self,
        pred_U: torch.Tensor,       # [B, N]
        target_U: torch.Tensor,     # [B, N]
        immediate: torch.Tensor,    # [B, N]
        bootstrap_term: torch.Tensor,  # [B, N]  (γ·1[τ>1]·(1-done)·targ_U)
        taus: torch.Tensor,         # [B, N]
        kids: torch.Tensor,         # [B, N] long
        mask: torch.Tensor,         # [B, N] bool/float
    ) -> None:
        """Populate per-τ-bucket and per-keypoint diagnostics in self._diag."""
        pred_flat = pred_U.reshape(-1)
        targ_flat = target_U.reshape(-1)
        imm_flat = immediate.reshape(-1)
        boot_flat = bootstrap_term.reshape(-1)
        tau_flat = taus.reshape(-1)
        kid_flat = kids.reshape(-1)
        mask_flat = mask.reshape(-1).bool()

        if not mask_flat.any():
            return

        td = (pred_flat - targ_flat)[mask_flat]
        p = pred_flat[mask_flat]
        t = targ_flat[mask_flat]
        im = imm_flat[mask_flat]
        bo = boot_flat[mask_flat]
        tau_active = tau_flat[mask_flat]
        kid_active = kid_flat[mask_flat]

        # -- Per τ bucket --
        for lo, hi in self._tau_buckets:
            in_bucket = (tau_active >= lo) & (tau_active <= hi)
            if not in_bucket.any():
                continue
            name = f"tau_{lo:02d}_{hi:02d}"
            self._diag_add(f"QueryTau/{name}/count", float(in_bucket.sum().item()))
            self._diag_add(f"QueryTau/{name}/U_pred_mean", p[in_bucket].mean().item())
            self._diag_add(f"QueryTau/{name}/U_target_mean", t[in_bucket].mean().item())
            self._diag_add(f"QueryTau/{name}/td_error_mean", td[in_bucket].mean().item())
            self._diag_add(f"QueryTau/{name}/immediate_mean", im[in_bucket].mean().item())
            # "Success" proxy: immediate reward is exactly K_τ(1)·c(s_{t+1}, q);
            # at h=1 the kernel is 1.0 when τ=1 and decays fast, so immediate
            # ≈ satisfaction only for short τ. Normalise by the max possible
            # kernel value in the bucket so the mean stays 0..1 comparable.
            kernel_max = math.exp(-((1 - (lo + hi) / 2) ** 2) / (2 * self.sigma_time ** 2))
            self._diag_add(
                f"QueryTau/{name}/satisfaction_mean",
                (im[in_bucket].mean() / max(kernel_max, 1e-6)).item(),
            )

        # -- Per keypoint --
        nk = self.policy.num_keypoints
        for k in range(nk):
            in_k = (kid_active == k)
            if not in_k.any():
                continue
            self._diag_add(f"QueryKeypoint/{k:02d}/count", float(in_k.sum().item()))
            self._diag_add(f"QueryKeypoint/{k:02d}/U_pred_mean", p[in_k].mean().item())
            self._diag_add(f"QueryKeypoint/{k:02d}/U_target_mean", t[in_k].mean().item())
            self._diag_add(f"QueryKeypoint/{k:02d}/td_error_mean", td[in_k].mean().item())
            # per-keypoint beta varies; normalise immediate by this keypoint's
            # beta so the reported satisfaction is comparable.
            self._diag_add(f"QueryKeypoint/{k:02d}/immediate_mean", im[in_k].mean().item())

        # -- Global critic diagnostics --
        self._diag_add("Critic/U_pred_mean", p.mean().item())
        self._diag_add("Critic/U_pred_std", p.std().item())
        self._diag_add("Critic/U_target_mean", t.mean().item())
        self._diag_add("Critic/U_target_std", t.std().item())
        self._diag_add("Critic/U_td_error_mean", td.mean().item())
        self._diag_add("Critic/U_td_error_std", td.std().item())
        self._diag_add("Critic/U_immediate_mean", im.mean().item())
        self._diag_add("Critic/U_bootstrap_mean", bo.mean().item())
        # bootstrap ratio = bo / y  (clip for numerical stability)
        denom = (im + bo).abs().clamp(min=1e-4)
        ratio = bo.abs() / denom
        self._diag_add("Critic/U_bootstrap_ratio", ratio.mean().item())

    @staticmethod
    def _pessimistic_q(q1: torch.Tensor, q2: torch.Tensor, penalty: float) -> torch.Tensor:
        """Ensemble-pessimism over a twin critic: mean(Q) - penalty * |Q1 - Q2|.

        Reduces to ``min(Q1, Q2)`` when ``penalty == 0.5`` (identity with a 2-element ensemble).
        Larger penalties are stricter than min, smaller are looser.
        """
        if penalty == 0.0:
            return 0.5 * (q1 + q2)
        return 0.5 * (q1 + q2) - penalty * (q1 - q2).abs()

    # ------------------------------------------------------------------
    # Multi-GPU helpers
    # ------------------------------------------------------------------

    def broadcast_parameters(self):
        """Broadcast the full policy ``state_dict()`` from rank 0 to every
        other rank.

        ``state_dict()`` includes every parameter AND every registered
        buffer on the module tree, so this single call covers:
          - encoders (query_encoder, constraint_encoder)
          - actor + twin successor / style / aux critics (source + targets)
          - style discriminator
          - ``aux_reward_normalizer`` buffers (count / mean / M2)
        Called once at the start of training so every rank begins with an
        identical snapshot. Per-iter drift is then closed by
        ``reduce_gradients`` on optimised weights and by the runner's
        ``_sync_normalizer`` / ``aux_reward_normalizer.sync_across_ranks``
        calls for the non-gradient running stats.
        """
        if not self.is_multi_gpu or self.gpu_world_size <= 1:
            return
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
