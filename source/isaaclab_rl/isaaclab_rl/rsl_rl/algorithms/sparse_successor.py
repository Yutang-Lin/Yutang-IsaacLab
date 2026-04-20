# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from isaaclab_rl.rsl_rl.modules.sparse_successor_policy import SparseSuccessorPolicy
from isaaclab_rl.rsl_rl.storage.successor_storage import SuccessorStorage
from isaaclab_rl.rsl_rl.storage.expert_motion_buffer import ExpertMotionBuffer
from isaaclab_rl.rsl_rl.utils import reduce_gradients, zero_grads_if_nonfinite


def _flatten_xy(pos: torch.Tensor) -> torch.Tensor:
    """Return ``pos`` with its z component zeroed out.

    Used to define a "heading-anchor" translation that ignores height — so
    z stays absolute (ground-referenced) across the sample-time → world →
    current-local round trip. Without this, a robot that fell while
    sampling an expert motion would anchor the target near the ground
    rather than at the motion's original height.
    """
    flat = pos.clone()
    flat[..., 2] = 0.0
    return flat


def _ref_to_world(
    ref_xyz: torch.Tensor,      # [..., 3]
    anchor_pos: torch.Tensor,   # [..., 3]
    anchor_heading_quat: torch.Tensor,  # [..., 4] wxyz, yaw-only
) -> torch.Tensor:
    """Lift a de-yawed-root-frame point at the anchor time into the world.

    ``ref_xyz`` lives in the de-yawed root frame at the anchor moment (the
    heading is already factored out). We rotate it by the anchor's yaw
    quat and translate by the anchor's *xy* root position to get an
    absolute world-frame point that stays fixed as the robot moves — z
    is preserved as a ground-referenced height.
    """
    rotated = quat_apply(anchor_heading_quat, ref_xyz)
    return rotated + _flatten_xy(anchor_pos)


def _world_to_ref(
    world_xyz: torch.Tensor,      # [..., 3]
    anchor_pos: torch.Tensor,     # [..., 3]
    anchor_heading_quat: torch.Tensor,  # [..., 4] wxyz, yaw-only
) -> torch.Tensor:
    """Express a world-frame point in the anchor's de-yawed root frame.

    Inverse of :func:`_ref_to_world`: subtract only the anchor's xy (z
    stays absolute), then de-yaw.
    """
    translated = world_xyz - _flatten_xy(anchor_pos)
    return quat_apply_inverse(anchor_heading_quat, translated)


def _world_to_local(
    world_xyz: torch.Tensor,      # [..., 3]
    current_pos: torch.Tensor,    # [..., 3]
    current_heading_quat: torch.Tensor,  # [..., 4] wxyz, yaw-only
) -> torch.Tensor:
    """Express a world-frame point in the CURRENT de-yawed root frame.

    Same as :func:`_world_to_ref` but with the *current* env pose instead
    of the sample-time anchor. Called every step so the network sees a
    target consistent with the current priv keypoint frame.
    """
    translated = world_xyz - _flatten_xy(current_pos)
    return quat_apply_inverse(current_heading_quat, translated)


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
        # Actor-loss Q weights. When ``scale_lambda_by_q_track`` is True
        # (BFM-style adaptive scaling, default), the effective weight is
        # ``λ × |q_track|.abs().mean().detach()`` — so the style / aux
        # branches contribute on the same scale as the task Q no matter
        # how large the satisfaction rewards become. When False, the
        # λ's are absolute coefficients (legacy behaviour).
        # BFM reference: reg_coeff=0.05 (style), reg_coeff_aux=0.02.
        lambda_style: float = 0.05,
        lambda_aux: float = 0.02,
        scale_lambda_by_q_track: bool = True,
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
        # 2-way per-env source mixture for fresh rollout chunks.
        # Normalised to sum to 1 at runtime; missing sources (e.g. no
        # expert buffer, empty replay) fold their mass into the remaining
        # one. Both are per-atom future-grounded — no pose-hold-style
        # "live self" source any more.
        rollout_replay_fraction: float = 0.4,
        rollout_expert_fraction: float = 0.6,
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
        # Weight of the z_C orthonormality regulariser added to the
        # successor critic loss. Pushes ``z_C @ z_C.T`` toward the
        # identity so different atomic-constraint sets don't collapse
        # onto a low-dimensional subspace. BFM-Zero FB-CPR uses 100.0
        # here (on B-map output); ours is analogous but applied to the
        # sparse-constraint encoder's output. Set 0.0 to disable.
        ortho_coef: float = 100.0,
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
        self.scale_lambda_by_q_track = bool(scale_lambda_by_q_track)
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
        self.ortho_coef = float(ortho_coef)
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

    # ------------------------------------------------------------------
    # Anchor-frame helpers
    # ------------------------------------------------------------------

    def _current_env_anchor(
        self, env_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the env's live root position + yaw quat.

        Matches the exact frame that ``priv_keypoint_positions`` uses at
        obs time: ``body_pos[:, 0]`` (pelvis world pos) and
        ``yaw_quat(body_quat[:, 0])`` (pelvis heading). Consistency
        between what the anchor pipeline writes (``ξ_local``) and what
        the priv tail reads on the next step is what makes the reward
        computation geometrically valid.
        """
        env = self.unwrapped_env
        root_pos = env.body_pos[:, 0]                  # [N, 3]
        heading = yaw_quat(env.body_quat[:, 0])        # [N, 4]
        if env_mask is not None:
            return root_pos[env_mask], heading[env_mask]
        return root_pos, heading

    @torch.no_grad()
    def _refresh_local_targets(self) -> None:
        """Re-express every live atom's world-anchored target into the
        env's current de-yawed root frame.

        Called once per env step (from ``act``, before ``z_C`` is encoded)
        so the network and the satisfaction kernel always see a target
        expressed in the same frame as the current priv keypoint block.
        The world anchor (``targets_world``) is what stays constant as
        the body moves; ``targets`` is the view of that anchor from the
        current root frame — what the network consumes.
        """
        if self._env_constraints is None:
            return
        current_pos, current_heading = self._current_env_anchor()
        N, M = self._env_constraints["keypoint_ids"].shape
        world = self._env_constraints["targets_world"]    # [N, M, 3]
        # Broadcast the current anchor across the M atoms of each env.
        cur_pos = current_pos.unsqueeze(1).expand(N, M, 3)
        cur_heading = current_heading.unsqueeze(1).expand(N, M, 4)
        local = _world_to_local(world, cur_pos, cur_heading)
        self._env_constraints["targets"] = local

    @torch.no_grad()
    def initialize_constraints(self, priv_obs: torch.Tensor) -> None:
        """Allocate ``_env_constraints`` and draw a fresh set for every env.

        Called once by the runner right after the initial reset, before
        the first ``act()``. Also hands the storage a known fixed
        episode length when the env exposes one — the sparse-successor
        env has a hard 500-step timeout with no early termination, so
        every env resets in lock-step and the safe-anchor sampler can
        use that structural property instead of a per-env done scan.
        """
        num_envs = priv_obs.shape[0]
        if self._env_constraints is None or self._env_constraints["mask"].shape[0] != num_envs:
            self._init_constraints(num_envs)

        # Inform the storage about the env's episode alignment (if any).
        # The only envs we care about here timeout at a fixed length
        # and never terminate early; surfacing that to the storage
        # turns the O(T·N) safe-anchor scan into a closed-form filter.
        env = self.unwrapped_env
        episode_length = getattr(env, "max_episode_length", None)
        if self.storage is not None and episode_length is not None:
            self.storage.set_episode_alignment(
                int(episode_length), phase_offset=0,
            )

        all_envs = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        # Feed a reset_mask of all-True so every env is marked needs_new.
        self._advance_chunk(priv_obs, all_envs)
        self._refresh_local_targets()

    def _init_constraints(self, num_envs: int):
        M = self.policy.max_constraints
        td = self.policy.target_dim
        self._env_constraints = {
            "keypoint_ids": torch.zeros(num_envs, M, dtype=torch.long, device=self.device),
            # ``targets`` holds the CURRENT-STEP local (de-yawed-root) view
            # of the anchored world target. Recomputed every env step from
            # ``targets_world`` below so the network + satisfaction kernel
            # see a target expressed in the same frame as the priv
            # keypoint positions. What the actor / critic consume.
            "targets": torch.zeros(num_envs, M, td, device=self.device),
            # ``targets_world`` is the GLOBAL anchor: a fixed world-frame
            # 3D point per atom, set at chunk sample time, NEVER modified
            # until the atom expires. Re-expressed into the current
            # de-yawed root frame each step (into ``targets``) before
            # ``act`` / satisfaction reads it.
            "targets_world": torch.zeros(num_envs, M, td, device=self.device),
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
        """Copy-in ``new_constraints`` only for envs where ``env_mask`` is True.

        ``new_constraints`` must provide full ``[num_envs, M, …]`` tensors
        for keypoint_ids / taus / weights / mask, plus a ``targets_ref``
        tensor of the same row count expressed in the de-yawed-root
        frame at sample time. We lift ``targets_ref`` into a
        world-anchored ``targets_world`` using the env's CURRENT root
        pose (which is the sample-time anchor), and commit both that
        and ``targets = targets_ref`` (the local view is ref at sample
        instant) into the persistent per-env storage.

        ``_refresh_local_targets`` then updates ``targets`` every step
        by re-expressing ``targets_world`` in the new root frame.
        """
        if not env_mask.any():
            return
        expected = {"keypoint_ids", "taus", "weights", "mask", "targets_ref"}
        missing = expected - set(new_constraints.keys())
        if missing:
            raise KeyError(
                f"new_constraints is missing required keys: {sorted(missing)}"
            )

        env = self.unwrapped_env
        M = self.policy.max_constraints
        # Full-size anchor tensors; only the rows under env_mask matter.
        anchor_pos_full = env.body_pos[:, 0]                         # [N, 3]
        anchor_heading_full = yaw_quat(env.body_quat[:, 0])          # [N, 4]
        anchor_pos_m = anchor_pos_full.unsqueeze(1).expand(-1, M, 3)
        anchor_heading_m = anchor_heading_full.unsqueeze(1).expand(-1, M, 4)

        targets_ref_full = new_constraints["targets_ref"]            # [N, M, 3]
        targets_world_full = _ref_to_world(
            targets_ref_full, anchor_pos_m, anchor_heading_m,
        )

        # Apply to the selected rows only.
        for key in ("keypoint_ids", "taus", "weights", "mask"):
            self._env_constraints[key][env_mask] = new_constraints[key][env_mask]
        self._env_constraints["targets"][env_mask] = targets_ref_full[env_mask]
        self._env_constraints["targets_world"][env_mask] = targets_world_full[env_mask]

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
            # Per-env 2-way mixture for fresh rollout chunks:
            #   - replay-future (per-atom, real future from storage)
            #   - expert-future (per-atom, from the expert buffer)
            # Every atom is future-grounded; no "live/self" (pose-hold)
            # source. If neither branch is viable for a given env (e.g.
            # replay empty at warmup + no expert buffer), we preserve
            # that env's existing C and skip resampling — which keeps
            # the rollout pipeline running without ever producing a
            # degenerate zero-mask chunk.
            n_new = int(needs_new.sum().item())
            new_env_ids = needs_new.nonzero(as_tuple=True)[0]

            # Pre-draw a fresh random_C placeholder of the right shape;
            # we overwrite the rows per source. ``targets_ref`` is
            # zero-filled + mask zero so any row that remains untouched
            # (neither source viable) simply has mask=0 for every atom.
            M = self.policy.max_constraints
            td = self.policy.target_dim
            random_C = {
                "keypoint_ids": torch.zeros(num_envs, M, dtype=torch.long, device=self.device),
                "targets_ref": torch.zeros(num_envs, M, td, device=self.device),
                "taus": torch.ones(num_envs, M, device=self.device),
                "weights": torch.ones(num_envs, M, device=self.device),
                "mask": torch.zeros(num_envs, M, device=self.device),
            }

            # Can we draw safe replay anchors? Requires enough populated
            # transitions AND at least one safe-anchor position exists.
            replay_ready = (
                self.storage is not None
                and (
                    self.storage._full
                    or self.storage.step >= self.tau_max + 1
                )
            )
            p_replay = float(self.rollout_replay_fraction) if replay_ready else 0.0
            p_expert = float(self.rollout_expert_fraction) if self.expert_buffer is not None else 0.0
            total_p = p_replay + p_expert
            if total_p <= 0.0:
                # Neither source available — leave ``needs_new`` envs
                # with their previous (possibly all-expired) constraint
                # set; the next step will try again. Should only happen
                # during the very first iters before warmup fills replay
                # when there's also no expert buffer, which is an
                # unsupported config but we fail gracefully.
                self._diag_rollout_source_replay = 0.0
                self._diag_rollout_source_expert = 0.0
                self._diag_rollout_fresh_frac = float(needs_new.float().mean().item())
                return
            p_replay /= total_p
            p_expert /= total_p
            probs = torch.tensor([p_replay, p_expert], device=self.device)
            source = torch.multinomial(probs, n_new, replacement=True)   # {0, 1}

            # --- Replay-future source (per-atom, same-episode-only) ---
            # ``sample_safe_future_anchors`` only returns (t, env) pairs
            # whose full tau_max-window is populated AND reset-free.
            # If it returns None (extreme edge case), fold replay mass
            # into expert for this step.
            replay_mask_of_new = source == 0
            if replay_mask_of_new.any():
                n_replay = int(replay_mask_of_new.sum().item())
                anchors = self.storage.sample_safe_future_anchors(
                    n_replay, horizon=self.tau_max,
                )
                if anchors is None:
                    # No safe anchor exists in replay right now — fall
                    # through: these envs will get the expert source if
                    # available, else leave their row unmasked (no C).
                    if p_expert > 0.0:
                        source[replay_mask_of_new] = 1  # reroute to expert
                        replay_mask_of_new = source == 0
                    else:
                        source[replay_mask_of_new] = -1  # dead; row keeps mask=0
                        replay_mask_of_new = source == 0
                else:
                    t_anchor, env_anchor = anchors
                    priv_window, valid = self.storage.gather_next_priv_at(
                        t_anchor, env_anchor, horizon=self.tau_max,
                    )
                    priv_window = priv_window.to(self.device)
                    valid = valid.to(self.device)
                    nk = self.policy.num_keypoints
                    priv_dim = priv_window.shape[-1]
                    kp_offset = priv_dim - nk * td
                    kp_window = priv_window[..., kp_offset:].reshape(
                        priv_window.shape[0], priv_window.shape[1], nk, td,
                    )                                                   # [n_replay, H+1, K, 3]
                    # Priv tail is already de-yawed per its own frame →
                    # trajectory anchor = identity.
                    traj_pos = torch.zeros(n_replay, 3, device=self.device)
                    traj_quat = torch.zeros(n_replay, 4, device=self.device)
                    traj_quat[:, 0] = 1.0
                    replay_C = self._sample_constraints_from_keypoint_future(
                        kp_window, traj_pos, traj_quat, valid_atom_mask=valid,
                    )
                    replay_env_ids = new_env_ids[replay_mask_of_new]
                    for key, rv in replay_C.items():
                        random_C[key][replay_env_ids] = rv

            # --- Expert-future source ---
            expert_mask_of_new = source == 1
            if expert_mask_of_new.any() and self.expert_buffer is not None:
                n_expert = int(expert_mask_of_new.sum().item())
                expert_batch = self.expert_buffer.sample_with_future_window(
                    n_expert, horizon=self.tau_max,
                )
                expert_window = expert_batch["kp_window"].to(self.device)
                expert_anchor_pos = expert_batch["anchor_root_pos"].to(self.device)
                expert_anchor_quat = expert_batch["anchor_root_quat"].to(self.device)
                expert_C = self._sample_constraints_from_keypoint_future(
                    expert_window, expert_anchor_pos, expert_anchor_quat,
                )
                expert_env_ids = new_env_ids[expert_mask_of_new]
                for key, ev in expert_C.items():
                    random_C[key][expert_env_ids] = ev

            # Diagnostic source fractions.
            self._diag_rollout_source_replay = float((source == 0).float().mean().item())
            self._diag_rollout_source_expert = float((source == 1).float().mean().item())
            # Envs whose replay source aborted without an expert fallback
            # show up as source == -1; we count them separately.
            self._diag_rollout_source_dead = float((source == -1).float().mean().item())

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

    @torch.no_grad()
    def _relabel_constraint_sets(
        self,
        stored: dict[str, torch.Tensor],
        next_priv: torch.Tensor,
        t_idx: torch.Tensor | None = None,
        env_idx: torch.Tensor | None = None,
        stored_root_pos: torch.Tensor | None = None,
        stored_root_quat: torch.Tensor | None = None,
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
        # Each row's anchor = the transition's own (t_idx, env_idx).
        # The priv tail at THAT step is in the stored step's de-yawed
        # root frame, and the replay gather returns it de-yawed already
        # (because each priv frame's tail is self-referential). So the
        # trajectory anchor passed to the sampler is identity, and the
        # resulting ``targets_ref`` lands in the stored step's root
        # frame — same frame as the stored priv the critic will read.
        if hind_mask.any() and t_idx is not None and env_idx is not None:
            idxs = hind_mask.nonzero(as_tuple=True)[0]
            n_hind = idxs.numel()
            priv_window, valid = self.storage.gather_next_priv_at(
                t_idx[idxs.to(t_idx.device)],
                env_idx[idxs.to(env_idx.device)],
                horizon=self.tau_max,
            )
            priv_window = priv_window.to(self.device)
            valid = valid.to(self.device)
            priv_dim = priv_window.shape[-1]
            kp_offset = priv_dim - nk * td
            kp_window = priv_window[..., kp_offset:].reshape(
                priv_window.shape[0], priv_window.shape[1], nk, td,
            )                                                               # [n_hind, H+1, K, 3]
            # Identity anchor: each priv frame's keypoint tail is already
            # de-yawed w.r.t. the body at that same frame. Passing a zero
            # translation + identity quat makes ``_world_to_ref`` a no-op.
            traj_pos = torch.zeros(n_hind, 3, device=self.device)
            traj_quat = torch.zeros(n_hind, 4, device=self.device)
            traj_quat[:, 0] = 1.0
            hind_C = self._sample_constraints_from_keypoint_future(
                kp_window, traj_pos, traj_quat, valid_atom_mask=valid,
            )
            # ``targets_ref`` IS the local target in the stored step's
            # root frame. ``relabeled["targets"]`` is what the critic
            # will read, so we write targets_ref there directly (no
            # world-lift at training time — the stored priv's frame is
            # the relabel frame).
            relabeled["keypoint_ids"][idxs] = hind_C["keypoint_ids"]
            relabeled["targets"][idxs] = hind_C["targets_ref"]
            relabeled["taus"][idxs] = hind_C["taus"]
            relabeled["weights"][idxs] = hind_C["weights"]
            relabeled["mask"][idxs] = hind_C["mask"]

        # --- Expert source (per-atom future-grounded, stored-frame-aligned) ---
        # The sampler returns ``targets_ref`` in the EXPERT's de-yawed
        # root frame at t0. But the critic reads the mini-batch's
        # STORED priv — whose keypoint tail is in the STORED step's
        # de-yawed root frame. To make the satisfaction kernel
        # frame-consistent, we lift the expert ref into world using
        # the expert's own anchor, then re-express in the stored
        # step's anchor. This is the exact mirror of what
        # ``_replace_constraints_for_envs`` does at rollout time, just
        # applied post-hoc at the sampled mini-batch's timestamp.
        if expert_mask.any() and self.expert_buffer is not None:
            if stored_root_pos is None or stored_root_quat is None:
                raise RuntimeError(
                    "expert-relabel requires stored_root_pos / stored_root_quat "
                    "so the expert target frame can be aligned with the stored "
                    "priv frame. The storage must populate these at rollout time."
                )
            idxs = expert_mask.nonzero(as_tuple=True)[0]
            n_exp = idxs.numel()
            expert_batch = self.expert_buffer.sample_with_future_window(
                n_exp, horizon=self.tau_max,
            )
            expert_window = expert_batch["kp_window"].to(self.device)
            expert_anchor_pos = expert_batch["anchor_root_pos"].to(self.device)
            expert_anchor_quat = expert_batch["anchor_root_quat"].to(self.device)
            expert_C = self._sample_constraints_from_keypoint_future(
                expert_window, expert_anchor_pos, expert_anchor_quat,
            )
            targets_ref_expert = expert_C["targets_ref"]             # [n_exp, M, 3]

            # Lift expert-frame ref → world, using the expert's own
            # yaw-only quat (so the ``_ref_to_world`` helper treats it
            # as a heading anchor the way our per-env lifts do).
            expert_yaw = yaw_quat(expert_anchor_quat)                # [n_exp, 4]
            expert_yaw_m = expert_yaw.unsqueeze(1).expand(-1, M, 4)
            expert_pos_m = expert_anchor_pos.unsqueeze(1).expand(-1, M, 3)
            targets_world_expert = _ref_to_world(
                targets_ref_expert, expert_pos_m, expert_yaw_m,
            )
            # Re-express in the stored step's de-yawed root frame.
            stored_pos_sel = stored_root_pos[idxs]                   # [n_exp, 3]
            stored_yaw_sel = yaw_quat(stored_root_quat[idxs])        # [n_exp, 4]
            stored_pos_m = stored_pos_sel.unsqueeze(1).expand(-1, M, 3)
            stored_yaw_m = stored_yaw_sel.unsqueeze(1).expand(-1, M, 4)
            targets_local_stored = _world_to_local(
                targets_world_expert, stored_pos_m, stored_yaw_m,
            )

            relabeled["keypoint_ids"][idxs] = expert_C["keypoint_ids"]
            relabeled["targets"][idxs] = targets_local_stored
            relabeled["taus"][idxs] = expert_C["taus"]
            relabeled["weights"][idxs] = expert_C["weights"]
            relabeled["mask"][idxs] = expert_C["mask"]

        counts = {
            "stored": int(stored_mask.sum().item()),
            "hindsight": int(hind_mask.sum().item()),
            "expert": int(expert_mask.sum().item()),
        }
        return relabeled, counts

    def _sample_constraints_from_keypoint_future(
        self,
        kp_window: torch.Tensor,
        traj_anchor_root_pos: torch.Tensor,
        traj_anchor_root_quat: torch.Tensor,
        valid_atom_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-atom future-grounded constraint sampling, anchored in the
        *trajectory's* de-yawed root frame.

        For each element b in the batch:
          - sample n_active atoms,
          - for each atomic slot i: sample ``k_i``, ``τ_i``, ``w_i``,
          - pick the keypoint's world position at ``t + τ_i``,
          - express it in the trajectory's de-yawed root frame at ``t0``:
              ξ_ref_i = R_yaw(traj_quat_t0)^-1 · (p_world − traj_pos_t0_xy)
          (z is preserved as absolute / ground-referenced — not anchored
           to the trajectory's pelvis height, so recovery poses from
           lie-down starts stay consistent with the expert-stand target).

        The caller is responsible for lifting ``targets_ref`` into a
        world-anchored ``targets_world`` using the **env's** sample-time
        root pose, which is how we separate the trajectory source (which
        describes relative motion) from the env instance (which decides
        where in space the motion is grounded).

        Args:
            kp_window: [B, H+1, K, 3] world-frame keypoint positions.
                Index 0 = anchor frame, index h>=1 = frame ``t + h``.
            traj_anchor_root_pos: [B, 3] world-frame root pos at the
                trajectory's own anchor frame.
            traj_anchor_root_quat: [B, 4] world-frame root quat (wxyz)
                at the trajectory's own anchor frame.
            valid_atom_mask: optional [B, H+1] bool. When provided, atoms
                landing on ``False`` frames (e.g. reset crossings in
                replay hindsight) are masked out.

        Returns:
            dict with keypoint_ids / taus / weights / mask and
            ``targets_ref`` [B, M, 3] — the de-yawed-root-frame target at
            the trajectory's anchor time.
        """
        B, Hp1, K, td = kp_window.shape
        H = Hp1 - 1
        M = self.policy.max_constraints
        if K != self.policy.num_keypoints:
            raise ValueError(
                f"kp_window has {K} keypoints but policy expects "
                f"{self.policy.num_keypoints}"
            )
        if td != self.policy.target_dim:
            raise ValueError(
                f"kp_window target_dim {td} does not match policy target_dim"
                f" {self.policy.target_dim}"
            )
        if H < 1:
            raise ValueError("future window must have at least one lookahead frame")
        if traj_anchor_root_pos.shape != (B, 3):
            raise ValueError(
                f"traj_anchor_root_pos shape {tuple(traj_anchor_root_pos.shape)}"
                f" != ({B}, 3)"
            )
        if traj_anchor_root_quat.shape != (B, 4):
            raise ValueError(
                f"traj_anchor_root_quat shape {tuple(traj_anchor_root_quat.shape)}"
                f" != ({B}, 4)"
            )

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

        # Gather per-atom world-frame target at (b, τ_i, k_i).
        b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, M)  # [B, M]
        tau_idx = taus.long().clamp(0, H)                                       # [B, M]
        k_idx = keypoint_ids.clamp(0, K - 1).long()                             # [B, M]
        targets_world_local_traj = kp_window[b_idx, tau_idx, k_idx]             # [B, M, 3]

        # Express each target in the TRAJECTORY'S de-yawed root frame at
        # its anchor time. z is preserved as absolute via ``_world_to_ref``
        # (which zeroes the anchor's z before the translation).
        traj_yaw = yaw_quat(traj_anchor_root_quat)                              # [B, 4]
        traj_yaw_m = traj_yaw.unsqueeze(1).expand(B, M, 4)
        traj_pos_m = traj_anchor_root_pos.unsqueeze(1).expand(B, M, 3)
        targets_ref = _world_to_ref(
            targets_world_local_traj, traj_pos_m, traj_yaw_m,
        )
        targets_ref = targets_ref + torch.randn_like(targets_ref) * self.target_noise_std

        # Apply dropout.
        dropout_mask = (torch.rand(B, M, device=self.device) > self.constraint_dropout_prob).float()
        mask = mask * dropout_mask

        # Mask atoms that landed on an invalid frame (e.g. reset crossing
        # in hindsight replay).
        if valid_atom_mask is not None:
            v = valid_atom_mask[b_idx, tau_idx].float()                         # [B, M]
            mask = mask * v

        return {
            "keypoint_ids": keypoint_ids,
            "targets_ref": targets_ref,
            "taus": taus,
            "weights": weights,
            "mask": mask,
        }

    def _sample_constraints_from_keypoint_pos(
        self,
        keypoint_pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Single-frame (non-future-grounded) constraint sampling.

        **Single-frame, trajectory-anchor-free variant** kept for callers
        that already have keypoint positions expressed in the desired
        de-yawed root frame (i.e. the priv tail, which IS de-yawed). The
        per-atom construction still samples an independent ``τ_i`` but
        the spatial target is a shared current-frame keypoint — so
        this sampler is **not** future-grounded; it's the pose-hold-style
        live-self source. Per-atom future-grounded construction from a
        trajectory window lives in
        :meth:`_sample_constraints_from_keypoint_future`.

        Args:
            keypoint_pos: [B, K, 3] — keypoints already in the desired
                de-yawed root frame (i.e. ``priv_keypoint_positions``).
        Returns:
            dict with keypoint_ids / taus / weights / mask and
            ``targets_ref`` [B, M, 3] — de-yawed-root-frame target (the
            caller lifts it to ``targets_world`` with the env's
            sample-time anchor).
        """
        B, K, td = keypoint_pos.shape
        M = self.policy.max_constraints
        if K != self.policy.num_keypoints:
            raise ValueError(
                f"keypoint_pos has {K} keypoints but policy expects "
                f"{self.policy.num_keypoints}"
            )
        if td != self.policy.target_dim:
            raise ValueError(
                f"keypoint_pos target_dim {td} does not match policy target_dim"
                f" {self.policy.target_dim}"
            )

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
        targets_ref = torch.gather(keypoint_pos, 1, kid_expanded)
        targets_ref = targets_ref + torch.randn_like(targets_ref) * self.target_noise_std

        dropout_mask = (torch.rand(B, M, device=self.device) > self.constraint_dropout_prob).float()
        mask = mask * dropout_mask

        return {
            "keypoint_ids": keypoint_ids,
            "targets_ref": targets_ref,
            "taus": taus,
            "weights": weights,
            "mask": mask,
        }

    def sample_constraint_set_vectorized(
        self,
        priv_state: torch.Tensor,
        num_envs: int,
    ) -> dict[str, torch.Tensor]:
        """Single-frame constraint sampler. Safety fallback only.

        Used solely by ``evaluate_live_tracking`` when the algorithm has
        no expert buffer — in that case the eval can't produce a
        per-atom future-grounded C and falls back to "target = current
        body priv, per-atom random τ". The rollout chunk pipeline no
        longer uses this path; all rollout fresh chunks come from
        replay-future or expert-future sources, both per-atom
        future-grounded via :meth:`_sample_constraints_from_keypoint_future`.

        ``targets_ref`` is the env's current priv tail — already
        de-yawed w.r.t. the current body — so the caller's lift to
        world via ``_ref_to_world`` produces a target at the env's
        current pelvis, with no temporal separation. The "task" is
        degenerate (hit where you already are), which is acceptable
        for the tiny eval-no-expert-buffer corner case.
        """
        M = self.policy.max_constraints
        td = self.policy.target_dim
        nk = self.policy.num_keypoints

        n_per_env = torch.randint(
            self.n_constraints_min, min(self.n_constraints_max, M) + 1,
            (num_envs,), device=self.device,
        )
        arange = torch.arange(M, device=self.device).unsqueeze(0).expand(num_envs, -1)
        mask = (arange < n_per_env.unsqueeze(1)).float()

        keypoint_ids = torch.randint(0, nk, (num_envs, M), device=self.device)
        taus = torch.randint(1, self.tau_max + 1, (num_envs, M), device=self.device).float()
        w_min, w_max = self.weight_range
        weights = torch.empty(num_envs, M, device=self.device).uniform_(w_min, w_max)

        # priv tail is already in the env's de-yawed root frame at the
        # current step; we reuse it as ``targets_ref`` directly.
        offset = priv_state.shape[-1] - nk * td
        kp_block = priv_state[:, offset:].reshape(num_envs, nk, td)           # [N, nk, td]
        kid_expanded = keypoint_ids.unsqueeze(-1).expand(-1, -1, td)          # [N, M, td]
        targets_ref = torch.gather(kp_block, 1, kid_expanded.clamp(0, nk - 1))
        targets_ref = targets_ref + torch.randn_like(targets_ref) * self.target_noise_std

        dropout_mask = (torch.rand(num_envs, M, device=self.device) > self.constraint_dropout_prob).float()
        mask = mask * dropout_mask

        return {
            "keypoint_ids": keypoint_ids,
            "targets_ref": targets_ref,
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

        # Re-express the world-anchored target into the env's CURRENT
        # de-yawed root frame. ``targets`` must match the frame of the
        # priv keypoint block at this exact step for the satisfaction
        # kernel and the per-atom encoder to be geometrically valid.
        self._refresh_local_targets()

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

        # Persist the env's root pose at THIS step so training-time
        # expert-relabel can re-express the expert's world-anchored
        # target into this step's de-yawed root frame. Without this,
        # relabeled targets land in a different frame than the stored
        # priv and the satisfaction kernel becomes frame-inconsistent.
        env = self.unwrapped_env
        self.transition.root_pos = env.body_pos[:, 0].clone()
        self.transition.root_quat = env.body_quat[:, 0].clone()

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
        # Fresh chunks are sampled against the env's CURRENT root pose
        # (which is the post-step state at t+1); for continuing chunks
        # the world anchor is unchanged but the local view needs to be
        # refreshed to the t+1 frame before being stored as
        # ``next_constraint_targets``.
        self._advance_chunk(next_priv_obs, reset_mask)
        self._refresh_local_targets()

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
                stored_root_pos, stored_root_quat,
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
                stored_root_pos=stored_root_pos,
                stored_root_quat=stored_root_quat,
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
            # Matches BFM's Discriminator.compute_reward default.
            eps = 1e-7
            disc_loss = None
            grad_penalty_val = 0.0
            if self.expert_buffer is not None:
                # Per-atom future-grounded expert z: snippet + kp_window, each
                # atomic constraint samples its own τ_i from [1, tau_max] and
                # draws its target from the corresponding future frame. Note
                # scheme (B): we do NOT require the snippet window to cover
                # every τ_i — the disc only sees the snippet as evidence of
                # per-step style, while the z_C it's paired with carries
                # multi-time query information the snippet cannot fully
                # verify. With the current SNIPPET_LENGTH=1 config the
                # disc is a per-frame judge, matching BFM-Zero.
                expert_batch = self.expert_buffer.sample_with_future_window(
                    snippets.shape[0], horizon=self.tau_max,
                )
                expert_snippets = expert_batch["snippet"].to(self.device)    # [B, L*style_dim]
                expert_window = expert_batch["kp_window"].to(self.device)    # [B, H+1, K, 3]
                expert_anchor_pos = expert_batch["anchor_root_pos"].to(self.device)
                expert_anchor_quat = expert_batch["anchor_root_quat"].to(self.device)
                expert_constraints = self._sample_constraints_from_keypoint_future(
                    expert_window, expert_anchor_pos, expert_anchor_quat,
                )
                with torch.no_grad():
                    # The expert_z is paired with the expert snippet — both
                    # live in the expert's own de-yawed root frame at t0.
                    # ``targets_ref`` IS the local view in that frame, so
                    # it's what the encoder needs.
                    expert_z = self.policy.encode_constraint_set(
                        expert_constraints["keypoint_ids"],
                        expert_constraints["targets_ref"],
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
                # Local NaN/Inf guard — zero this rank's grads if the loss
                # exploded, but STILL call reduce_gradients so the DDP
                # collective never deadlocks. Other ranks' finite grads
                # contribute; this rank contributes zero.
                nan_disc = zero_grads_if_nonfinite(
                    disc_loss, self.policy.style_discriminator,
                )
                if nan_disc:
                    self._diag_add("NaN/disc_skip", 1.0)
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

            # z_C orthonormality regulariser (BFM's ``ortho_coef`` applied to
            # B(obs).BT). Push the Gram matrix of the batch z_C rows toward
            # the identity:
            #   diag  → 1      (unit norm per row — redundant under the
            #                   unit-sphere projection, but cheap and keeps
            #                   parity with BFM's diag_term)
            #   off   → 0      (decorrelate z_C across the batch to prevent
            #                   collapse to a rank-1 subspace)
            # Gradient flows through z_C into QueryEncoder (the only
            # trainable component inside ``ConstraintSetEncoder`` now that
            # ``post_mlp`` is removed), which is exactly what we want —
            # this is the encoder-side regulariser analogous to BFM's.
            if self.ortho_coef > 0.0:
                # Normalise z_C per-row so diag_term isn't dominated by the
                # unit-sphere scaling constant (||z|| = sqrt(d_model)).
                z_normed = z_C / z_C.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                cov = z_normed @ z_normed.transpose(0, 1)                      # [B, B]
                B_size = cov.shape[0]
                off_diag = 1.0 - torch.eye(B_size, device=cov.device)
                off_diag_sum = off_diag.sum().clamp(min=1.0)
                orth_loss_offdiag = 0.5 * (cov * off_diag).pow(2).sum() / off_diag_sum
                orth_loss_diag = -cov.diag().mean()
                orth_loss = orth_loss_offdiag + orth_loss_diag
                loss_U = loss_U + self.ortho_coef * orth_loss
                self._diag_add("Loss/z_C_ortho", float(orth_loss.item()))
                self._diag_add("Loss/z_C_ortho_offdiag", float(orth_loss_offdiag.item()))
                self._diag_add("Loss/z_C_ortho_diag", float(orth_loss_diag.item()))

            self.opt_query.zero_grad()
            self.opt_U1.zero_grad()
            self.opt_U2.zero_grad()
            loss_U.backward()
            nan_U = zero_grads_if_nonfinite(
                loss_U,
                self.policy.query_encoder,
                self.policy.successor_critic_1,
                self.policy.successor_critic_2,
            )
            if nan_U:
                self._diag_add("NaN/U_skip", 1.0)
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
                if zero_grads_if_nonfinite(loss_QS1, self.policy.style_critic_1):
                    self._diag_add("NaN/QS1_skip", 1.0)
                if self._sync_grads:
                    reduce_gradients(self.policy.style_critic_1)
                nn.utils.clip_grad_norm_(self.policy.style_critic_1.parameters(), self.max_grad_norm)
                self.opt_QS1.step()

                self.opt_QS2.zero_grad()
                loss_QS2.backward()
                if zero_grads_if_nonfinite(loss_QS2, self.policy.style_critic_2):
                    self._diag_add("NaN/QS2_skip", 1.0)
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
                # Update the running stats from this batch AND immediately
                # merge across DDP ranks so r_env_norm below uses the global
                # Welford statistics. Without this sync, each rank's 16
                # inner updates would each normalize against its own drifting
                # local stats — producing inconsistent TD targets that
                # reduce_gradients then averages into a meaningless signal.
                # This is the "every stat that touches a loss must be
                # synced every inner step" invariant.
                self.policy.aux_reward_normalizer.update(rewards_flat)
                if self._sync_grads and hasattr(
                    self.policy.aux_reward_normalizer, "sync_across_ranks"
                ):
                    self.policy.aux_reward_normalizer.sync_across_ranks(self.device)
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
                if zero_grads_if_nonfinite(loss_QA1, self.policy.aux_critic_1):
                    self._diag_add("NaN/QA1_skip", 1.0)
                if self._sync_grads:
                    reduce_gradients(self.policy.aux_critic_1)
                nn.utils.clip_grad_norm_(self.policy.aux_critic_1.parameters(), self.max_grad_norm)
                self.opt_QA1.step()

                self.opt_QA2.zero_grad()
                loss_QA2.backward()
                if zero_grads_if_nonfinite(loss_QA2, self.policy.aux_critic_2):
                    self._diag_add("NaN/QA2_skip", 1.0)
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

            # Adaptive Q-scale for style/aux branches (BFM-style
            # ``scale_reg``). The ``|q_track|.abs().mean().detach()``
            # factor keeps the style/aux contributions proportionate to
            # the task Q no matter how the satisfaction-reward scale
            # drifts during training. Detached so the actor-loss
            # gradient doesn't try to shrink q_track just to down-weight
            # style/aux. ``scale_lambda_by_q_track=False`` recovers the
            # fixed-coefficient behaviour.
            if self.scale_lambda_by_q_track:
                lambda_scale = q_track.abs().mean().detach().clamp(min=1e-6)
            else:
                lambda_scale = torch.tensor(1.0, device=q_track.device)

            if self.expert_buffer is not None and self.lambda_style > 0.0:
                q_style = self._pessimistic_q(
                    self.policy.style_critic_1(obs, priv, new_action, z_C_actor),
                    self.policy.style_critic_2(obs, priv, new_action, z_C_actor),
                    self.actor_pessimism_penalty,
                )
                q_total = q_total + self.lambda_style * lambda_scale * q_style
            else:
                q_style = None

            if self.lambda_aux > 0.0:
                q_aux = self._pessimistic_q(
                    self.policy.aux_critic_1(obs, priv, new_action, z_C_actor),
                    self.policy.aux_critic_2(obs, priv, new_action, z_C_actor),
                    self.actor_pessimism_penalty,
                )
                q_total = q_total + self.lambda_aux * lambda_scale * q_aux
            else:
                q_aux = None

            # Record the effective adaptive weight for diagnostics.
            self._diag_add("Scale/lambda_scale", float(lambda_scale.item()))

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
                        (self.lambda_style * lambda_scale * q_style).mean().item(),
                    )
                if q_aux is not None:
                    self._diag_add("Scale/q_aux_mean", q_aux.mean().item())
                    self._diag_add("Scale/q_aux_std", q_aux.std().item())
                    self._diag_add(
                        "Scale/lambda_aux_times_q_aux_mean",
                        (self.lambda_aux * lambda_scale * q_aux).mean().item(),
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
            if zero_grads_if_nonfinite(loss_actor, self.policy.actor):
                self._diag_add("NaN/actor_skip", 1.0)
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
            # ``_advance_chunk`` call with fresh envs). Fractions sum
            # to ~1 across the two live sources (replay + expert). The
            # ``dead`` counter is fresh envs that couldn't be assigned
            # to any source that step — non-zero typically means replay
            # has no safe anchors AND no expert buffer; those envs keep
            # their previous C for one more step.
            for key in (
                "_diag_rollout_source_replay",
                "_diag_rollout_source_expert",
                "_diag_rollout_source_dead",
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
            #    the expert buffer when available. Lift ``targets_ref``
            #    (de-yawed-root at the expert's own t0) into world-anchored
            #    ``targets_world`` using the env's post-reset root pose —
            #    this is the sample-time anchor, mirroring what
            #    ``_replace_constraints_for_envs`` does at rollout time.
            #    When no expert buffer exists, fall back to the single-
            #    frame self-priv sampler anchored the same way.
            if self.expert_buffer is not None:
                expert_batch = self.expert_buffer.sample_with_future_window(
                    num_envs, horizon=self.tau_max,
                )
                kp_window = expert_batch["kp_window"].to(device)
                expert_anchor_pos = expert_batch["anchor_root_pos"].to(device)
                expert_anchor_quat = expert_batch["anchor_root_quat"].to(device)
                C = self._sample_constraints_from_keypoint_future(
                    kp_window, expert_anchor_pos, expert_anchor_quat,
                )
            else:
                C = self.sample_constraint_set_vectorized(priv_obs, num_envs)

            # In eval we want the *declared* per-query τ to be respected
            # exactly, so drop the random dropout that the sampler applies.
            mask = (C["mask"] > 0).float()

            kid = C["keypoint_ids"].long()            # [N, M]
            targets_ref = C["targets_ref"]            # [N, M, 3] de-yawed at traj t0
            tau = C["taus"].long()                    # [N, M]
            mask_b = mask.bool()                      # [N, M]

            # Lift into world frame using the env's CURRENT root pose
            # (just reset, so this is the eval sample-time anchor).
            env_u_local = env.unwrapped if hasattr(env, "unwrapped") else env
            M = kid.shape[1]
            env_anchor_pos = env_u_local.body_pos[:, 0]                   # [N, 3]
            env_anchor_heading = yaw_quat(env_u_local.body_quat[:, 0])    # [N, 4]
            env_anchor_pos_m = env_anchor_pos.unsqueeze(1).expand(-1, M, 3)
            env_anchor_heading_m = env_anchor_heading.unsqueeze(1).expand(-1, M, 4)
            targets_world = _ref_to_world(
                targets_ref, env_anchor_pos_m, env_anchor_heading_m,
            )
            # At sample time, the local view == targets_ref by construction.
            targets_local = targets_ref.clone()

            # Encode z_C once at the start — the underlying world anchor
            # is fixed, but the local view passed to the encoder must
            # match the current root frame. We re-encode every step
            # inside the rollout loop below; this initial encode is
            # just so the first ``act_inference`` has a valid latent.
            z_C = self.policy.encode_constraint_set(
                kid, targets_local, tau.float(), C["weights"], mask,
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

                # Re-express the world-anchored target in the env's
                # NEW de-yawed root frame — same frame the priv tail
                # we're about to read is in. Without this the realized-
                # vs-target comparison below would mix frames.
                cur_pos = env_u_local.body_pos[:, 0]
                cur_heading = yaw_quat(env_u_local.body_quat[:, 0])
                cur_pos_m = cur_pos.unsqueeze(1).expand(-1, M, 3)
                cur_heading_m = cur_heading.unsqueeze(1).expand(-1, M, 4)
                targets_local = _world_to_local(
                    targets_world, cur_pos_m, cur_heading_m,
                )
                # Re-encode z_C so the actor's next step sees the
                # up-to-date local-frame target (the world target hasn't
                # moved, but the local view of it has).
                z_C = self.policy.encode_constraint_set(
                    kid, targets_local, tau.float(), C["weights"], mask,
                )

                # Extract realized keypoint positions from the priv tail.
                priv_dim = priv_next.shape[-1]
                kp_offset = priv_dim - nk * td
                kp_block = priv_next[:, kp_offset:].reshape(num_envs, nk, td)   # [N, nk, td]

                # For every query q=(k, ξ, τ), if τ == step+1 *and* the env
                # is still alive, record the realized keypoint for that q.
                this_step = (tau == (step + 1)) & mask_b & alive.unsqueeze(-1)   # [N, M]
                if this_step.any():
                    kid_exp = kid.unsqueeze(-1).expand(-1, -1, td).clamp(0, nk - 1)
                    realized = torch.gather(
                        kp_block.unsqueeze(1).expand(-1, M, -1, -1),
                        -2,
                        kid_exp.unsqueeze(-2),
                    ).squeeze(-2)                                           # [N, M, td]
                    # Both realized and targets_local are in the current
                    # env de-yawed root frame — valid to subtract.
                    err = (realized - targets_local).norm(dim=-1)
                    beta_q = beta[kid.clamp(0, nk - 1)]
                    err_norm = err / beta_q
                    err_hist[..., step][this_step] = err_norm[this_step]
                    seen_hist[..., step][this_step] = 1.0

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
