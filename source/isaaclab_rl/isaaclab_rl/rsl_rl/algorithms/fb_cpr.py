# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FB-CPR-Aux algorithm — faithful port of BFM-Zero's ``FBcprAuxAgent``.

This file replicates BFM-Zero's `humanoidverse/agents/fb_cpr_aux/agent.py`
(itself layered over `fb_cpr/agent.py` and `fb/agent.py`) inside our
rsl_rl conventions. It consumes the networks and helpers from
:mod:`isaaclab_rl.rsl_rl.modules.fb_cpr_policy` and the replay+expert
buffers from :mod:`isaaclab_rl.rsl_rl.storage.fb_cpr_storage`.

Update order each step (must match BFM-Zero exactly):
    1. discriminator (with WGAN-GP)
    2. sample mixed z + optional relabel
    3. forward-backward (F/B) update
    4. critic update (twin Q on discriminator log-odds reward)
    5. aux-critic update (twin Q on scaled+normalized env aux rewards)
    6. actor update (combines Q_fb + reg_coeff*Q_disc*|Q_fb| + reg_coeff_aux*Q_aux*|Q_fb|)
    7. Polyak soft updates on target nets

The agent owns its own optimizers and the target networks inside its
:class:`~isaaclab_rl.rsl_rl.modules.fb_cpr_policy.FBCprAuxPolicy`. The
outer runner drives the rollout → replay → update loop.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import field
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import autograd

from isaaclab.utils import configclass

from ..modules.fb_cpr_policy import (
    FBCprAuxPolicy,
    FBCprNetworkCfg,
    _soft_update_params,
    eval_mode,
    weight_init,
)
from ..utils import (
    finish_async_reduce,
    finish_merged_async_reduce,
    reduce_gradients,
    reduce_gradients_async,
    reduce_gradients_merged_async,
)

__all__ = [
    "FBCprAuxAlgorithmCfg",
    "FBCprAux",
]


# --------------------------------------------------------------------------- #
# Algorithm configuration
# --------------------------------------------------------------------------- #


@configclass
class FBCprAuxAlgorithmCfg:
    """Training hyperparameters for :class:`FBCprAux`.

    Defaults mirror BFM-Zero's production ``train_bfm_zero()`` overrides.
    """

    class_name: str = "FBCprAux"

    # --- Global-through-Anchoring (BFM-One-Anchored) ----------------------
    # Consumed only by AnchoredFBCprAux; ignored by the base. ``store_world_pose``
    # tells the runner to record per-transition world SE(2) pose for relabeling.
    store_world_pose: bool = False
    anchored_pose_key: str = "anchored_pose"
    anchor_pose_clamp: float = 10.0          # ±metres clamp on A^-1 g xy
    anchor_alpha_gt: float = 0.34            # p(anchor = g_t)
    anchor_beta_gh: float = 0.33             # p(anchor = g_h); rest -> random
    anchor_random_xy_range: float = 10.0     # random anchor xy ± around g_t
    anchor_kl_coef: float = 0.0              # lambda_A on two-anchor policy KL
    anchor_q_coef: float = 0.0               # lambda on two-anchor Q consistency
    spatial_cpr_coeff: float = 1.0           # weight of spatial CPR reward vs local
    goal_future_ratio: float = 0.4
    goal_nearby_ratio: float = 0.2
    goal_replay_ratio: float = 0.2
    goal_composed_ratio: float = 0.2
    goal_nearby_radius: float = 2.0

    # Learning rates
    lr_f: float = 3e-4
    lr_b: float = 1e-5
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_aux_critic: float = 3e-4
    lr_discriminator: float = 1e-5

    # LR anneling. When ``lr_anneal_enable=True`` and ``lr_anneal_steps>0``,
    # linearly decay each optimizer's LR from the DDP-scaled start value
    # (``base_lr * sqrt(world_size)``) down to the un-scaled base value
    # (``base_lr``) over ``lr_anneal_steps`` env-steps. Past that point the
    # LR stays at ``base_lr``. Rationale: on DDP the sqrt-scaled LR matches
    # the reduced gradient noise but overshoots the single-rank training
    # dynamics once the policy is well-anchored; annealing recovers the
    # single-rank late-stage learning rate.
    lr_anneal_enable: bool = False
    lr_anneal_steps: int = 0

    # Optim
    weight_decay: float = 0.0
    weight_decay_discriminator: float = 0.0
    clip_grad_norm: float = 0.0  # 0 = disabled

    # Target-network Polyak rates
    fb_target_tau: float = 0.01
    critic_target_tau: float = 0.005

    # Pessimism penalties (Q_mean - penalty * Q_unc)
    fb_pessimism_penalty: float = 0.0
    critic_pessimism_penalty: float = 0.5
    aux_critic_pessimism_penalty: float = 0.5
    actor_pessimism_penalty: float = 0.5

    # TD3-style noise clip on the TruncatedNormal sample() call
    stddev_clip: float = 0.3

    # FB loss regularizers
    ortho_coef: float = 100.0
    q_loss_coef: float = 0.0  # 0 = disabled
    # Reconstruction-head weight: MSE between decoder(B(goal)) and the
    # concat of obs slices declared in ``policy.recon_targets`` (e.g.
    # end-effector XYZ). 0 disables even if the head is built; set >0
    # to make ``B`` retain the target info.
    recons_coeff: float = 0.0

    # When True, ``FBCprExpertBuffer`` scales both the initial uniform
    # priors AND any ``update_priorities()`` call by per-motion length so
    # the per-transition draw probability stays uniform across motions
    # regardless of clip-length imbalance. Default True — important for
    # datasets that mix long continuous motions with short clips.
    length_proportional_priors: bool = True

    # When True AND the runner is running under DDP (world_size>1), each
    # rank loads a disjoint shard of the expert dataset. The shard is
    # chosen by applying a seeded random permutation to the motion list
    # (same permutation on every rank, seeded by ``runner_cfg.seed``),
    # then taking the ``perm[rank::world_size]`` slice. This cuts GPU
    # memory linearly with world_size for large datasets.
    # RSI still uses the rank-local shard (envs RSI only into the motions
    # this rank owns), which for large datasets is still a statistically
    # reasonable coverage. Tracking-eval metrics are all-reduced across
    # ranks in the runner so global numbers are reported.
    distributed_expert: bool = False

    # Discriminator
    grad_penalty_discriminator: float = 10.0

    # Reg-coeffs in the actor objective
    reg_coeff: float = 0.05       # weight on Q_discriminator inside actor loss
    reg_coeff_aux: float = 0.02   # weight on Q_aux_critic inside actor loss
    scale_reg: bool = True         # multiply regs by |Q_fb|.abs().mean().detach()

    # Mixed-z sampling (at training time and for the in-rollout ZBuffer)
    batch_size: int = 1024
    # Disc batch sized as disc_num_slices * seq_length. When None, falls
    # back to batch_size rounded down to a multiple of seq_length.
    disc_num_slices: int | None = None
    # Cap per-side batch for the manifold attractor (bounds compute when
    # disc batch grows via disc_num_slices).
    ma_max_batch: int = 1024
    discount: float = 0.98
    # Separate discount for the disc-critic TD target (TD on
    # disc.compute_reward). Falls back to ``discount`` when None.
    discount_disc: float | None = None
    # Separate discount for the aux-critic TD target (penalty_xy_tracking
    # etc.). Falls back to ``discount`` when None.
    discount_aux: float | None = None
    relabel_ratio: float | None = 0.8
    train_goal_ratio: float = 0.2
    expert_asm_ratio: float = 0.6

    # Rollout-context sampling
    update_z_every_step: int = 100
    use_mix_rollout: bool = True
    rollout_expert_trajectories: bool = True
    rollout_expert_trajectories_length: int = 250
    rollout_expert_trajectories_percentage: float = 0.5
    z_buffer_size: int = 8192
    tracking_T_min: int = 1
    tracking_T_max: int = 16
    # If non-empty, sample T from this discrete set instead of uniformly
    # from [T_min, T_max]. Used for both expert disc encoding and per-env
    # tracking z window.
    tracking_T_choices: tuple[int, ...] = ()
    # Per-choice probabilities (must match len(tracking_T_choices) when set).
    # Empty tuple = uniform over choices.
    tracking_T_choice_probs: tuple[float, ...] = ()
    # If True, the discriminator's positive (expert) window is ALWAYS the full
    # seq_length regardless of the per-sequence z-window T — i.e. every frame
    # in the sub-sequence is a valid positive. The z is still computed from the
    # per-T window; only the disc_mask is forced to all-True. If False (default),
    # the positive window matches T (frames 0..T-1 only).
    disc_positive_full_window: bool = False
    # EMA alignment of the Global-FB reference frame: if > 0, each step
    # the stored ``_tracking_robot_xy`` and ``_tracking_heading_delta``
    # are pulled toward the robot's current root xy/yaw with rate
    # ``global_fb_align_ema``. The motion delta from the anchor is then
    # applied on top of this drifting frame, so the penalty stays
    # bounded if the policy drifts laterally / off-heading. 0 = off.
    global_fb_align_ema: float = 0.0

    # AMP (bf16). NOTE: the autocast context is NOT currently wired around
    # our forward/backward passes. Setting this True has no numerical
    # effect — it only gates the ``_amp_dtype`` attribute used in a few
    # diagnostic places. BFM uses per-update ``autocast(amp=cfg.amp)``
    # wrappers around every sub-network forward; porting them over is a
    # larger change we haven't taken on. Leave False unless you also wire
    # the autocast contexts.
    amp: bool = False

    # --- B200 / NVSwitch perf flags ------------------------------------
    # Parallelize phase-1 backwards (disc + F/B + aux_critic) across
    # independent CUDA streams. On fast intra-node fabrics (NVSwitch)
    # this overlaps 4 otherwise-serial backward compute chunks, saving
    # ~10-15% iter time at some engineering/debugging cost. Default
    # False (keep the safe sequential path). Disables automatically on
    # single-rank.
    stream_parallel_phase1: bool = False

    # torch.compile on the 5 trainable online networks. First iter pays
    # a ~1-2 min compile cost; FB-CPR's shapes are fully static so there
    # are no recompiles after warm-up.
    #
    # IMPORTANT: on PyTorch 2.7, ``reduce-overhead`` (CUDA graphs) is
    # broken when combined with user CUDA streams — see pytorch/pytorch
    # issues #180396 / #180497 (fix lands in release/2.12). So:
    #   - With ``stream_parallel_phase1=True``, use ``"default"`` only.
    #   - On 2.12+, ``"reduce-overhead"`` becomes safe again.
    # Default "" (disabled). Options: "", "default", "reduce-overhead",
    # "max-autotune".
    compile_mode: str = ""

    # Merge the 4 phase-1 allreduces into one. Helps on slow/high-
    # latency fabrics (EFA without GDR). On NVSwitch this loses the
    # compute/comm overlap that DDP bucket hooks provide — leave False.
    merge_phase1_reduce: bool = False

    # Aux rewards: mapping name -> scaling coefficient (applied BEFORE the
    # aux_reward normalizer). Env-exposed rewards not listed here are
    # ignored (still logged upstream).
    aux_rewards_scaling: dict[str, float] = field(default_factory=dict)

    # Optional override for ``FBCprRunner._BFM_KEY_GROUPS`` — lets a task
    # with extra obs terms (e.g. BFM-Terrain's ``height_scan``) route them
    # into new agent-input dict keys without touching the runner. Leave
    # empty to use the flat-floor BFM-Zero default.
    #
    # Example:
    #   obs_key_groups = {
    #       "state": ("state", "gravity", "root_ang_vel"),
    #       "last_action": ("last_action",),
    #       "history_actor": (... 5 history terms ...),
    #       "privileged_state": ("priv_max_local_self",),
    #       "height_scan": ("height_scan",),
    #   }
    obs_key_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)

    # --- Manifold attractor (unconditional discriminator) ---------------------
    # When True, adds a second discriminator D_ma(s_t, s_{t+1}) that
    # classifies transitions WITHOUT z conditioning. This constrains the
    # policy to stay on the expert motion manifold regardless of z.
    # The D_ma reward is added to the critic target alongside the existing
    # z-conditioned discriminator reward.
    manifold_attractor: bool = False
    manifold_attractor_coeff: float = 0.05  # weight in actor loss (same role as reg_coeff)
    lr_manifold_attractor: float = 1e-5
    grad_penalty_manifold_attractor: float = 10.0

    # --- Soft FB (entropy-regularised variant) --------------------------------
    # When True, sample z from a ball (not sphere), train an entropy critic
    # Q_H, and add an entropy bonus ``beta_z * (log_pi - Q_H)`` to the actor
    # loss. beta_z = soft_fb_entropy_coef * clamp(1 - ||z||/R, 0, 1) so
    # policies near the sphere surface stay deterministic while interior z's
    # are stochastic. Bitwise-equivalent to standard FB when False.
    soft_fb: bool = False
    soft_fb_entropy_coef: float = 1.0
    soft_fb_expert_future_min: tuple[float, float] = (0.3, 0.7)
    lr_entropy_critic: float = 3e-4
    entropy_critic_target_tau: float = 0.005


# --------------------------------------------------------------------------- #
# Algorithm
# --------------------------------------------------------------------------- #


class FBCprAux:
    """FB-CPR-Aux agent — twin F/critic/aux-critic, CPR-style discriminator, TD3 actor."""

    # Expected buffer keys (set by runner).
    _REPLAY_KEY = "train"
    _EXPERT_KEY = "expert_slicer"

    def __init__(
        self,
        policy: FBCprAuxPolicy,
        cfg: FBCprAuxAlgorithmCfg,
        device: str | torch.device = "cuda",
    ) -> None:
        self.policy = policy
        self.cfg = cfg
        self.device = str(device) if not isinstance(device, str) else device
        self.is_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

        # Disc batch is sized as disc_num_slices * seq_length (must be a
        # multiple of seq_length for [num_slices, seq_length] reshape).
        # Main cfg.batch_size is independent and stays exact (e.g. 1024).
        seq_length = int(self.policy.seq_length)
        disc_num_slices = getattr(cfg, "disc_num_slices", None)
        if disc_num_slices is not None:
            self._disc_batch_size = int(disc_num_slices) * seq_length
        else:
            self._disc_batch_size = max(seq_length, (cfg.batch_size // seq_length) * seq_length)

        # Remember the un-scaled base LRs BEFORE DDP sqrt-scaling. Used as
        # the target ("bottom") of the linear anneal schedule when
        # ``lr_anneal_enable`` is set.
        self._base_lrs: Dict[str, float] = {
            "actor": float(cfg.lr_actor),
            "critic": float(cfg.lr_critic),
            "aux_critic": float(cfg.lr_aux_critic),
            "f": float(cfg.lr_f),
            "b": float(cfg.lr_b),
            "discriminator": float(cfg.lr_discriminator),
        }

        # LR scaling. Two independent sqrt terms stack multiplicatively:
        #   (1) DDP world_size: gradient averaging over ``world_size`` ranks
        #       reduces noise by sqrt(world_size); bumping the LR by the
        #       same factor keeps the per-example step size unchanged.
        #   (2) Batch size: same story within a rank — going from the
        #       reference batch of 1024 to ``batch_size`` further reduces
        #       noise by sqrt(batch_size/1024). Applied alongside world_size.
        #
        # Combined multiplier: sqrt(world_size) * sqrt(batch_size / 1024).
        # Reference batch is 1024 (single-rank BFM default); at batch=1024
        # single-rank this is a no-op. The LR-anneal schedule targets the
        # un-scaled base LR, annealing BOTH multipliers away over
        # ``lr_anneal_steps`` env-steps.
        #
        # Discriminator gets the same ``combined_mult`` as every other branch.
        # The "disc saturates too fast on clean gradient" artifact is now
        # addressed on the downstream side by scaling
        # ``critic_target_tau`` / ``fb_target_tau`` with sqrt(W*B/B_ref) so
        # the critic's target network keeps up with the online's new speed —
        # rather than by slowing disc down.
        import math
        REF_BATCH_SIZE = 1024
        ws = (int(torch.distributed.get_world_size())
              if self.is_distributed else 1)
        bs_mult = math.sqrt(max(int(cfg.batch_size), 1) / REF_BATCH_SIZE)
        ws_mult = math.sqrt(max(ws, 1))
        combined_mult = ws_mult * bs_mult
        if combined_mult != 1.0:
            cfg.lr_actor = float(cfg.lr_actor) * combined_mult
            cfg.lr_critic = float(cfg.lr_critic) * combined_mult
            cfg.lr_aux_critic = float(cfg.lr_aux_critic) * combined_mult
            cfg.lr_f = float(cfg.lr_f) * combined_mult
            cfg.lr_b = float(cfg.lr_b) * combined_mult
            cfg.lr_discriminator = float(cfg.lr_discriminator) * combined_mult
            cfg.lr_entropy_critic = float(cfg.lr_entropy_critic) * combined_mult
            cfg.lr_manifold_attractor = float(cfg.lr_manifold_attractor) * combined_mult
            print(
                f"[FBCprAux] LR scaling: world_size={ws} (×{ws_mult:.3f})  "
                f"batch_size={cfg.batch_size}/{REF_BATCH_SIZE} (×{bs_mult:.3f})  "
                f"combined ×{combined_mult:.3f}",
                flush=True,
            )
            print(
                f"[FBCprAux] scaled LRs: "
                f"actor={cfg.lr_actor:.3g} critic={cfg.lr_critic:.3g} "
                f"aux_critic={cfg.lr_aux_critic:.3g} F={cfg.lr_f:.3g} "
                f"B={cfg.lr_b:.3g} disc={cfg.lr_discriminator:.3g}",
                flush=True,
            )

        # EMA normalizer time-constant scaling.
        #
        # The obs_normalizer (per-key BatchNorm1d) and aux_reward_normalizer
        # (EMA) are low-pass filters specified in per-iter units. When LR
        # scaling makes the online network move sqrt(W*B/B_ref) faster per
        # iter, the policy's obs / aux-reward distributions drift the same
        # factor faster — and the normalizer stats lag proportionally unless
        # we speed them up. We bump each effective ``momentum`` (= 1-tau for
        # EMA) by ``combined_mult`` so the EMA window in policy-drift units
        # stays invariant with batch/ws.
        #
        # Clipped: BatchNorm momentum ≤ 0.5 (instability past that), EMA tau
        # ≥ 0.5 (momentum ≤ 0.5). We use the SAME combined_mult as LR (no
        # disc-damp); these are downstream of the online network's motion,
        # not upstream like disc.
        if combined_mult != 1.0 and combined_mult > 0.0:
            import math as _math
            # BatchNorm per-key momentum on _obs_normalizer._normalizers[<k>]._normalizer
            new_obs_moms: Dict[str, float] = {}
            if hasattr(self.policy, "_obs_normalizer") and hasattr(
                self.policy._obs_normalizer, "_normalizers"
            ):
                for key, mod in self.policy._obs_normalizer._normalizers.items():
                    bn = getattr(mod, "_normalizer", None)
                    if bn is None or not hasattr(bn, "momentum"):
                        continue
                    old_mom = float(bn.momentum if bn.momentum is not None else 0.01)
                    new_mom = min(0.5, old_mom * combined_mult)
                    bn.momentum = new_mom
                    new_obs_moms[key] = new_mom

            # EMA aux_reward_normalizer: tau is the retention factor, so
            # effective momentum = 1 - tau. Scale the momentum, clip to 0.5.
            new_aux_tau: float | None = None
            if hasattr(self.policy, "_aux_reward_normalizer"):
                ema = self.policy._aux_reward_normalizer
                if hasattr(ema, "tau"):
                    old_tau = float(ema.tau)
                    old_mom = 1.0 - old_tau
                    new_mom = min(0.5, old_mom * combined_mult)
                    new_aux_tau = 1.0 - new_mom
                    ema.tau = new_aux_tau

            aux_tau_str = (
                f"{new_aux_tau:.4g}" if new_aux_tau is not None else "n/a"
            )
            obs_moms_str = ", ".join(
                f"{k}={v:.3g}" for k, v in new_obs_moms.items()
            )
            print(
                f"[FBCprAux] EMA normalizer scaling (×{combined_mult:.3f}): "
                f"obs_momentum={{{obs_moms_str}}}  aux_tau={aux_tau_str}",
                flush=True,
            )

        # Post-DDP-scaling LRs — the "top" of the anneal ramp. Equal to
        # ``_base_lrs`` under single-rank (so anneal is a no-op).
        self._start_lrs: Dict[str, float] = {
            "actor": float(cfg.lr_actor),
            "critic": float(cfg.lr_critic),
            "aux_critic": float(cfg.lr_aux_critic),
            "f": float(cfg.lr_f),
            "b": float(cfg.lr_b),
            "discriminator": float(cfg.lr_discriminator),
        }

        # Put the policy on device. The policy holds *all* networks, including
        # obs normalizer + aux reward normalizer + target networks.
        self.policy.to(self.device)

        # Initialize + prepare for training.
        self.policy.train(True)
        self.policy.requires_grad_(True)
        self.policy.apply(weight_init)
        self.policy._prepare_for_train()

        # DDP wrapping: wrap the 5 trainable online networks so backward
        # fires bucketed async all_reduce during the backward pass (80%+
        # overlap of reduce with compute, vs ~30% for post-backward async
        # reduce). We do NOT wrap the discriminator — its WGAN-GP term
        # uses ``autograd.grad(..., create_graph=True)`` which DDP's
        # backward hook doesn't trap correctly; that network stays on
        # manual async reduce.
        #
        # Target networks are never wrapped (no gradients — Polyak-only).
        # The obs / aux reward normalizers are not wrapped either — their
        # running stats are synced by ``_sync_running_stats`` once per
        # learn-iter via a fused all_reduce.
        self._is_ddp_wrapped = False
        if self.is_distributed:
            torch.distributed.barrier()

            from torch.nn.parallel import DistributedDataParallel as DDP
            local_rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
            try:
                dev_idx = torch.device(self.device).index
                if dev_idx is None:
                    dev_idx = local_rank
            except Exception:
                dev_idx = local_rank
            ddp_kwargs = dict(
                device_ids=[dev_idx],
                output_device=dev_idx,
                broadcast_buffers=False,          # our normalizer buffers are synced manually
                find_unused_parameters=False,
                gradient_as_bucket_view=True,     # reuse bucket storage as .grad views
            )
            self.policy._forward_map = DDP(self.policy._forward_map, **ddp_kwargs)
            self.policy._backward_map = DDP(self.policy._backward_map, **ddp_kwargs)
            self.policy._actor = DDP(self.policy._actor, **ddp_kwargs)
            self.policy._critic = DDP(self.policy._critic, **ddp_kwargs)
            self.policy._aux_critic = DDP(self.policy._aux_critic, **ddp_kwargs)
            if self.policy._entropy_critic is not None:
                self.policy._entropy_critic = DDP(self.policy._entropy_critic, **ddp_kwargs)
            self._is_ddp_wrapped = True
            _ec_str = "/entropy_critic" if self.policy._entropy_critic is not None else ""
            print(f"[FBCprAux] DDP-wrapped F/B/actor/critic/aux_critic{_ec_str} "
                  f"(disc kept on manual async reduce)", flush=True)

        # --- torch.compile on the 5 trainable online networks ---------- #
        # Applied AFTER DDP wrapping so the compiled graph includes the
        # DDP forward dispatch (``DDPOptimizer`` inserts bucket-aligned
        # graph breaks — see pytorch.org/docs/stable/notes/ddp.html).
        #
        # IMPORTANT on PyTorch 2.7: ``reduce-overhead`` mode is broken
        # when combined with user CUDA streams (issues #180396/#180497,
        # fix lands in release/2.12). If ``stream_parallel_phase1`` is
        # also enabled we force-downgrade to ``"default"`` to avoid the
        # silent breakage.
        compile_mode = getattr(cfg, "compile_mode", "") or ""
        if (
            compile_mode == "reduce-overhead"
            and getattr(cfg, "stream_parallel_phase1", False)
            and self.is_distributed
        ):
            print(f"[FBCprAux] WARNING: torch.compile(mode='reduce-overhead') "
                  f"is broken with user CUDA streams on PyTorch 2.7 "
                  f"(pytorch#180396). Downgrading to mode='default'.",
                  flush=True)
            compile_mode = "default"
        if compile_mode:
            compile_kwargs = {"mode": compile_mode, "fullgraph": False}
            self.policy._forward_map = torch.compile(self.policy._forward_map, **compile_kwargs)
            self.policy._backward_map = torch.compile(self.policy._backward_map, **compile_kwargs)
            self.policy._actor = torch.compile(self.policy._actor, **compile_kwargs)
            self.policy._critic = torch.compile(self.policy._critic, **compile_kwargs)
            self.policy._aux_critic = torch.compile(self.policy._aux_critic, **compile_kwargs)
            if self.policy._entropy_critic is not None:
                self.policy._entropy_critic = torch.compile(self.policy._entropy_critic, **compile_kwargs)
            # Disc uses autograd.grad(create_graph=True) for WGAN-GP, which
            # is known to hit graph breaks with torch.compile — leave it
            # eager. Target networks never need compile (no backward).
            print(f"[FBCprAux] torch.compile mode={compile_mode} applied to "
                  f"F/B/actor/critic/aux_critic (disc stays eager)", flush=True)

        # --- Stream-parallel phase-1 backward setup -------------------- #
        # When enabled, phase-1 backward passes (disc + F/B + aux_critic)
        # run concurrently on dedicated CUDA streams, then sync before
        # the merged allreduce + optimizer.step(). Requires
        # ``merge_phase1_reduce=True`` to work — DDP's in-backward bucket
        # hooks serialize on the main stream and defeat the parallelism.
        self._stream_parallel_phase1 = bool(
            getattr(cfg, "stream_parallel_phase1", False)
            and self.is_distributed
            and self._is_ddp_wrapped
        )
        if self._stream_parallel_phase1:
            # Force merge mode on when streams are active (required).
            cfg.merge_phase1_reduce = True
            self._phase1_stream_disc = torch.cuda.Stream()
            self._phase1_stream_fb = torch.cuda.Stream()
            self._phase1_stream_aux = torch.cuda.Stream()
            print(f"[FBCprAux] stream_parallel_phase1 enabled "
                  f"(auto-enabled merge_phase1_reduce)", flush=True)
        else:
            self._phase1_stream_disc = None
            self._phase1_stream_fb = None
            self._phase1_stream_aux = None

        # Optimizers.
        self._build_optimizers()

        # Batch-level precompute (off-diag mask for F-B loss).
        self._off_diag = 1.0 - torch.eye(
            cfg.batch_size, cfg.batch_size, device=self.device
        )
        self._off_diag_sum = self._off_diag.sum()

        # Track AMP dtype (bf16 when enabled).
        self._amp_dtype = torch.bfloat16

        # ZBuffer will be created lazily when `sample_mixed_z` first writes.
        self._z_buffer: Optional[torch.Tensor] = None
        self._z_buffer_cursor = 0
        self._z_buffer_size = 0

    # --- optimizer setup --------------------------------------------------- #

    def _build_optimizers(self) -> None:
        cfg = self.cfg
        p = self.policy
        # ``fused=True`` enables PyTorch's fused Adam CUDA kernel, which
        # issues one kernel per param group instead of one per param
        # tensor. On a 440M-param network with 96 optimizer steps/iter
        # (6 nets × 16 updates), this saves ~100-200 ms/iter on B200.
        # Requires all params on CUDA — always true for our setup.
        adam_kwargs = {"fused": True}
        # Backward optimizer also trains the reconstruction head (if any)
        # so B and the decoder move together under the same LR schedule.
        b_params = list(p._backward_map.parameters())
        if getattr(p, "_reconstruction_head", None) is not None:
            b_params += list(p._reconstruction_head.parameters())
        self.backward_optimizer = torch.optim.Adam(
            b_params,
            lr=cfg.lr_b,
            weight_decay=cfg.weight_decay,
            **adam_kwargs,
        )
        self.forward_optimizer = torch.optim.Adam(
            p._forward_map.parameters(),
            lr=cfg.lr_f,
            weight_decay=cfg.weight_decay,
            **adam_kwargs,
        )
        self.actor_optimizer = torch.optim.Adam(
            p._actor.parameters(),
            lr=cfg.lr_actor,
            weight_decay=cfg.weight_decay,
            **adam_kwargs,
        )
        self.critic_optimizer = torch.optim.Adam(
            p._critic.parameters(),
            lr=cfg.lr_critic,
            weight_decay=cfg.weight_decay,
            **adam_kwargs,
        )
        self.aux_critic_optimizer = torch.optim.Adam(
            p._aux_critic.parameters(),
            lr=cfg.lr_aux_critic,
            weight_decay=cfg.weight_decay,
            **adam_kwargs,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self._discriminator_opt_params(),
            lr=cfg.lr_discriminator,
            weight_decay=cfg.weight_decay_discriminator,
            **adam_kwargs,
        )
        # Manifold attractor (unconditional disc).
        self.manifold_attractor_optimizer: torch.optim.Optimizer | None = None
        if p._manifold_attractor is not None:
            self.manifold_attractor_optimizer = torch.optim.Adam(
                p._manifold_attractor.parameters(),
                lr=cfg.lr_manifold_attractor,
                weight_decay=cfg.weight_decay_discriminator,
                **adam_kwargs,
            )

        # Entropy critic (Soft FB only).
        self.entropy_critic_optimizer: torch.optim.Optimizer | None = None
        if p._entropy_critic is not None:
            self.entropy_critic_optimizer = torch.optim.Adam(
                p._entropy_critic.parameters(),
                lr=cfg.lr_entropy_critic,
                weight_decay=cfg.weight_decay,
                **adam_kwargs,
            )

        # Param lists for fast soft updates.
        self._forward_map_params = tuple(x.data for x in p._forward_map.parameters())
        self._target_forward_map_params = tuple(
            x.data for x in p._target_forward_map.parameters()
        )
        self._backward_map_params = tuple(x.data for x in p._backward_map.parameters())
        self._target_backward_map_params = tuple(
            x.data for x in p._target_backward_map.parameters()
        )
        self._critic_params = tuple(x.data for x in p._critic.parameters())
        self._target_critic_params = tuple(
            x.data for x in p._target_critic.parameters()
        )
        self._aux_critic_params = tuple(x.data for x in p._aux_critic.parameters())
        self._target_aux_critic_params = tuple(
            x.data for x in p._target_aux_critic.parameters()
        )
        if p._entropy_critic is not None:
            self._entropy_critic_params = tuple(x.data for x in p._entropy_critic.parameters())
            self._target_entropy_critic_params = tuple(
                x.data for x in p._target_entropy_critic.parameters()
            )
        else:
            self._entropy_critic_params = ()
            self._target_entropy_critic_params = ()

    @property
    def optimizer_dict(self) -> Dict[str, Any]:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "backward_optimizer": self.backward_optimizer.state_dict(),
            "forward_optimizer": self.forward_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "aux_critic_optimizer": self.aux_critic_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            **({"entropy_critic_optimizer": self.entropy_critic_optimizer.state_dict()}
               if self.entropy_critic_optimizer is not None else {}),
            **({"manifold_attractor_optimizer": self.manifold_attractor_optimizer.state_dict()}
               if self.manifold_attractor_optimizer is not None else {}),
        }

    # --- inference surface ------------------------------------------------- #

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        mean: bool = False,
    ) -> torch.Tensor:
        return self.policy.act(obs, z, mean=mean)

    # --- mixed-z + rollout-context sampling --------------------------------- #

    @torch.no_grad()
    def _zbuf_add(self, z: torch.Tensor) -> None:
        if self._z_buffer is None:
            self._z_buffer = torch.zeros(
                (self.cfg.z_buffer_size, z.shape[-1]),
                device=self.device,
                dtype=torch.float32,
            )
        n = z.shape[0]
        buf = self._z_buffer
        cap = buf.shape[0]
        idxs = (torch.arange(n, device=buf.device) + self._z_buffer_cursor) % cap
        buf[idxs] = z.detach().to(buf.dtype)
        self._z_buffer_cursor = int((self._z_buffer_cursor + n) % cap)
        self._z_buffer_size = min(self._z_buffer_size + n, cap)

    @torch.no_grad()
    def _zbuf_empty(self) -> bool:
        return self._z_buffer_size == 0

    @torch.no_grad()
    def _zbuf_sample(self, batch_size: int) -> torch.Tensor:
        assert self._z_buffer is not None and self._z_buffer_size > 0
        idx = torch.randint(
            0, self._z_buffer_size, (batch_size,), device=self._z_buffer.device
        )
        return self._z_buffer[idx].clone()

    @torch.no_grad()
    def sample_mixed_z(
        self,
        train_goal: torch.Tensor | dict[str, torch.Tensor],
        expert_encodings: torch.Tensor,
    ) -> torch.Tensor:
        """Mix of uniform-random / goal-encoded / expert-encoded z's.

        Mirrors BFM-Zero's ``FBcprAgent.sample_mixed_z``.
        """
        batch = self.cfg.batch_size
        z = self.policy.sample_z(batch, device=self.device)
        p_goal = self.cfg.train_goal_ratio
        p_expert_asm = self.cfg.expert_asm_ratio
        prob = torch.tensor(
            [p_goal, p_expert_asm, 1.0 - p_goal - p_expert_asm],
            dtype=torch.float32,
            device=self.device,
        )
        mix_idxs = torch.multinomial(prob, num_samples=batch, replacement=True).view(-1, 1)

        # Goal-encoded z's
        perm = torch.randperm(batch, device=self.device)
        shuffled = self._permute_obs(train_goal, perm)
        goals = self.policy._backward_map(shuffled)
        goals = self.policy.project_z(goals)
        z = torch.where(mix_idxs == 0, goals, z)

        # Expert-encoded z's. Sample with replacement so expert pool size
        # can differ from main batch size.
        n_expert = expert_encodings.shape[0]
        idx = torch.randint(0, n_expert, (batch,), device=self.device)
        expert_z = expert_encodings[idx]
        z = torch.where(mix_idxs == 1, expert_z, z)
        return z

    @staticmethod
    def _permute_obs(
        obs: torch.Tensor | dict[str, torch.Tensor], perm: torch.Tensor
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(obs, dict):
            return {k: v[perm] for k, v in obs.items()}
        return obs[perm]

    @torch.no_grad()
    def encode_expert(
        self, next_obs: torch.Tensor | dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode expert sub-sequences through B with variable-T window.

        For each sub-sequence, samples T ∈ [T_min, T_max], computes z as
        mean of first T frames. Only frames within the T-window are "real"
        for discriminator training; returns a mask marking valid frames.

        Returns:
            z_expert: [batch_size, z_dim] z replicated per frame
            disc_mask: [batch_size] bool, True for frames within T-window.
                None if no variable-T (all frames valid).
        """
        B_expert = self.policy._backward_map(next_obs).detach()
        seq_length = self.policy.seq_length
        # Use the actual batch returned (may be _disc_batch_size, not cfg.batch_size).
        N = B_expert.shape[0] // seq_length
        B_expert = B_expert.view(N, seq_length, B_expert.shape[-1])
        device = B_expert.device

        # Variable T per sub-sequence
        choices = tuple(getattr(self.cfg, "tracking_T_choices", ()) or ())
        choice_probs = tuple(getattr(self.cfg, "tracking_T_choice_probs", ()) or ())
        T_min = getattr(self.cfg, "tracking_T_min", 1)
        T_max = min(getattr(self.cfg, "tracking_T_max", 16), seq_length)
        disc_mask: torch.Tensor | None = None
        T_per_seq: torch.Tensor | None = None
        if choices:
            kept = [(c, choice_probs[i] if choice_probs else 1.0)
                    for i, c in enumerate(choices) if c <= seq_length]
            choices_kept = [c for c, _ in kept]
            probs_kept = [p for _, p in kept]
            choices_t = torch.tensor(choices_kept, device=device, dtype=torch.long)
            if choice_probs and len(probs_kept) == len(choices_kept):
                w = torch.tensor(probs_kept, device=device, dtype=torch.float32)
                sel = torch.multinomial(w, N, replacement=True)
            else:
                sel = torch.randint(0, len(choices_kept), (N,), device=device)
            T_per_seq = choices_t[sel]
        elif T_min < T_max:
            T_per_seq = torch.randint(T_min, T_max + 1, (N,), device=device)
        if T_per_seq is not None:
            d = B_expert.shape[-1]
            cumz = torch.cat([torch.zeros(N, 1, d, device=device),
                              torch.cumsum(B_expert, dim=1)], dim=1)  # [N, seq+1, d]
            arange_N = torch.arange(N, device=device)
            z_sum = cumz[arange_N, T_per_seq]  # [N, d]
            z_expert = z_sum / T_per_seq.float().unsqueeze(-1)
            # Frames 0..T-1 are within window; T..seq_length-1 are not.
            arange_T = torch.arange(seq_length, device=device).unsqueeze(0)
            disc_mask = (arange_T < T_per_seq.unsqueeze(1)).reshape(-1)  # [N*seq_length]
            # Optionally use the FULL seq_length as the discriminator positive
            # window for every sub-sequence, regardless of its z-window T (z is
            # still the per-T mean above; only the positive mask is widened).
            if bool(getattr(self.cfg, "disc_positive_full_window", False)):
                disc_mask = None
        else:
            z_expert = B_expert.mean(dim=1)

        if self.cfg.soft_fb:
            norm = z_expert.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            z_expert = z_expert / (norm + 1.0)
        else:
            z_expert = self.policy.project_z(z_expert)

        z_expert = torch.repeat_interleave(z_expert, seq_length, dim=0)
        return z_expert, disc_mask

    @torch.no_grad()
    def maybe_update_rollout_context(
        self,
        z: torch.Tensor | None,
        step_count: torch.Tensor,
        expert_buffer: Any | None = None,
        robot_root_xy: torch.Tensor | None = None,
        robot_root_quat: torch.Tensor | None = None,
        terrain_z_fn=None,
    ) -> tuple[torch.Tensor, dict | None]:
        """Update the rollout-time z context.

        Called once per env step by the runner with per-env ``step_count``.

        Returns:
            (z, terrain_reset_env_ids): z is the updated per-env latent.
            terrain_reset_env_ids is a tensor of env indices that were
            assigned terrain-required motions and need an env reset to
            align with the terrain, or None.
        """
        if z is None:
            z = self.policy.sample_z(step_count.shape[0], device=self.device)
            if self.cfg.rollout_expert_trajectories and expert_buffer is not None:
                terrain_envs = self._resample_tracking(
                    step_count, expert_buffer, robot_root_xy, robot_root_quat,
                    terrain_z_fn=terrain_z_fn,
                )
                z[self._tracking_env_idx] = self._tracking_z[:, 0]
                return z, terrain_envs
            else:
                self._tracking_env_idx = None
            return z, None

        # Periodic z refresh for non-tracking envs.
        mask_reset_z = (step_count % self.cfg.update_z_every_step == 0).view(-1, 1)
        if self.cfg.use_mix_rollout and not self._zbuf_empty():
            new_z = self._zbuf_sample(z.shape[0])
        else:
            new_z = self.policy.sample_z(z.shape[0], device=self.device)
        z = torch.where(mask_reset_z, new_z, z.to(self.device))

        terrain_envs = None
        if self.cfg.rollout_expert_trajectories and expert_buffer is not None:
            idxs = step_count % self.cfg.rollout_expert_trajectories_length
            if bool((idxs == 0).any()):
                terrain_envs = self._resample_tracking(
                    step_count, expert_buffer, robot_root_xy, robot_root_quat,
                    terrain_z_fn=terrain_z_fn,
                )
            if getattr(self, "_tracking_env_idx", None) is not None:
                mod_time = idxs[self._tracking_env_idx].view(-1)
                mod_time = torch.clamp(mod_time, 0, self._tracking_z.shape[1] - 1)
                n = len(self._tracking_env_idx)
                z[self._tracking_env_idx] = self._tracking_z[
                    torch.arange(n, device=self.device), mod_time,
                ]
        return z, terrain_envs

    def _resample_tracking(
        self,
        step_count: torch.Tensor,
        expert_buffer: Any,
        robot_root_xy: torch.Tensor | None,
        robot_root_quat: torch.Tensor | None,
        terrain_z_fn=None,
    ) -> dict | None:
        """Pick new tracking envs, sample trajectories, encode z, store viz anchors.

        Returns dict with terrain env info for caller to reset, or None.
        ``terrain_z_fn``: callable([M,2] -> [M]) for sim terrain height query.
        """
        n_envs = step_count.shape[0]
        n_elem = max(1, int(self.cfg.rollout_expert_trajectories_percentage * n_envs))
        self._tracking_env_idx = torch.randint(0, n_envs, (n_elem,), device=self.device)
        traj_len = self.cfg.rollout_expert_trajectories_length
        # Decide global root_h flag BEFORE z encoding.
        grh_prob = getattr(self.cfg, "terrain_variant_root_h_prob", 0.25)
        use_global = torch.rand(n_elem, device=self.device) < grh_prob
        global_rh = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
        global_rh[self._tracking_env_idx] = use_global
        self._tracking_terrain_variant_root_h = global_rh
        # Global FB: sample active mask once per tracking episode.
        global_fb_prob = getattr(self.cfg, "global_fb_zero_prob", 0.5)
        self._tracking_global_fb_active = torch.rand(n_elem, device=self.device) >= global_fb_prob
        # Per-env variable T for z computation window.
        choices = tuple(getattr(self.cfg, "tracking_T_choices", ()) or ())
        choice_probs = tuple(getattr(self.cfg, "tracking_T_choice_probs", ()) or ())
        T_min = getattr(self.cfg, "tracking_T_min", 1)
        T_max = getattr(self.cfg, "tracking_T_max", 16)
        if choices:
            choices_t = torch.tensor(choices, device=self.device, dtype=torch.long)
            if choice_probs and len(choice_probs) == len(choices):
                w = torch.tensor(choice_probs, device=self.device, dtype=torch.float32)
                sel = torch.multinomial(w, n_elem, replacement=True)
            else:
                sel = torch.randint(0, len(choices), (n_elem,), device=self.device)
            self._tracking_T = choices_t[sel]
        elif T_min < T_max:
            self._tracking_T = torch.randint(T_min, T_max + 1, (n_elem,), device=self.device)
        else:
            self._tracking_T = None
        # Store robot pose for reference viz.
        if robot_root_xy is not None:
            self._tracking_robot_xy = robot_root_xy[self._tracking_env_idx].to(self.device).clone()
        else:
            self._tracking_robot_xy = None
        # Sample trajectories first (sets _tracking_motion_ids/starts/lens).
        batch = expert_buffer.sample_tracking_trajectories(n_elem, traj_len)
        self._tracking_motion_ids = batch["motion_ids"].to(self.device)
        self._tracking_starts = batch["starts"].to(self.device)
        self._tracking_motion_lens = batch["motion_lens"].to(self.device)
        rt = batch.get("requires_terrain")
        self._tracking_requires_terrain = rt.to(self.device) if rt is not None else None
        # Now compute heading delta (needs motion_ids/starts).
        self._tracking_heading_delta = self._compute_heading_delta(
            expert_buffer, robot_root_quat,
        )
        # Encode z with expert obs patching for terrain-variant root_h.
        self._tracking_z = self._sample_tracking_z(
            expert_buffer, n_elem, traj_len,
            terrain_variant_root_h=use_global,
            terrain_z_fn=terrain_z_fn,
            batch=batch,
        )
        # Return terrain env indices for caller to reset.
        rt = self._tracking_requires_terrain
        if rt is not None and rt.any():
            mask = rt
            return {
                "env_ids": self._tracking_env_idx[mask],
                "motion_ids": self._tracking_motion_ids[mask],
                "starts": self._tracking_starts[mask],
            }
        return None

    def update_tracking_pose_after_reset(
        self,
        reset_env_ids: torch.Tensor,
        robot_root_xy: torch.Tensor,
        robot_root_quat: torch.Tensor,
    ) -> None:
        """Update stored robot pose for terrain-reset envs.

        Called by the runner AFTER ``_reset_idx`` so that reference viz
        uses the post-reset robot position (not the stale pre-reset one).
        """
        if self._tracking_robot_xy is None or self._tracking_env_idx is None:
            return
        # Vectorized: find tracking slots that correspond to reset envs.
        reset_set = reset_env_ids.to(self.device)
        mask = (self._tracking_env_idx.unsqueeze(1) == reset_set.unsqueeze(0)).any(dim=1)
        if not mask.any():
            return
        reset_eids = self._tracking_env_idx[mask]
        self._tracking_robot_xy[mask] = robot_root_xy[reset_eids].to(self.device)
        if self._tracking_heading_delta is not None and self._cached_root_quat_dev is not None:
            robot_yaw = self._yaw_from_quat(robot_root_quat[reset_eids].to(self.device))
            anchor_idx = self._tracking_anchor_global_idx()[mask]
            motion_yaw = self._yaw_from_quat(self._cached_root_quat_dev[anchor_idx])
            self._tracking_heading_delta[mask] = robot_yaw - motion_yaw

    def _ensure_buffer_cache(self, expert_buffer: Any) -> None:
        """Lazily cache expert buffer tensors on self.device."""
        if not hasattr(self, "_cached_obs_starts_dev"):
            self._cached_obs_starts_dev = expert_buffer._motion_obs_starts.to(self.device)
            self._cached_root_pos_dev = expert_buffer.root_pos_buffer.to(self.device)
            self._cached_root_quat_dev = (
                expert_buffer.root_quat_buffer.to(self.device)
                if expert_buffer.root_quat_buffer is not None else None
            )

    def _tracking_anchor_global_idx(self) -> torch.Tensor:
        """Global flat index for each tracking trajectory's anchor frame."""
        usable = (self._tracking_motion_lens - 1).clamp_min(1)
        frame0 = self._tracking_starts % usable
        return (self._cached_obs_starts_dev[self._tracking_motion_ids] + frame0).long()

    @torch.no_grad()
    def get_tracking_ref_root_pos(
        self, step_count: torch.Tensor, expert_buffer: Any,
    ) -> torch.Tensor | None:
        """Return per-env reference root_pos [N, 3] for tracking envs.

        XY delta from the motion anchor is rotated by the heading difference
        between the robot at sample time and the motion at its anchor frame.
        """
        if getattr(self, "_tracking_env_idx", None) is None:
            return None
        if getattr(self, "_tracking_motion_ids", None) is None:
            return None
        if expert_buffer.root_pos_buffer is None:
            return None
        self._ensure_buffer_cache(expert_buffer)
        N = step_count.shape[0]
        ref = torch.zeros(N, 3, device=self.device)
        traj_len = self.cfg.rollout_expert_trajectories_length
        local_t = step_count[self._tracking_env_idx] % traj_len
        usable = (self._tracking_motion_lens - 1).clamp_min(1)
        frame = (self._tracking_starts + local_t.view(-1) + 1) % usable
        obs_starts = self._cached_obs_starts_dev[self._tracking_motion_ids]
        anchor_idx = self._tracking_anchor_global_idx()
        motion_anchor_xy = self._cached_root_pos_dev[anchor_idx, :2]
        motion_pos = self._cached_root_pos_dev[(obs_starts + frame).long()]
        if self._tracking_robot_xy is not None:
            delta_xy = motion_pos[:, :2] - motion_anchor_xy
            if self._tracking_heading_delta is not None:
                cos_d = torch.cos(self._tracking_heading_delta).unsqueeze(-1)
                sin_d = torch.sin(self._tracking_heading_delta).unsqueeze(-1)
                dx, dy = delta_xy[:, 0:1], delta_xy[:, 1:2]
                delta_xy = torch.cat([cos_d * dx - sin_d * dy,
                                      sin_d * dx + cos_d * dy], dim=-1)
            motion_pos = motion_pos.clone()
            motion_pos[:, :2] = self._tracking_robot_xy + delta_xy
        ref[self._tracking_env_idx] = motion_pos
        return ref

    @torch.no_grad()
    def get_global_fb_targets(
        self,
        step_count: torch.Tensor,
        expert_buffer: Any,
        robot_root_xy: torch.Tensor | None = None,
        robot_root_quat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return (target_xy [N,2], target_yaw [N], active [N] bool, tracking_ids [M]).

        For tracking envs: target is the motion's current frame root_pos_xy
        (with heading rotation + spawn offset applied). Active mask is
        sampled once per tracking episode in ``_resample_tracking``
        (controls obs visibility, not penalty).

        If ``cfg.global_fb_align_ema > 0`` AND ``robot_root_*`` are provided,
        the stored ``_tracking_robot_xy`` (anchor for the motion frame) and
        ``_tracking_heading_delta`` are pulled toward the robot's current
        pose by ``ema`` each step, before computing the target. This keeps
        the penalty bounded when the policy drifts off the motion path.
        """
        if getattr(self, "_tracking_env_idx", None) is None:
            return None
        if getattr(self, "_tracking_motion_ids", None) is None:
            return None
        if expert_buffer.root_pos_buffer is None:
            return None
        self._ensure_buffer_cache(expert_buffer)
        N = step_count.shape[0]
        target_xy = torch.zeros(N, 2, device=self.device)
        target_yaw = torch.zeros(N, device=self.device)
        active = torch.zeros(N, dtype=torch.bool, device=self.device)

        # ---- Optional EMA alignment of the reference frame ----
        ema = float(getattr(self.cfg, "global_fb_align_ema", 0.0))
        if (ema > 0.0 and robot_root_xy is not None
                and self._tracking_robot_xy is not None):
            cur_xy = robot_root_xy[self._tracking_env_idx].to(self.device)
            self._tracking_robot_xy = (1.0 - ema) * self._tracking_robot_xy + ema * cur_xy
            if (robot_root_quat is not None
                    and self._tracking_heading_delta is not None
                    and self._cached_root_quat_dev is not None):
                cur_yaw = self._yaw_from_quat(
                    robot_root_quat[self._tracking_env_idx].to(self.device)
                )
                anchor_idx = self._tracking_anchor_global_idx()
                motion_anchor_yaw = self._yaw_from_quat(self._cached_root_quat_dev[anchor_idx])
                target_heading_delta = cur_yaw - motion_anchor_yaw
                # Wrap to [-pi, pi] before EMA to avoid 2pi jumps.
                d = (target_heading_delta - self._tracking_heading_delta + math.pi) % (2 * math.pi) - math.pi
                self._tracking_heading_delta = self._tracking_heading_delta + ema * d

        traj_len = self.cfg.rollout_expert_trajectories_length
        local_t = step_count[self._tracking_env_idx] % traj_len
        usable = (self._tracking_motion_lens - 1).clamp_min(1)
        frame = (self._tracking_starts + local_t.view(-1) + 1) % usable
        obs_starts = self._cached_obs_starts_dev[self._tracking_motion_ids]
        global_idx = (obs_starts + frame).long()

        motion_pos = self._cached_root_pos_dev[global_idx]
        motion_quat = self._cached_root_quat_dev[global_idx]

        # Apply heading rotation + XY offset (same as get_tracking_ref_root_pos)
        anchor_idx = self._tracking_anchor_global_idx()
        motion_anchor_xy = self._cached_root_pos_dev[anchor_idx, :2]
        if self._tracking_robot_xy is not None:
            delta_xy = motion_pos[:, :2] - motion_anchor_xy
            if self._tracking_heading_delta is not None:
                cos_d = torch.cos(self._tracking_heading_delta).unsqueeze(-1)
                sin_d = torch.sin(self._tracking_heading_delta).unsqueeze(-1)
                dx, dy = delta_xy[:, 0:1], delta_xy[:, 1:2]
                delta_xy = torch.cat([cos_d * dx - sin_d * dy,
                                      sin_d * dx + cos_d * dy], dim=-1)
            world_xy = self._tracking_robot_xy + delta_xy
        else:
            world_xy = motion_pos[:, :2]

        # Target yaw from motion quat + heading delta
        motion_yaw = self._yaw_from_quat(motion_quat)
        if self._tracking_heading_delta is not None:
            world_yaw = motion_yaw + self._tracking_heading_delta
        else:
            world_yaw = motion_yaw

        target_xy[self._tracking_env_idx] = world_xy
        target_yaw[self._tracking_env_idx] = world_yaw
        global_fb_mask = getattr(self, "_tracking_global_fb_active", None)
        if global_fb_mask is not None:
            active[self._tracking_env_idx] = global_fb_mask
        else:
            active[self._tracking_env_idx] = True

        return target_xy, target_yaw, active, self._tracking_env_idx

    @torch.no_grad()
    def get_tracking_ref_body_pos(
        self, step_count: torch.Tensor, expert_buffer: Any,
        terrain_z_fn=None,
    ) -> torch.Tensor | None:
        """Return per-env reference body keypoints [N, K, 3] via FK.

        For non-terrain motions, applies heading rotation + XY offset.
        If ``terrain_z_fn`` is provided (callable: [M, 2] -> [M]), queries
        terrain height at the motion's transformed root XY and adds it
        to body z so reference sits on the sim terrain surface.
        """
        if getattr(self, "_tracking_env_idx", None) is None:
            return None
        if getattr(self, "_tracking_motion_ids", None) is None:
            return None
        if expert_buffer.root_pos_buffer is None:
            return None
        self._ensure_buffer_cache(expert_buffer)
        traj_len = self.cfg.rollout_expert_trajectories_length
        local_t = step_count[self._tracking_env_idx] % traj_len
        usable = (self._tracking_motion_lens - 1).clamp_min(1)
        frame = (self._tracking_starts + local_t.view(-1) + 1) % usable
        obs_starts = self._cached_obs_starts_dev[self._tracking_motion_ids]
        global_idx = (obs_starts + frame).long()
        body_pos = expert_buffer.compute_body_pos(global_idx)
        if body_pos is None:
            return None
        N = step_count.shape[0]
        K = body_pos.shape[1]
        ref = torch.zeros(N, K, 3, device=self.device)
        body_pos = body_pos.to(self.device)
        # Terrain motions: raw dataset positions (robot init matches exactly).
        # Non-terrain motions: apply heading rotation + XY offset from spawn randomization.
        rt = getattr(self, "_tracking_requires_terrain", None)
        if rt is not None and self._tracking_robot_xy is not None:
            is_terrain = rt.to(self.device)
            non_terrain = ~is_terrain
            if non_terrain.any():
                anchor_idx = self._tracking_anchor_global_idx()
                anchor_xy = self._cached_root_pos_dev[anchor_idx, :2]
                nt = non_terrain
                delta_all = body_pos[nt, :, :2] - anchor_xy[nt].unsqueeze(1)
                if self._tracking_heading_delta is not None:
                    hd = self._tracking_heading_delta[nt]
                    cos_d = torch.cos(hd).view(-1, 1, 1)
                    sin_d = torch.sin(hd).view(-1, 1, 1)
                    dx, dy = delta_all[:, :, 0:1], delta_all[:, :, 1:2]
                    delta_all = torch.cat([cos_d * dx - sin_d * dy,
                                           sin_d * dx + cos_d * dy], dim=-1)
                body_pos[nt] = body_pos[nt].clone()
                body_pos[nt, :, :2] = self._tracking_robot_xy[nt].unsqueeze(1) + delta_all
                # Query terrain z at the motion's transformed root (pelvis) XY.
                if terrain_z_fn is not None:
                    motion_root_xy = body_pos[nt, 0, :2]  # pelvis after XY offset
                    tz = terrain_z_fn(motion_root_xy)
                    body_pos[nt, :, 2] = body_pos[nt, :, 2] + tz.unsqueeze(1)
        ref[self._tracking_env_idx] = body_pos
        return ref

    @torch.no_grad()
    def get_tracking_ref_whole_body(
        self, step_count: torch.Tensor, expert_buffer: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Per-env reference whole-body state for tracking envs, for an
        explicit heading-frame imitation reward.

        Returns ``(ref_priv [N, 463], ref_joint_pos [N, 29],
        ref_joint_vel [N, 29], tracking_mask [N] bool)``:
          * ref_priv = the expert ``_flat_priv`` (heading-local, pelvis-relative
            463-D: root_h + keybody pos/rot6d/lin_vel/ang_vel) at the motion's
            current tracking frame. Directly comparable to the env's live
            ``_compute_priv_state()`` (same layout/frame), so NO world
            alignment is needed.
          * ref_joint_pos/vel from the RSI joint buffers at the same frame.
        Rows for non-tracking envs are zero with tracking_mask=False. Returns
        None if no tracking context or the buffer lacks the fields.
        """
        if getattr(self, "_tracking_env_idx", None) is None:
            return None
        if getattr(self, "_tracking_motion_ids", None) is None:
            return None
        fp = getattr(expert_buffer, "_flat_priv", None)
        jp_buf = getattr(expert_buffer, "joint_pos_buffer", None)
        jv_buf = getattr(expert_buffer, "joint_vel_buffer", None)
        if fp is None or jp_buf is None or jv_buf is None:
            return None
        self._ensure_buffer_cache(expert_buffer)
        N = step_count.shape[0]
        traj_len = self.cfg.rollout_expert_trajectories_length
        local_t = step_count[self._tracking_env_idx] % traj_len
        usable = (self._tracking_motion_lens - 1).clamp_min(1)
        frame = (self._tracking_starts + local_t.view(-1) + 1) % usable
        obs_starts = self._cached_obs_starts_dev[self._tracking_motion_ids]
        global_idx = (obs_starts + frame).long()

        ref_priv = torch.zeros(N, fp.shape[-1], device=self.device)
        ref_jp = torch.zeros(N, jp_buf.shape[-1], device=self.device)
        ref_jv = torch.zeros(N, jv_buf.shape[-1], device=self.device)
        mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        idx = global_idx.to(fp.device)
        ref_priv[self._tracking_env_idx] = fp[idx].to(self.device)
        ref_jp[self._tracking_env_idx] = jp_buf[idx].to(self.device)
        ref_jv[self._tracking_env_idx] = jv_buf[idx].to(self.device)
        mask[self._tracking_env_idx] = True
        return ref_priv, ref_jp, ref_jv, mask

    @staticmethod
    def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
        """Extract yaw from wxyz quaternion. Returns [N]."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _compute_heading_delta(
        self, expert_buffer: Any, robot_root_quat: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return yaw difference (robot - motion anchor) or None."""
        if robot_root_quat is None:
            return None
        self._ensure_buffer_cache(expert_buffer)
        if self._cached_root_quat_dev is None:
            return None
        robot_yaw = self._yaw_from_quat(
            robot_root_quat[self._tracking_env_idx].to(self.device),
        )
        motion_yaw = self._yaw_from_quat(
            self._cached_root_quat_dev[self._tracking_anchor_global_idx()],
        )
        return robot_yaw - motion_yaw

    @torch.no_grad()
    def _sample_tracking_z(
        self,
        expert_buffer: Any,
        batch_dim: int,
        traj_length: int,
        terrain_variant_root_h: torch.Tensor | None = None,
        terrain_z_fn=None,
        batch: dict | None = None,
    ) -> torch.Tensor:
        """Encode z from expert sub-trajectories.

        If ``batch`` is provided (pre-sampled by caller), uses it directly.
        Otherwise samples via ``expert_buffer.sample_tracking_trajectories``.
        """
        seq_length = self.policy.seq_length
        if batch is None:
            batch = expert_buffer.sample_tracking_trajectories(batch_dim, traj_length)
            self._tracking_motion_ids = batch["motion_ids"].to(self.device)
            self._tracking_starts = batch["starts"].to(self.device)
            self._tracking_motion_lens = batch["motion_lens"].to(self.device)
            rt = batch.get("requires_terrain")
            self._tracking_requires_terrain = rt.to(self.device) if rt is not None else None
        next_obs = batch["next_observation"]
        next_obs = self._to_device(next_obs)
        # Patch expert root_h for global-root_h envs.
        if terrain_variant_root_h is not None and terrain_variant_root_h.any() and "privileged_state" in next_obs:
            self._ensure_buffer_cache(expert_buffer)
            priv = next_obs["privileged_state"]  # [B*T, priv_dim]
            B_T = priv.shape[0]
            # Get root_pos_z for each frame from the flat buffer.
            # Reconstruct global indices (same as sample_tracking_trajectories).
            arange = torch.arange(traj_length, device=self.device).unsqueeze(0)
            raw_frame = self._tracking_starts.unsqueeze(1) + arange
            usable = (self._tracking_motion_lens - 1).clamp_min(1).unsqueeze(1)
            is_t = self._tracking_requires_terrain
            if is_t is not None:
                frame_nxt = torch.where(
                    is_t.unsqueeze(1),
                    (raw_frame + 1).clamp(max=usable - 1),
                    (raw_frame + 1) % usable,
                )
            else:
                frame_nxt = (raw_frame + 1) % usable
            obs_starts = self._cached_obs_starts_dev[self._tracking_motion_ids]
            global_nxt = (obs_starts.unsqueeze(1) + frame_nxt).long().reshape(-1)
            root_pos_z = self._cached_root_pos_dev[global_nxt, 2]  # [B*T]
            # Expand terrain_variant_root_h to [B*T]
            grh_flat = terrain_variant_root_h.unsqueeze(1).expand(-1, traj_length).reshape(-1)
            # For terrain motions with global_rh: root_h = root_pos_z (already absolute).
            # For non-terrain motions with global_rh: root_h = root_pos_z + sim_terrain_z.
            new_root_h = root_pos_z.clone()
            if terrain_z_fn is not None and is_t is not None:
                nt_grh = grh_flat & (~is_t.unsqueeze(1).expand(-1, traj_length).reshape(-1))
                if nt_grh.any():
                    # Get motion root XY after offset for non-terrain motions.
                    root_pos_xy = self._cached_root_pos_dev[global_nxt, :2]
                    # Apply the same heading rotation + XY offset as ref viz.
                    anchor_idx = self._tracking_anchor_global_idx()
                    anchor_xy = self._cached_root_pos_dev[anchor_idx, :2]  # [B, 2]
                    anchor_xy_flat = anchor_xy.unsqueeze(1).expand(-1, traj_length, -1).reshape(-1, 2)
                    delta_xy = root_pos_xy - anchor_xy_flat
                    if self._tracking_robot_xy is not None and self._tracking_heading_delta is not None:
                        hd = self._tracking_heading_delta.unsqueeze(1).expand(-1, traj_length).reshape(-1)
                        cos_d = torch.cos(hd)
                        sin_d = torch.sin(hd)
                        dx, dy = delta_xy[:, 0], delta_xy[:, 1]
                        rot_xy = torch.stack([cos_d * dx - sin_d * dy,
                                              sin_d * dx + cos_d * dy], dim=-1)
                        rxy_flat = self._tracking_robot_xy.unsqueeze(1).expand(-1, traj_length, -1).reshape(-1, 2)
                        world_xy = rxy_flat + rot_xy
                    else:
                        world_xy = root_pos_xy
                    tz = terrain_z_fn(world_xy[nt_grh])
                    new_root_h[nt_grh] = new_root_h[nt_grh] + tz
            # Apply: replace priv[:, 0] where terrain_variant_root_h is set.
            priv = priv.clone()
            priv[grh_flat, 0] = new_root_h[grh_flat]
            next_obs["privileged_state"] = priv
        next_obs = self.policy._normalize(next_obs)
        z = self.policy._backward_map(next_obs)
        z = z.view(batch_dim, traj_length, z.shape[-1])

        # Variable-T rolling mean: per-env window from self._tracking_T
        T_per_env = getattr(self, "_tracking_T", None)
        if T_per_env is not None and T_per_env.shape[0] == batch_dim:
            # Vectorized cumsum approach — no loops
            d = z.shape[-1]
            cumz = torch.cat([torch.zeros(batch_dim, 1, d, device=z.device), torch.cumsum(z, dim=1)], dim=1)
            steps = torch.arange(traj_length, device=z.device)
            end = (steps.unsqueeze(0) + T_per_env.unsqueeze(1)).clamp(max=traj_length)  # [B, T_len]
            window = (end - steps.unsqueeze(0)).float().unsqueeze(-1)  # [B, T_len, 1]
            arange_B = torch.arange(batch_dim, device=z.device).unsqueeze(1)
            start_sum = cumz[arange_B, steps.unsqueeze(0)]  # [B, T_len, d]
            end_sum = cumz[arange_B, end]  # [B, T_len, d]
            z = (end_sum - start_sum) / window

            if self.cfg.soft_fb:
                norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                z = z / (norm + 1.0)
            else:
                z = self.policy.project_z(z)
        else:
            # Fallback: fixed seq_length (original behavior)
            if self.cfg.soft_fb:
                for step in range(traj_length):
                    end_idx = min(step + seq_length, traj_length)
                    z_mean = z[:, step:end_idx].mean(dim=1)
                    norm = z_mean.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    z[:, step] = z_mean / (norm + 1.0)
            else:
                for step in range(traj_length):
                    end_idx = min(step + seq_length, traj_length)
                    z[:, step] = z[:, step:end_idx].mean(dim=1)
                z = self.policy.project_z(z)
        return z

    def _soft_fb_scale_z(
        self, z: torch.Tensor, alpha: float = 5.0, beta: float = 2.0,
        normalize_first: bool = True,
    ) -> torch.Tensor:
        """Scale z by a Beta-sampled norm.

        Args:
            z: input z tensor [N, d].
            alpha, beta: Beta distribution parameters.
            normalize_first: if True, project to sphere (‖z‖=R) before
                scaling. If False, multiply the existing z directly
                (preserves the pre-existing norm structure).
        """
        if normalize_first:
            R = math.sqrt(z.shape[-1])
            z = R * F.normalize(z, dim=-1)
        n = z.shape[0]
        beta_dist = torch.distributions.Beta(
            torch.tensor(alpha, device=z.device),
            torch.tensor(beta, device=z.device),
        )
        scale = beta_dist.sample((n, 1))
        return z * scale

    def _to_device(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        if isinstance(obs, dict):
            return {k: v.to(self.device, non_blocking=True) for k, v in obs.items()}
        return obs.to(self.device, non_blocking=True)

    @staticmethod
    def _unwrap(net):
        """Return the underlying module — strips DDP if wrapped.

        DDP forwards ``__call__`` but NOT arbitrary attribute access
        (e.g. ``.num_parallel``). Route attribute reads through
        ``_unwrap(net)`` so the 5 DDP-wrapped nets and the unwrapped
        discriminator behave identically from the algorithm's POV.
        """
        return net.module if hasattr(net, "module") else net

    # --- update surface ----------------------------------------------------- #

    def broadcast_parameters(self) -> None:
        """Broadcast ALL policy parameters + buffers from rank 0 to every rank.

        Called once by the runner at init so every rank starts from the
        same weights. Also includes the EMA / BatchNorm running stats.
        """
        if not self.is_distributed:
            return
        objs = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(objs, src=0)
        self.policy.load_state_dict(objs[0])

    @torch.no_grad()
    def _sync_running_stats(self) -> None:
        """Average BatchNorm + EMA running buffers across ranks.

        Called once per ``update()`` after all backward passes are done.
        Fuses every float buffer into ONE all_reduce (was O(N_buffers)
        small NCCL calls per update — ~15 buffers × 16 updates ≈ 240
        kernel launches per iter, which dominated learn time on DDP).
        """
        if not self.is_distributed:
            return
        world = float(torch.distributed.get_world_size())
        all_bufs = list(self.policy._obs_normalizer.buffers()) + \
                   list(self.policy._aux_reward_normalizer.buffers())

        float_bufs = [b for b in all_bufs if b is not None and b.dtype != torch.long]
        int_bufs = [b for b in all_bufs if b is not None and b.dtype == torch.long]

        # Fused SUM-reduce for all float buffers, then /world.
        if float_bufs:
            flat = torch.cat([b.view(-1) for b in float_bufs])
            torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
            flat.div_(world)
            offset = 0
            for b in float_bufs:
                n = b.numel()
                b.view(-1).copy_(flat[offset: offset + n])
                offset += n

        # Fused MAX-reduce for integer counters (BatchNorm's num_batches_tracked).
        if int_bufs:
            flat = torch.cat([b.view(-1).to(torch.float32) for b in int_bufs])
            torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.MAX)
            offset = 0
            for b in int_bufs:
                n = b.numel()
                b.view(-1).copy_(flat[offset: offset + n].to(b.dtype))
                offset += n

    def _anneal_lrs(self, step: int) -> Dict[str, float]:
        """Linearly anneal every optimizer's LR from the DDP-scaled start
        down to the un-scaled base LR over ``cfg.lr_anneal_steps``
        env-steps. Returns the current LR per branch for logging. No-op
        (returns the start LRs) when ``cfg.lr_anneal_enable=False`` or
        ``cfg.lr_anneal_steps <= 0``.

        ``step`` is the global env-step counter threaded in from the
        runner (``tot_timesteps`` — summed across ranks), so the anneal
        schedule is the same regardless of world_size.
        """
        if not bool(self.cfg.lr_anneal_enable):
            return dict(self._start_lrs)
        total = int(self.cfg.lr_anneal_steps)
        if total <= 0:
            return dict(self._start_lrs)
        alpha = max(0.0, min(1.0, float(step) / float(total)))
        out: Dict[str, float] = {}
        for name, opt in (
            ("actor", self.actor_optimizer),
            ("critic", self.critic_optimizer),
            ("aux_critic", self.aux_critic_optimizer),
            ("f", self.forward_optimizer),
            ("b", self.backward_optimizer),
            ("discriminator", self.discriminator_optimizer),
        ):
            lr = (1.0 - alpha) * self._start_lrs[name] + alpha * self._base_lrs[name]
            for g in opt.param_groups:
                g["lr"] = lr
            out[name] = lr
        return out

    def update(self, replay_buffer: Dict[str, Any], step: int) -> Dict[str, torch.Tensor]:
        """One full FB-CPR-Aux update step.

        Expects ``replay_buffer`` to contain:
          - ``"train"``: main replay, with ``.sample(batch_size)`` returning a
            dict like the one produced by the runner (`observation`, `action`,
            `z`, `next`, `aux_rewards`, etc.).
          - ``"expert_slicer"``: expert buffer, with ``.sample(batch_size)``
            returning at least `observation` and `next.observation`.
        """
        current_lrs = self._anneal_lrs(step)

        expert_batch = replay_buffer[self._EXPERT_KEY].sample(self._disc_batch_size)
        train_batch = replay_buffer[self._REPLAY_KEY].sample(self.cfg.batch_size)

        train_obs = self._to_device(train_batch["observation"])
        train_next_obs = self._to_device(train_batch["next"]["observation"])
        train_action = train_batch["action"].to(self.device, non_blocking=True)
        train_terminated = train_batch["next"]["terminated"].to(self.device, non_blocking=True)
        not_term = (~train_terminated.bool()).float()
        discount = self.cfg.discount * not_term
        # Separate aux/disc discounts; default to main discount when None.
        _disc_aux = self.cfg.discount if self.cfg.discount_aux is None else self.cfg.discount_aux
        _disc_disc = self.cfg.discount if self.cfg.discount_disc is None else self.cfg.discount_disc
        discount_aux = float(_disc_aux) * not_term
        discount_disc = float(_disc_disc) * not_term

        expert_obs = self._to_device(expert_batch["observation"])
        expert_next_obs = self._to_device(expert_batch["next"]["observation"])

        # Update obs-normalizer running stats on the train batch.
        self.policy._obs_normalizer(train_obs)
        self.policy._obs_normalizer(train_next_obs)

        # Freeze normalizer momentum for downstream passes.
        with torch.no_grad(), eval_mode(self.policy._obs_normalizer):
            train_obs = self.policy._obs_normalizer(train_obs)
            train_next_obs = self.policy._obs_normalizer(train_next_obs)
            expert_obs = self.policy._obs_normalizer(expert_obs)
            expert_next_obs = self.policy._obs_normalizer(expert_next_obs)

        # Encode expert → z_expert (+ disc validity mask for variable T)
        expert_z, expert_disc_mask = self.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"].to(self.device, non_blocking=True)

        # BFM order: disc sees ORIGINAL train_z (from rollout), THEN relabel.
        # The discriminator must train on the actual (s, z) pairs from the
        # replay — not freshly sampled z's that were never rolled out.
        disc_train_z = train_z

        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self._zbuf_add(z)
        if self.cfg.relabel_ratio is not None:
            mask = torch.rand(
                (self.cfg.batch_size, 1), device=self.device
            ) <= self.cfg.relabel_ratio
            train_z = torch.where(mask, z, train_z)

        # --- Anchoring seam (Global-through-Anchoring) -----------------
        # Default is identity: ``fb_goal`` is the transition's own next obs
        # (the FB successor-query state s_+) and obs/z pass through unchanged,
        # so non-anchored tasks are byte-identical. The anchored subclass
        # overrides ``_anchor_relabel`` to: sample a coordinate anchor A and an
        # INDEPENDENT successor query s_+, inject the anchored pose A^-1 g into
        # obs/next_obs/goal, sample the task goal s_h ~ p_goal, and set z's
        # spatial block = B_spatial(anchored s_h).
        fb_goal = train_next_obs
        train_obs, train_next_obs, fb_goal, train_z = self._anchor_relabel(
            train_batch=train_batch,
            train_obs=train_obs,
            train_next_obs=train_next_obs,
            train_z=train_z,
            mixed_z=z,
            expert_z=expert_z,
        )

        q_loss_coef = self.cfg.q_loss_coef if self.cfg.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.clip_grad_norm if self.cfg.clip_grad_norm > 0 else None

        # Assemble aux_reward up-front (needed for aux_critic backward in phase 1).
        aux_reward = torch.zeros(
            (self.cfg.batch_size, 1), device=self.device, dtype=torch.float32
        )
        aux_batch = train_batch.get("aux_rewards", None)
        aux_rew_logs: Dict[str, torch.Tensor] = {}
        if aux_batch is not None and len(self.cfg.aux_rewards_scaling) > 0:
            for name, scale in self.cfg.aux_rewards_scaling.items():
                if name not in aux_batch:
                    continue
                vals = aux_batch[name].to(self.device, non_blocking=True).view(-1, 1)
                aux_rew_logs[f"aux_rew/{name}"] = vals.mean().detach()
                aux_reward = aux_reward + scale * vals
        # Pass through EMA reward normalizer (BFM's `RewardNormalizer(scale=True)`).
        aux_reward = self.policy._aux_reward_normalizer(aux_reward)

        # =============================================================
        # PHASE 1: disc + F/B + aux_critic all have NO data dependency
        # on each other's updated weights.
        #
        # Two reduce strategies:
        #   (a) DDP bucket hooks fire async all_reduce during backward,
        #       overlapping comm with compute on the next network's
        #       backward. Best on NVSwitch / fast intra-node fabrics
        #       where bandwidth is plentiful and latency is negligible.
        #   (b) Merge all 4 allreduces into ONE by wrapping DDP in
        #       no_sync() and reducing manually at phase end. Saves
        #       NCCL per-call latency — wins on slow/high-latency
        #       fabrics (EFA without GDR, cross-node IB).
        #
        # On B200 + NVSwitch (intra-node): strategy (a) is faster.
        # On EFA without GDR / cross-node bandwidth-limited: (b) wins.
        # Controlled by ``merge_phase1_reduce`` on the algorithm cfg;
        # default False (favor overlap) which is correct for modern
        # GPU clusters with NVSwitch.
        # =============================================================
        self._merge_phase1_reduce = bool(
            self.is_distributed
            and self._is_ddp_wrapped
            and getattr(self.cfg, "merge_phase1_reduce", False)
        )

        if self._merge_phase1_reduce:
            phase1_ddp_nets = [
                self.policy._forward_map,
                self.policy._backward_map,
                self.policy._aux_critic,
            ]
            phase1_ctx = contextlib.ExitStack()
            for net in phase1_ddp_nets:
                phase1_ctx.enter_context(net.no_sync())
        else:
            phase1_ctx = contextlib.nullcontext()

        # Stream-parallel uses three dedicated CUDA streams so the three
        # backward passes run concurrently. Each stream waits on the
        # current (default) stream so the upstream tensors (train_obs,
        # aux_reward, etc.) are visible. After all three finish, the
        # default stream waits on them before the merged allreduce runs.
        if self._stream_parallel_phase1:
            cur_stream = torch.cuda.current_stream()
            self._phase1_stream_disc.wait_stream(cur_stream)
            self._phase1_stream_fb.wait_stream(cur_stream)
            self._phase1_stream_aux.wait_stream(cur_stream)

        with phase1_ctx:
            # When merge is off, ``backward_discriminator`` fires its
            # own async reduce and returns a handle — we must wait on
            # it before step. When merge is on the handle is None and
            # the merged reduce below handles disc's grads too.
            if self._stream_parallel_phase1:
                with torch.cuda.stream(self._phase1_stream_disc):
                    disc_metrics, disc_handle = self.backward_discriminator(
                        expert_obs=expert_obs,
                        expert_z=expert_z,
                        train_obs=train_obs,
                        train_z=disc_train_z,
                        grad_penalty=self.cfg.grad_penalty_discriminator
                        if self.cfg.grad_penalty_discriminator > 0
                        else None,
                        expert_mask=expert_disc_mask,
                    )
                    # Manifold attractor on same stream as disc (no dependency).
                    if self.cfg.manifold_attractor:
                        ma_metrics = self.backward_manifold_attractor(
                            expert_obs=expert_obs,
                            train_obs=train_obs,
                        )
                        disc_metrics.update(ma_metrics)
            else:
                disc_metrics, disc_handle = self.backward_discriminator(
                    expert_obs=expert_obs,
                    expert_z=expert_z,
                    train_obs=train_obs,
                    train_z=disc_train_z,
                    grad_penalty=self.cfg.grad_penalty_discriminator
                    if self.cfg.grad_penalty_discriminator > 0
                    else None,
                    expert_mask=expert_disc_mask,
                )
                if self.cfg.manifold_attractor:
                    ma_metrics = self.backward_manifold_attractor(
                        expert_obs=expert_obs,
                        train_obs=train_obs,
                    )
                    disc_metrics.update(ma_metrics)

            # DDP bucket hooks on F, B, aux_critic fire async allreduce
            # during backward; they return None handles and are waited
            # on internally by DDP. Merge mode is in no_sync(), so
            # those hooks are suppressed and the merged reduce covers
            # them.
            if self._stream_parallel_phase1:
                with torch.cuda.stream(self._phase1_stream_fb):
                    fb_metrics, _, _ = self.backward_fb(
                        obs=train_obs,
                        action=train_action,
                        discount=discount,
                        next_obs=train_next_obs,
                        goal=fb_goal,
                        z=train_z,
                        q_loss_coef=q_loss_coef,
                    )
                with torch.cuda.stream(self._phase1_stream_aux):
                    aux_metrics, _ = self.backward_aux_critic(
                        obs=train_obs,
                        action=train_action,
                        discount=discount_aux,
                        aux_reward=aux_reward,
                        next_obs=train_next_obs,
                        z=train_z,
                    )
            else:
                fb_metrics, _, _ = self.backward_fb(
                    obs=train_obs,
                    action=train_action,
                    discount=discount,
                    next_obs=train_next_obs,
                    goal=fb_goal,
                    z=train_z,
                    q_loss_coef=q_loss_coef,
                )
                aux_metrics, _ = self.backward_aux_critic(
                    obs=train_obs,
                    action=train_action,
                    discount=discount_aux,
                    aux_reward=aux_reward,
                    next_obs=train_next_obs,
                    z=train_z,
                )

        # Rejoin the streams back onto the default stream. The merged
        # allreduce below runs on the default stream and must see
        # finalized grads from all three backward streams.
        if self._stream_parallel_phase1:
            cur_stream = torch.cuda.current_stream()
            cur_stream.wait_stream(self._phase1_stream_disc)
            cur_stream.wait_stream(self._phase1_stream_fb)
            cur_stream.wait_stream(self._phase1_stream_aux)

        # One merged all_reduce across {disc, F, B, aux_critic} grads
        # (only when merge strategy is enabled).
        if self._merge_phase1_reduce:
            merged_handle = reduce_gradients_merged_async([
                self.policy._discriminator,  # not DDP-wrapped; use manually
                # For DDP-wrapped nets iterate .module so we target the real
                # parameters (grads live on the same tensors either way, but
                # .module.parameters() is the clean path).
                self._unwrap(self.policy._forward_map),
                self._unwrap(self.policy._backward_map),
                self._unwrap(self.policy._aux_critic),
            ])
            finish_merged_async_reduce(merged_handle)
            disc_handle = None  # merged reduce covered disc too

        # Step all three optimizers. ``step_fb`` still applies grad
        # clipping before optimizer.step(). Each step returns a
        # grad_norm/* dict we merge into metrics.
        disc_gn = self.step_discriminator(disc_handle)
        fb_gn = self.step_fb(None, None, clip_grad_norm)
        aux_gn = self.step_aux_critic(None)

        # Clear the merge flag so nested / repeat calls (there shouldn't
        # be any) don't carry stale state.
        self._merge_phase1_reduce = False

        metrics = {}
        metrics.update(disc_metrics)
        metrics.update(fb_metrics)
        metrics.update(aux_metrics)
        metrics.update(aux_rew_logs)
        metrics.update(disc_gn)
        metrics.update(fb_gn)
        metrics.update(aux_gn)

        # =============================================================
        # PHASE 2: critic. Depends on NEW disc (for disc_reward), so
        # must follow phase 1.
        # =============================================================
        critic_metrics, critic_handle = self.backward_critic(
            obs=train_obs,
            action=train_action,
            discount=discount_disc,
            next_obs=train_next_obs,
            z=train_z,
        )
        # Entropy critic has no dependency on the just-updated critic
        # (it uses target networks), so run it while critic grads reduce.
        if self.cfg.soft_fb:
            ec_metrics = self.backward_entropy_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                z=train_z,
            )
            metrics.update(ec_metrics)
        critic_gn = self.step_critic(critic_handle)
        metrics.update(critic_metrics)
        metrics.update(critic_gn)

        # =============================================================
        # PHASE 3: actor. Depends on NEW critic / aux_critic / F for
        # its Q targets.
        # =============================================================
        actor_metrics, actor_handle = self.backward_actor(
            obs=train_obs,
            action=train_action,
            z=train_z,
        )
        actor_gn = self.step_actor(actor_handle, clip_grad_norm)
        metrics.update(actor_metrics)
        metrics.update(actor_gn)

        # 7) Polyak soft updates
        with torch.no_grad():
            _soft_update_params(
                self._forward_map_params,
                self._target_forward_map_params,
                self.cfg.fb_target_tau,
            )
            _soft_update_params(
                self._backward_map_params,
                self._target_backward_map_params,
                self.cfg.fb_target_tau,
            )
            _soft_update_params(
                self._critic_params,
                self._target_critic_params,
                self.cfg.critic_target_tau,
            )
            _soft_update_params(
                self._aux_critic_params,
                self._target_aux_critic_params,
                self.cfg.critic_target_tau,
            )
            if self._entropy_critic_params:
                _soft_update_params(
                    self._entropy_critic_params,
                    self._target_entropy_critic_params,
                    self.cfg.entropy_critic_target_tau,
                )

        # NOTE: running-stat sync across ranks is NOT done here. It happens
        # once per learn-iter in ``FBCprRunner`` (after all num_agent_updates
        # backward passes), so per-update drift inside one iter is acceptable
        # and we save 15× the NCCL traffic.

        # Emit per-branch LR when anneal is active. One scalar per branch;
        # useful for confirming the schedule hits base_lr by the target
        # step (and as a sanity check that the knob is being honored).
        if bool(self.cfg.lr_anneal_enable) and int(self.cfg.lr_anneal_steps) > 0:
            for name, lr in current_lrs.items():
                metrics[f"lr/{name}"] = torch.tensor(lr, device=self.device)

        return metrics

    # --- individual update blocks ------------------------------------------ #

    def backward_discriminator(
        self,
        expert_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        train_obs: torch.Tensor | dict[str, torch.Tensor],
        train_z: torch.Tensor,
        grad_penalty: float | None,
        expert_mask: torch.Tensor | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], Any]:
        """Compute disc loss, backward, fire async reduce. Returns (metrics, reduce_handle).

        ``expert_mask``: optional [batch_size] bool, True for valid expert
        (s, z) pairs (frame within T-window). Invalid frames are excluded
        from expert_loss. Train loss uses full batch.
        """
        disc = self.policy._discriminator
        if expert_mask is not None:
            if isinstance(expert_obs, dict):
                expert_obs_valid = {k: v[expert_mask] for k, v in expert_obs.items()}
            else:
                expert_obs_valid = expert_obs[expert_mask]
            expert_z_valid = expert_z[expert_mask]
        else:
            expert_obs_valid = expert_obs
            expert_z_valid = expert_z

        # Merged forward: concat real + fake, run once, split logits.
        n_real = expert_z_valid.shape[0]
        if isinstance(expert_obs_valid, dict):
            merged_obs = {k: torch.cat([expert_obs_valid[k], train_obs[k]], dim=0)
                          for k in expert_obs_valid}
        else:
            merged_obs = torch.cat([expert_obs_valid, train_obs], dim=0)
        merged_z = torch.cat([expert_z_valid, train_z], dim=0)
        merged_logits = disc.compute_logits(merged_obs, merged_z)
        expert_logits = merged_logits[:n_real]
        unlabeled_logits = merged_logits[n_real:]
        expert_loss = -F.logsigmoid(expert_logits)
        unlabeled_loss = F.softplus(unlabeled_logits)
        loss = expert_loss.mean() + unlabeled_loss.mean()

        wgan_gp = None
        if grad_penalty is not None:
            # GP needs equal expert/train counts; if expert was filtered,
            # the train side stays full so subsample train to match.
            if expert_mask is not None:
                n_valid = int(expert_mask.sum().item())
                if isinstance(train_obs, dict):
                    first_key = next(iter(train_obs))
                    train_n = train_obs[first_key].shape[0]
                else:
                    train_n = train_obs.shape[0]
                idx = torch.randperm(train_n, device=train_z.device)[:n_valid]
                if isinstance(train_obs, dict):
                    train_obs_gp = {k: v[idx] for k, v in train_obs.items()}
                else:
                    train_obs_gp = train_obs[idx]
                train_z_gp = train_z[idx]
                wgan_gp = self._gradient_penalty_wgan(
                    expert_obs_valid, expert_z_valid, train_obs_gp, train_z_gp,
                )
            else:
                wgan_gp = self._gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
            loss = loss + grad_penalty * wgan_gp

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # When ``_merge_phase1_reduce`` is set, the caller issues ONE merged
        # all_reduce across disc + F + B + aux_critic grads, so skip the
        # per-network reduce here. Otherwise fall back to the legacy
        # manual async reduce.
        if getattr(self, "_merge_phase1_reduce", False):
            handle = None
        else:
            handle = reduce_gradients_async(self.policy._discriminator) if self.is_distributed else None

        with torch.no_grad():
            out = {
                "disc_loss": loss.detach(),
                "disc_expert_loss": expert_loss.detach().mean(),
                "disc_train_loss": unlabeled_loss.detach().mean(),
            }
            if wgan_gp is not None:
                out["disc_wgan_gp_loss"] = wgan_gp.detach()
        return out, handle

    # --- Manifold attractor ------------------------------------------------- #

    def backward_manifold_attractor(
        self,
        expert_obs: torch.Tensor | dict[str, torch.Tensor],
        train_obs: torch.Tensor | dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Train D_ma(s) — unconditional state discriminator."""
        p = self.policy
        if p._manifold_attractor is None:
            return {}
        ma = p._manifold_attractor
        # Equalize and cap both sides for the GP (which requires equal
        # batch sizes for the alpha interpolation).
        ma_cap = int(getattr(self.cfg, "ma_max_batch", 1024))
        n_e = next(iter(expert_obs.values())).shape[0] if isinstance(expert_obs, dict) else expert_obs.shape[0]
        n_t = next(iter(train_obs.values())).shape[0] if isinstance(train_obs, dict) else train_obs.shape[0]
        target = min(ma_cap, n_e, n_t)

        def _resample(o, n, k):
            if n == k:
                return o
            idx = torch.randperm(n, device=self.device)[:k]
            if isinstance(o, dict):
                return {kk: v[idx] for kk, v in o.items()}
            return o[idx]

        expert_obs = _resample(expert_obs, n_e, target)
        train_obs = _resample(train_obs, n_t, target)
        n_real = target
        if isinstance(expert_obs, dict):
            merged_obs = {k: torch.cat([expert_obs[k], train_obs[k]], dim=0) for k in expert_obs}
        else:
            merged_obs = torch.cat([expert_obs, train_obs], dim=0)
        merged_logits = ma.compute_logits(merged_obs)
        expert_logits = merged_logits[:n_real]
        train_logits = merged_logits[n_real:]
        expert_loss = -F.logsigmoid(expert_logits)
        train_loss = F.softplus(train_logits)
        loss = expert_loss.mean() + train_loss.mean()

        if self.cfg.grad_penalty_manifold_attractor > 0:
            gp = self._gradient_penalty_manifold_attractor(expert_obs, train_obs)
            loss = loss + self.cfg.grad_penalty_manifold_attractor * gp

        self.manifold_attractor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ma.parameters(), float("inf"))
        self.manifold_attractor_optimizer.step()
        return {
            "ma_loss": loss.detach(),
            "ma_expert_loss": expert_loss.mean().detach(),
            "ma_train_loss": train_loss.mean().detach(),
        }

    def _gradient_penalty_manifold_attractor(
        self,
        real_obs, fake_obs,
    ) -> torch.Tensor:
        """WGAN-GP on manifold attractor."""
        ma = self.policy._manifold_attractor
        obs_filter = ma.input_filter
        real_o = obs_filter(real_obs)
        fake_o = obs_filter(fake_obs)
        alpha = torch.rand(real_o.shape[0], 1, device=real_o.device)
        interp = (alpha * real_o + (1 - alpha) * fake_o).requires_grad_(True)
        d_interp = ma.trunk(interp)
        grad = autograd.grad(
            d_interp, interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True,
        )[0]
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def step_discriminator(self, handle: Any) -> Dict[str, torch.Tensor]:
        finish_async_reduce(handle)
        # ``clip_grad_norm_(..., max_norm=inf)`` is a no-op clip that
        # returns the pre-clip L2 grad norm for free; avoids a second
        # pass-over-params that a manual sum-of-squares would need.
        gn = torch.nn.utils.clip_grad_norm_(
            self.policy._discriminator.parameters(), float("inf"),
        )
        self.discriminator_optimizer.step()
        return {"grad_norm/discriminator": gn.detach()}

    def _gradient_penalty_wgan(
        self,
        real_obs: torch.Tensor | dict[str, torch.Tensor],
        real_z: torch.Tensor,
        fake_obs: torch.Tensor | dict[str, torch.Tensor],
        fake_z: torch.Tensor,
    ) -> torch.Tensor:
        # Interpolate each tensor (obs is a dict of concat-able tensors).
        if isinstance(real_obs, torch.Tensor):
            bs = real_obs.shape[0]
            interp_obs_list = []
            alpha = torch.rand(bs, 1, device=real_z.device)
            interp = (alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True)
            interp_obs: torch.Tensor | dict[str, torch.Tensor] = interp
            interp_obs_list.append(interp)
        else:
            bs = next(iter(real_obs.values())).shape[0]
            alpha = torch.rand(bs, 1, device=real_z.device)
            interp_obs = {}
            interp_obs_list = []
            for k in real_obs.keys():
                if k not in fake_obs:
                    continue
                ro = real_obs[k]
                fo = fake_obs[k]
                interp_obs[k] = (alpha * ro + (1 - alpha) * fo).requires_grad_(True)
                interp_obs_list.append(interp_obs[k])

        interp_z = (alpha * real_z + (1 - alpha) * fake_z).requires_grad_(True)

        d_interp = self.policy._discriminator.compute_logits(interp_obs, interp_z)
        grads = autograd.grad(
            outputs=d_interp,
            inputs=interp_obs_list + [interp_z],
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )
        grads = [g for g in grads if g is not None]
        cat_g = torch.cat(grads, dim=1)
        return ((cat_g.norm(2, dim=1) - 1) ** 2).mean()

    def _discriminator_opt_params(self):
        """Parameters optimized by the discriminator optimizer. Base returns
        the single CPR discriminator; the anchored subclass appends the
        spatial discriminator's params."""
        return self.policy._discriminator.parameters()

    def _cpr_reward(self, obs, z):
        """CPR (style) reward fed to the disc-reward critic. Base = the single
        discriminator's log-odds reward. The anchored subclass adds a spatial
        discriminator channel."""
        return self.policy._discriminator.compute_reward(obs, z)

    def _anchor_relabel(self, train_batch, train_obs, train_next_obs, train_z,
                        mixed_z, expert_z):
        """Anchoring seam (no-op in the base FB-CPR-Aux).

        The anchored subclass overrides this to relabel coordinate anchors,
        the FB successor-query state, and the task goal. The base returns
        ``(obs, next_obs, fb_goal=next_obs, z)`` unchanged so non-anchored
        tasks are byte-identical.
        """
        return train_obs, train_next_obs, train_next_obs, train_z

    def backward_fb(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        goal: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        q_loss_coef: float | None,
    ) -> Tuple[Dict[str, torch.Tensor], Any, Any]:
        """Compute FB loss, backward, fire async reduces on F and B.

        Returns ``(metrics, F_handle, B_handle)``.
        """
        p = self.policy
        with torch.no_grad():
            # next_action via actor
            dist = p._actor(next_obs, z, p.actor_std)
            next_action = dist.sample(clip=self.cfg.stddev_clip)
            target_Fs = p._target_forward_map(next_obs, z, next_action)  # (num_par, B, d)
            target_B = p._target_backward_map(goal)  # (B, d)
            target_Ms = torch.matmul(target_Fs, target_B.T)  # (num_par, B, B)
            _, _, target_M = self._pessimistic_value(target_Ms, self.cfg.fb_pessimism_penalty)

        Fs = p._forward_map(obs, z, action)
        B = p._backward_map(goal)
        Ms = torch.matmul(Fs, B.T)

        diff = Ms - discount.view(-1, 1) * target_M
        fb_offdiag = 0.5 * (diff * self._off_diag).pow(2).sum() / self._off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
        fb_loss = fb_offdiag + fb_diag

        # Orthonormality loss on B: ||E[B(s)B(s)^T] - I||^2
        if self.cfg.soft_fb:
            # Feature covariance [z_dim, z_dim] — push toward identity.
            BtB = torch.matmul(B.T, B) / B.shape[0]  # [d, d]
            diff = BtB - torch.eye(B.shape[1], device=B.device)
            orth_loss = diff.pow(2).mean()
            orth_loss_diag = diff.diag().pow(2).mean()
            d = B.shape[1]
            orth_loss_offdiag = (diff.pow(2).sum() - diff.diag().pow(2).sum()) / (d * (d - 1))
        else:
            # Legacy batch gram matrix for standard FB.
            Cov = torch.matmul(B, B.T)
            orth_loss_diag = -Cov.diag().mean()
            orth_loss_offdiag = 0.5 * (Cov * self._off_diag).pow(2).sum() / self._off_diag_sum
            orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss = fb_loss + self.cfg.ortho_coef * orth_loss

        # Reconstruction regulariser: decode end-effector (or any
        # configured) slices of ``goal`` from ``z = B(goal)`` and minimise
        # MSE. Pushes B to preserve task-relevant spatial info that the
        # bare FB loss may collapse out of z.
        recon_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        recon_head = getattr(p, "_reconstruction_head", None)
        recons_coeff = float(self.cfg.recons_coeff)
        if recon_head is not None and recons_coeff > 0 and isinstance(goal, dict):
            pred = recon_head(B)
            target = recon_head.gather_target(goal).detach()
            recon_loss = F.mse_loss(pred, target)
            fb_loss = fb_loss + recons_coeff * recon_loss

        q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
        if q_loss_coef is not None:
            with torch.no_grad():
                next_Qs = (target_Fs * z).sum(dim=-1)
                _, _, next_Q = self._pessimistic_value(next_Qs, self.cfg.fb_pessimism_penalty)
                cov = torch.matmul(B.T, B) / B.shape[0]
                B_inv_cov = torch.linalg.solve(cov, B, left=False)
                implicit_reward = (B_inv_cov * z).sum(dim=-1)
                target_Q = implicit_reward.detach() + discount.squeeze(-1) * next_Q
                expanded = target_Q.expand(Fs.shape[0], -1)
            Qs = (Fs * z).sum(dim=-1)
            q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded)
            fb_loss = fb_loss + q_loss_coef * q_loss

        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        # DDP on F / B already fired async all_reduce INSIDE backward via
        # bucket hooks; nothing to fire here. Leave handles as None so
        # ``step_fb`` becomes a plain opt.step().
        F_handle = None
        B_handle = None

        with torch.no_grad():
            out = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B.mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "q_loss": q_loss,
                "recon_loss": recon_loss,
            }
        return out, F_handle, B_handle

    def step_fb(
        self, F_handle: Any, B_handle: Any, clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        p = self.policy
        finish_async_reduce(F_handle)
        finish_async_reduce(B_handle)
        # ``clip_grad_norm_`` returns the pre-clip total L2 norm. Use the
        # configured max_norm if clipping is on; otherwise use inf as a
        # no-op clip that still returns the norm (cheaper than a manual
        # sum-of-squares second pass).
        max_norm = float(clip_grad_norm) if clip_grad_norm is not None else float("inf")
        gn_f = torch.nn.utils.clip_grad_norm_(p._forward_map.parameters(), max_norm)
        gn_b = torch.nn.utils.clip_grad_norm_(p._backward_map.parameters(), max_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()
        return {
            "grad_norm/forward_map": gn_f.detach(),
            "grad_norm/backward_map": gn_b.detach(),
        }

    def backward_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Any]:
        p = self.policy
        pol_cfg = self._unwrap(p).cfg
        distributional = bool(getattr(pol_cfg, "critic_distributional", False))
        num_parallel = self._unwrap(p._critic).num_parallel
        _ma_reward = None
        with torch.no_grad():
            reward = self._cpr_reward(obs, z)
            # Manifold attractor: add unconditional state reward.
            if self.cfg.manifold_attractor and p._manifold_attractor is not None:
                _ma_reward = p._manifold_attractor.compute_reward(obs)
                reward = reward + self.cfg.manifold_attractor_coeff * _ma_reward
            dist = p._actor(next_obs, z, p.actor_std)
            next_action = dist.sample(clip=self.cfg.stddev_clip)
            next_Qs = p._target_critic(next_obs, z, next_action)
            # Shapes:
            #   scalar critic:       next_Qs = [num_parallel, batch, 1]
            #   distributional QR:   next_Qs = [num_parallel, batch, n_q]
            if distributional:
                # Ensemble-reduce to scalar for pessimism computation, then
                # apply a uniform shift across all target quantiles. The
                # target distribution keeps its shape from the target net,
                # just pulled down by ``penalty * unc``.
                next_Qs_scalar = next_Qs.mean(dim=-1)                 # [np, batch]
                Q_mean, Q_unc, _ = self._pessimistic_value(
                    next_Qs_scalar, self.cfg.critic_pessimism_penalty,
                )
                next_q_mean = next_Qs.mean(dim=0)                     # [batch, n_q]
                next_V = next_q_mean - self.cfg.critic_pessimism_penalty * Q_unc.unsqueeze(-1)
                target_Q = reward + discount.view(-1, 1) * next_V     # [batch, n_q]
            else:
                Q_mean, Q_unc, next_V = self._pessimistic_value(
                    next_Qs, self.cfg.critic_pessimism_penalty
                )
                target_Q = reward + discount.view(-1, 1) * next_V     # [batch, 1]

        Qs = p._critic(obs, z, action)                                # [np, batch, n_q_or_1]

        if distributional:
            critic_loss = self._quantile_huber_loss(
                Qs, target_Q, kappa=float(pol_cfg.critic_huber_kappa),
            )
        else:
            expanded = target_Q.expand(num_parallel, -1, -1)
            critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        # DDP handled reduce inside backward.
        handle = None

        with torch.no_grad():
            out = {
                "target_Q": target_Q.mean(),
                "Q1": Qs.mean(),
                "mean_next_Q": Q_mean.mean(),
                "unc_Q": Q_unc.mean(),
                "critic_loss": critic_loss.mean(),
                "mean_disc_reward": reward.mean(),
                **({"mean_ma_reward": _ma_reward.mean()} if _ma_reward is not None else {}),
            }
            if distributional:
                # Spread across the quantile axis — if it collapses to ~0
                # the distributional head has degenerated to scalar Q.
                out["critic_q_spread"] = Qs.std(dim=-1).mean()
        return out, handle

    def step_critic(self, handle: Any) -> Dict[str, torch.Tensor]:
        finish_async_reduce(handle)
        gn = torch.nn.utils.clip_grad_norm_(
            self.policy._critic.parameters(), float("inf"),
        )
        self.critic_optimizer.step()
        return {"grad_norm/critic": gn.detach()}

    def backward_aux_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        aux_reward: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Any]:
        p = self.policy
        pol_cfg = self._unwrap(p).cfg
        distributional = bool(getattr(pol_cfg, "aux_critic_distributional", False))
        num_parallel = self._unwrap(p._aux_critic).num_parallel
        with torch.no_grad():
            dist = p._actor(next_obs, z, p.actor_std)
            next_action = dist.sample(clip=self.cfg.stddev_clip)
            next_Qs = p._target_aux_critic(next_obs, z, next_action)
            if distributional:
                next_Qs_scalar = next_Qs.mean(dim=-1)
                Q_mean, Q_unc, _ = self._pessimistic_value(
                    next_Qs_scalar, self.cfg.aux_critic_pessimism_penalty,
                )
                next_q_mean = next_Qs.mean(dim=0)
                next_V = next_q_mean - self.cfg.aux_critic_pessimism_penalty * Q_unc.unsqueeze(-1)
                target_Q = aux_reward + discount.view(-1, 1) * next_V
            else:
                Q_mean, Q_unc, next_V = self._pessimistic_value(
                    next_Qs, self.cfg.aux_critic_pessimism_penalty
                )
                target_Q = aux_reward + discount.view(-1, 1) * next_V

        Qs = p._aux_critic(obs, z, action)

        if distributional:
            aux_critic_loss = self._quantile_huber_loss(
                Qs, target_Q, kappa=float(pol_cfg.aux_critic_huber_kappa),
            )
        else:
            expanded = target_Q.expand(num_parallel, -1, -1)
            aux_critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded)

        self.aux_critic_optimizer.zero_grad(set_to_none=True)
        aux_critic_loss.backward()
        # DDP handled reduce inside backward.
        handle = None

        with torch.no_grad():
            out = {
                "target_auxQ": target_Q.mean(),
                "auxQ1": Qs.mean(),
                "mean_next_auxQ": Q_mean.mean(),
                "unc_auxQ": Q_unc.mean(),
                "aux_critic_loss": aux_critic_loss.mean(),
                "mean_aux_reward": aux_reward.mean(),
            }
            if distributional:
                out["aux_critic_q_spread"] = Qs.std(dim=-1).mean()
        return out, handle

    def step_aux_critic(self, handle: Any) -> Dict[str, torch.Tensor]:
        finish_async_reduce(handle)
        gn = torch.nn.utils.clip_grad_norm_(
            self.policy._aux_critic.parameters(), float("inf"),
        )
        self.aux_critic_optimizer.step()
        return {"grad_norm/aux_critic": gn.detach()}

    # --- Soft FB: entropy critic -------------------------------------------- #

    def backward_entropy_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Train Q_H via soft TD: target = -log_pi(a'|s',z) + γ Q_H_target(s',a',z)."""
        p = self.policy
        if p._entropy_critic is None:
            return {}

        with torch.no_grad():
            next_dist = p._actor(next_obs, z, p.actor_std)
            next_action = next_dist.sample()
            log_pi_next = next_dist.log_prob(next_action).mean(dim=-1)  # [B] per-dim avg
            # _target_entropy_critic returns [1, B, 1] (num_parallel=1)
            target_qh_raw = p._target_entropy_critic(
                next_obs, z, next_action,
            ).squeeze(0).squeeze(-1)  # [B]
            target_qh = -log_pi_next + discount.view(-1) * target_qh_raw

        # [1, B, 1] → [B]
        qh_pred = p._entropy_critic(obs, z, action).squeeze(0).squeeze(-1)
        loss = F.mse_loss(qh_pred, target_qh)

        self.entropy_critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(p._entropy_critic.parameters(), float("inf"))
        self.entropy_critic_optimizer.step()
        return {
            "entropy_critic_loss": loss.detach(),
            "entropy_critic_q_mean": qh_pred.mean().detach(),
            "entropy_critic_target_mean": target_qh.mean().detach(),
        }

    # --- actor --------------------------------------------------------------- #

    def backward_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Any]:
        p = self.policy
        dist = p._actor(obs, z, p.actor_std)
        sampled_action = dist.sample(clip=self.cfg.stddev_clip)

        # --- action saturation diagnostics ------------------------------
        # ``loc`` is the actor's pre-sample mean; TruncatedNormal clamps
        # final samples into [-1+eps, +1-eps]. Saturation happens when
        # |loc| approaches 1 (mean is pushed to the action bound) or when
        # noise regularly drives loc+eps outside the interval.
        with torch.no_grad():
            loc = dist.loc
            act_stats = {
                "act_loc/abs_mean": loc.abs().mean().detach(),
                "act_loc/frac_gt_0_9": (loc.abs() > 0.9).float().mean().detach(),
            }

        # Q from discriminator-reward critic. If the critic is distributional
        # (output shape [num_parallel, batch, n_q]), collapse to scalar Q by
        # averaging the quantile dimension before applying pessimism — the
        # mean of the quantile vector is the distribution's mean, which is
        # the right statistic for the policy-gradient objective.
        pol_cfg = self._unwrap(p).cfg
        Qs_disc = p._critic(obs, z, sampled_action)
        if bool(getattr(pol_cfg, "critic_distributional", False)):
            Qs_disc = Qs_disc.mean(dim=-1, keepdim=True)
        _, _, Q_discriminator = self._pessimistic_value(
            Qs_disc, self.cfg.actor_pessimism_penalty
        )
        # Q from aux-reward critic (same contract).
        Qs_aux = p._aux_critic(obs, z, sampled_action)
        if bool(getattr(pol_cfg, "aux_critic_distributional", False)):
            Qs_aux = Qs_aux.mean(dim=-1, keepdim=True)
        _, _, Q_aux = self._pessimistic_value(Qs_aux, self.cfg.actor_pessimism_penalty)
        # Q from FB (implicit Q = F·z)
        Fs = p._forward_map(obs, z, sampled_action)
        Qs_fb = (Fs * z).sum(dim=-1)
        _, _, Q_fb = self._pessimistic_value(Qs_fb, self.cfg.actor_pessimism_penalty)

        if self.cfg.soft_fb:
            R = 1.0  # soft FB uses unit ball
            z_norms = z.norm(dim=-1)

        if self.cfg.soft_fb and self.cfg.scale_reg:
            z_norm_clamped = z_norms.clamp(min=0.1 * R)
            Q_fb_normalized = Q_fb * (R / z_norm_clamped)
            weight = Q_fb_normalized.abs().mean().detach()
        elif self.cfg.scale_reg:
            weight = Q_fb.abs().mean().detach()
        else:
            weight = 1.0

        if self.cfg.soft_fb and p._entropy_critic is not None:
            beta_z = self.cfg.soft_fb_entropy_coef * (
                1.0 - z_norms / R
            ).clamp(min=0.0)
            log_pi = dist.log_prob(sampled_action).mean(dim=-1)  # [B] per-dim avg
            # Q_H evaluated on the actor's sampled_action (different from
            # the replay action used in backward_entropy_critic). Detached
            # so no gradient flows through the entropy critic.
            Q_H = p._entropy_critic(obs, z, sampled_action).squeeze(0).squeeze(-1).detach()  # [B]
            soft_core_unweighted = (beta_z * (log_pi - Q_H)).mean()
            soft_core = soft_core_unweighted * weight
            actor_loss = (
                soft_core
                - Q_fb.mean()
                - Q_discriminator.mean() * self.cfg.reg_coeff * weight
                - Q_aux.mean() * self.cfg.reg_coeff_aux * weight
            )
            # Stash for logging (detached) — raw values, not weighted.
            self._soft_fb_actor_logs = {
                "z_norm_mean": z_norms.mean().detach(),
                "z_norm_std": z_norms.std().detach(),
                "z_norm_min": z_norms.min().detach(),
                "z_norm_max": z_norms.max().detach(),
                "beta_z_mean": beta_z.mean().detach(),
                "beta_z_std": beta_z.std().detach(),
                "policy_entropy_mean": -log_pi.mean().detach(),
                "log_pi_mean": log_pi.mean().detach(),
                "q_fb_mean": Q_fb.mean().detach(),
                "q_h_mean": Q_H.mean().detach(),
                "soft_actor_core_loss": soft_core_unweighted.detach(),
            }
            if hasattr(p._actor, 'learned_std') and p._actor.learned_std:
                log_std = dist.scale.log()
                self._soft_fb_actor_logs["actor_log_std_mean"] = log_std.mean().detach()
                self._soft_fb_actor_logs["actor_log_std_min"] = log_std.min().detach()
                self._soft_fb_actor_logs["actor_log_std_max"] = log_std.max().detach()
        else:
            actor_loss = (
                -Q_discriminator.mean() * self.cfg.reg_coeff * weight
                - Q_aux.mean() * self.cfg.reg_coeff_aux * weight
                - Q_fb.mean()
            )

        # Anchoring seam: extra actor-side losses (e.g. two-anchor policy-KL
        # consistency). No-op in the base FB-CPR-Aux (returns 0, {}). The
        # A1-anchor dist / action / FB-Q are already computed above — pass them
        # so the seam doesn't recompute the actor+F forwards for anchor A1.
        extra_loss, extra_logs = self._actor_extra_loss(
            obs, z, dist=dist, sampled_action=sampled_action,
        )
        actor_loss = actor_loss + extra_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        # DDP handled reduce inside backward.
        handle = None

        with torch.no_grad():
            out = {
                "actor_loss": actor_loss.detach(),
                "Q_discriminator": Q_discriminator.mean(),
                "Q_aux": Q_aux.mean(),
                "Q_fb": Q_fb.mean(),
            }
            out.update(act_stats)
            out.update(extra_logs)
            if hasattr(self, "_soft_fb_actor_logs"):
                out.update(self._soft_fb_actor_logs)
                del self._soft_fb_actor_logs
        return out, handle

    def _actor_extra_loss(self, obs, z, dist=None, sampled_action=None):
        """Extra actor-side loss seam (no-op base). Returns ``(loss, logs)``.

        ``dist`` / ``sampled_action`` are the A1-anchor actor distribution and
        its sampled action already computed in ``backward_actor`` — passed so
        subclasses can avoid recomputing the actor forward / re-sampling for
        anchor A1.
        """
        return torch.zeros((), device=self.device), {}

    def step_actor(
        self, handle: Any, clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        p = self.policy
        finish_async_reduce(handle)
        max_norm = float(clip_grad_norm) if clip_grad_norm is not None else float("inf")
        gn = torch.nn.utils.clip_grad_norm_(p._actor.parameters(), max_norm)
        self.actor_optimizer.step()
        return {"grad_norm/actor": gn.detach()}

    # --- helper: pessimistic value (BFM's get_targets_uncertainty) --------- #

    @staticmethod
    def _pessimistic_value(
        preds: torch.Tensor, pessimism_penalty: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(mean, uncertainty, mean - penalty * uncertainty)``.

        Mirrors BFM-Zero's ``FBAgent.get_targets_uncertainty``. Assumes the
        leading dim is the ensemble (``num_parallel``) axis.
        """
        dim = 0
        preds_mean = preds.mean(dim=dim)
        # Pairwise absolute differences across the ensemble.
        preds_uns = preds.unsqueeze(dim=dim)
        preds_uns2 = preds.unsqueeze(dim=dim + 1)
        diffs = torch.abs(preds_uns - preds_uns2)
        n = preds.shape[dim]
        scaling = n * n - n
        preds_unc = diffs.sum(dim=(dim, dim + 1)) / max(scaling, 1)
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    # --- helper: quantile regression Huber loss ---------------------------- #

    @staticmethod
    def _quantile_huber_loss(
        pred_quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        """Quantile-regression Huber loss (Dabney et al. 2018).

        Args:
            pred_quantiles: ``[num_parallel, batch, n_q]`` — per-ensemble
                quantile outputs from the online critic.
            target_quantiles: ``[batch, n_q]`` — target quantile values
                (shared across ensemble; each head fits the same
                distribution). Targets are NOT sorted; TD-style updates
                preserve the quantile index because τ_i is a fixed grid.
            kappa: Huber switch-point.

        Returns:
            Scalar loss (mean over batch, sum over predicted-quantile
            axis, averaged across target-quantile axis and ensemble
            members — matches the widely-used QR-DQN normalization).
        """
        n_q = pred_quantiles.shape[-1]
        device = pred_quantiles.device
        dtype = pred_quantiles.dtype
        # τ_i = (i + 0.5) / n_q — midpoint quantile fractions.
        tau = (torch.arange(n_q, device=device, dtype=dtype) + 0.5) / n_q
        tau = tau.view(1, 1, 1, n_q)                                 # [1, 1, 1, n_q_pred]

        # u[..., i, j] = target[j] - pred[i]
        pred = pred_quantiles.unsqueeze(-1)                           # [np, B, n_q_pred, 1]
        tgt = target_quantiles.unsqueeze(0).unsqueeze(-2)             # [1,  B, 1,         n_q_tgt]
        u = tgt - pred                                                # [np, B, n_q_pred, n_q_tgt]
        abs_u = u.abs()
        huber = torch.where(
            abs_u <= kappa,
            0.5 * u.pow(2),
            kappa * (abs_u - 0.5 * kappa),
        )
        # Quantile weight: |τ - 1{u < 0}|.
        w = (tau - (u < 0).to(dtype)).abs()
        loss = (w * huber / max(kappa, 1e-8)).mean(dim=-1).sum(dim=-1)   # sum over n_q_pred
        return loss.mean()                                            # mean over ensemble & batch

    # --- (de)serialization ------------------------------------------------- #

    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizers": self.optimizer_dict,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        policy_sd = state["policy"]
        # Drop keys for modules that no longer exist (e.g. removed
        # reconstruction head), or whose shape no longer matches (e.g.
        # MA changed from D(s,s') to D(s) -- input dim halved).
        model_sd = self.policy.state_dict()
        model_keys = set(model_sd.keys())
        unexpected = [k for k in policy_sd if k not in model_keys]
        shape_mismatch = [
            k for k in policy_sd
            if k in model_keys and policy_sd[k].shape != model_sd[k].shape
        ]
        drop = set(unexpected) | set(shape_mismatch)
        if drop:
            print(f"[FBCprAux] load_state_dict: dropping {len(drop)} keys "
                  f"({len(unexpected)} unexpected, {len(shape_mismatch)} shape mismatch). "
                  f"e.g. {next(iter(drop))}")
            policy_sd = {k: v for k, v in policy_sd.items() if k not in drop}
        self.policy.load_state_dict(policy_sd, strict=False)

        # If we dropped MA weights, also drop the MA optimizer state so
        # it doesn't reference stale params.
        ma_dropped = any("manifold_attractor" in k for k in drop)
        optim = state.get("optimizers", {})
        for name, sd in optim.items():
            if ma_dropped and name == "manifold_attractor_optimizer":
                print(f"[FBCprAux] skipping {name} load (MA reinitialized)")
                continue
            opt = getattr(self, name, None)
            if opt is not None:
                try:
                    opt.load_state_dict(sd)
                except (ValueError, RuntimeError) as e:
                    print(f"[FBCprAux] skipping optimizer '{name}' load: {e}")


##########################
# FBCprCond — variant with an extra "measure_cond" exteroceptive obs key
# fed into F / actor / aux_critic. All training logic is inherited from
# :class:`FBCprAux` — the algorithm is obs-key agnostic.
##########################


@configclass
class FBCprCondAlgorithmCfg(FBCprAuxAlgorithmCfg):
    """Training hyperparameters for :class:`FBCprCond`. Inherits every
    knob from :class:`FBCprAuxAlgorithmCfg`; only the ``class_name`` is
    flipped so the runner factory picks the :class:`FBCprCond` class."""

    class_name: str = "FBCprCond"


class FBCprCond(FBCprAux):
    """FBCprAux trained with an extra exteroceptive ``measure_cond`` obs.

    All training logic — obs batching, target updates, optimizer steps —
    is inherited unchanged. The policy module (``FBCprCondPolicy``)
    handles the new obs key through its input filters and normalizer;
    this class exists purely so the runner's ``class_name`` dispatch can
    pick it up.
    """

    pass
