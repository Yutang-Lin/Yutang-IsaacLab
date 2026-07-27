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

from ..fb_cpr_math import (
    advance_tracking_phases,
    aux_q_for_actor,
    aux_reward_for_critic,
    centered_subwindow_start,
    completed_tracking_bins,
    ema_grad_spike_state,
    innovation_alignment_loss,
    normalized_forward_value,
    normalized_gamma_loss_weights,
    sample_log_horizon_gamma,
    sample_relabel_z,
    stochastic_integral_weights,
    tracking_failure_metrics,
)
from ..modules.fb_cpr_policy import (
    FBCprAuxPolicy,
    FBCprNetworkCfg,
    TransformerActorWrapper,
    _soft_update_params,
    eval_mode,
    gamma_forward_output_to_raw,
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


def _grad_norm_without_clipping(
    parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    """Return the total L2 gradient norm without modifying any gradient."""
    grads = [
        param.grad.detach()
        for param in parameters
        if param.grad is not None
    ]
    if grads:
        return torch.nn.utils.get_total_norm(
            grads, norm_type=2.0, foreach=None
        )
    device = parameters[0].device if parameters else torch.device("cpu")
    return torch.zeros((), device=device)


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
    anchor_pose_clamp: float = 10.0          # ±metres clamp / signed-log full-scale R
    anchor_alpha_gt: float = 0.34            # p(anchor = g_t)
    # Anchor-frame body pose: reframe privileged_state body POS/ROT6D into the
    # per-row anchor A_i (train+expert) so B/F/critic/disc see globally-
    # positioned body. MUST be a declared field — _build_algo_cfg only copies
    # cfg keys for which hasattr(cfg, k) is True, so an undeclared flag is
    # silently dropped and the feature never activates.
    anchor_frame_body: bool = False
    # Append a heading-frame body (pos+rot6d) tail to privileged_state so B/F/
    # critic/disc see anchor-INVARIANT local shape alongside the anchor-framed
    # (global-position) leading block. Drives both the env obs term and the
    # expert buffer compose; layout becomes 24K-5. Declared here so the runner's
    # hasattr-filter forwards it to the buffer.
    priv_include_heading_body: bool = False
    # No-anchor control: pin all anchors at the world origin (no per-row relabel,
    # no equivariance augmentation). anchored_pose == true world pose; priv body
    # reframe to origin is a no-op. Requires the env to spawn all robots at
    # origin/+x (shared world frame). For isolating whether plain world-frame
    # tracking works without the anchoring machinery.
    anchor_disable: bool = False
    # Reset EVERY resampled tracking env (not just terrain-tied) to the new
    # motion's RSI frame on a mid-episode tracking resample. Needed under
    # origin-spawn / no-anchor: the robot drifts from origin within a window but
    # the new origin-anchored tracking-z assumes it's at the motion's frame-0,
    # so without a reset the z goal frame and the robot pose jump-mismatch.
    reset_tracking_on_resample: bool = False
    anchor_beta_gh: float = 0.33             # p(anchor = g_h); rest -> random
    anchor_random_xy_range: float = 10.0     # random anchor xy ± around g_t
    # Two-frame rollout anchor: per tracking window, sample ONE offset A_anchor
    # (init-local) shared by the env ``anchored_pose`` obs (A_init·A_anchor, sim)
    # and the tracking-z encode (A^m_init·A_anchor, motion). prob anchor_alpha_gt
    # -> A_anchor=0 (spawn-anchored); else uniform ±this xy / ±π yaw. Displaces
    # the spatial goal so the actor practices reaching far targets (fills the
    # displaced-goal coverage hole). Defaults to anchor_random_xy_range if unset.
    rollout_anchor_xy_range: float = 0.0
    # Goal-z displaced practice (update-time): for this fraction of goal-z rows,
    # replace the (travel-bounded) goal displacement with a FRESH wide random
    # draw (uniform ±goal_z_displace_xy_range xy, ±π yaw), applied coherently to
    # BOTH anchored_pose and the priv body. Decouples far-reaching actor practice
    # from current rollout coverage (fixes the self-reinforcing drift trap). 0 =
    # off (goal displacement = real cross-row travel only).
    goal_z_displace_prob: float = 0.0
    goal_z_displace_xy_range: float = 0.0    # defaults to rollout_anchor_xy_range
    # Tracking-rollout knobs — MUST be declared here (FBCprAuxAlgorithmCfg is the
    # anchored runner's _ALGO_CFG_CLS). Read via getattr(self.cfg, ...) in
    # _resample_tracking, so if undeclared the runner's hasattr-filter drops the
    # cfg's set value and the DEFAULT silently applies. (This bit
    # global_fb_zero_prob=1.0 -> ran at 0.5, and terrain_variant_root_h_prob.)
    global_fb_zero_prob: float = 0.5         # p(tracking env's global-FB obs zeroed)
    terrain_variant_root_h_prob: float = 0.0  # p(use absolute root_h variant); 0 for BFM-One
    # Keypoint list — declared so it survives the runner's hasattr-filter into
    # self.cfg (object-read by _priv_K's fallback). Primary K derivation is from
    # the priv dim, but keep this consistent to avoid the silent-drop footgun.
    expert_keypoint_names: list | None = None
    # Per-env exploration-std gradient (BEHAVIOR policy only; TD target uses
    # actor_std). Each env draws a fresh std in [min, max] per episode for
    # broader (s,a) coverage. Disabled when max<=min (falls back to actor_std).
    explore_std_min: float = 0.0
    explore_std_max: float = 0.0
    # (anchor KL/Q consistency penalties removed — invariance comes from FB TD
    # over anchor-relabeled transitions, not an explicit loss.)
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

    # Startup LR scaling control. The batch contribution is
    # sqrt(batch_size / 1024).
    lr_scale_with_world_size: bool = True
    lr_scale_with_batch_size: bool = True
    obs_normalizer_scale_momentum: bool = True
    obs_normalizer_global_moments: bool = True

    # Rollout/replay sampling controls. The replacement flag is independent of
    # the legacy episode-phase clock so standard coherent 250-step tracking can
    # use UFO's cohort-selection law without restoring other legacy semantics.
    rollout_tracking_legacy_schedule: bool = False
    rollout_tracking_with_replacement: bool = False
    expert_tracking_circular_wrap: bool = False
    replay_sampling_mode: str = "uniform_transition"
    replay_mark_eval_boundary: bool = True
    # Opt-in tracking curriculum. A fixed fraction of the unique physical
    # tracking envs terminates after sustained mean joint-position error,
    # resets onto the failed reference frame, and contributes a binary outcome
    # to a per-motion failure EMA used only for future tracking assignments.
    tracking_early_termination_fraction: float = 0.0
    tracking_failure_joint_mae_threshold_rad: float = 0.20
    tracking_failure_root_height_threshold_m: float = 0.25
    tracking_failure_grace_steps: int = 10
    tracking_failure_consecutive_steps: int = 5
    tracking_failure_priority_ema_decay: float = 0.90
    tracking_failure_priority_scale: float = 3.0
    tracking_failure_priority_max_multiplier: float = 4.0
    tracking_failure_bin_size_s: float = 0.0

    # Optional world-size scaling for the FB/critic target-network Polyak rates.
    target_tau_scale_with_world_size: bool = False
    target_tau_world_size_cap: int = 0

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
    # Optional finite-spike guard for the coupled F/B update. Each branch keeps
    # its own pre-clip gradient-norm EMA. Once warm, a norm above
    # ``multiplier * EMA`` is clipped back to the EMA instead of skipping the
    # optimizer step. Non-finite gradients are still skipped.
    fb_grad_spike_clip: bool = False
    fb_grad_spike_ema_decay: float = 0.99
    fb_grad_spike_multiplier: float = 5.0
    fb_grad_spike_warmup_steps: int = 128
    # Expensive tail percentiles are diagnostics only. Maxima and gamma-at-worst
    # are still computed every update.
    fb_tail_quantile_every: int = 1

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
    expert_shard_seed: int = 42

    # Discriminator
    grad_penalty_discriminator: float = 10.0

    # Reg-coeffs in the actor objective
    reg_coeff: float = 0.05       # weight on Q_discriminator inside actor loss
    reg_coeff_aux: float = 0.02   # weight on Q_aux_critic inside actor loss
    aux_actor_denormalize_q: bool = False
    """Restore actor Q_aux using the fixed reward scale when configured,
    otherwise using the detached auxiliary-reward EMA sigma."""
    aux_reward_fixed_scale: float = 0.0
    """If positive, train the aux critic on raw reward divided by this fixed
    scale. With aux_actor_denormalize_q, the actor multiplies Q_aux by it."""
    aux_reward_sigma_min: float = 3.0
    """Minimum adaptive aux-reward sigma for denormalized-Q configurations.

    This is active when ``aux_actor_denormalize_q`` is true and no fixed scale
    is configured. Below the floor, critic rewards are divided by this value
    instead of the live EMA sigma. Actor Q_aux remains normalized with
    multiplier one both below and above the floor.
    """
    scale_reg: bool = True         # multiply regs by |Q_fb|.abs().mean().detach()
    # Actor-only FB scale alignment. Applied before scale_reg so the direct FB
    # and regularizer terms scale together; FB TD targets remain unchanged.
    actor_fb_scale: float = 1.0

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
    # --- gamma-conditioned F (requires policy.forward_gamma_embed_dim > 0) ---
    # When on, each FB TD update samples a per-row gamma from a range so the one
    # F(s,z,a,gamma) fits successor measures across horizons, and the actor's FB
    # term becomes Q_fb(gamma_L) + (1-gamma_S)/(1-gamma_L)*alpha*Q_fb(gamma_S).
    fb_gamma_conditioned: bool = False
    # Short horizon used by the actor's second FB term (long horizon = discount).
    actor_gamma_short: float = 0.8
    # Weight alpha on the short-horizon FB term in the actor loss.
    actor_gamma_short_alpha: float = 0.5
    # Weight each gamma-conditioned Bellman FB row by (1-gamma)^power,
    # normalized under the uniform log-horizon sampling distribution.
    fb_gamma_loss_weighting: bool = False
    fb_gamma_loss_weight_power: float = 2.0
    # Stochastic-integral FB actor objective (overrides the two-gamma term when on;
    # requires fb_gamma_conditioned). Stratified-sample fb_integral_K horizons in
    # [gamma_short, discount], softmax-weight the normalized per-step values, and
    # integrate. See backward_actor.
    fb_stochastic_integral: bool = False
    fb_integral_K: int = 8
    # Align scale on the integrated Q: 1/(1-fb_integral_align_gamma) (default
    # gamma=0.98 -> 50) so the per-step integral sits at the standard-gamma
    # Q magnitude.
    fb_integral_align_gamma: float = 0.98
    # Adaptive softmax temperature for stochastic-integral weights:
    # sqrt(mean(abs(N - N_max))) with a floor of 1.0. Growing Q gaps therefore
    # sharpen horizon selection gradually rather than linearly.
    fb_integral_adaptive_tau: bool = False
    # Exponential prior over the sampled log-horizon h=-log(1-gamma):
    # p0(h) ∝ exp(-lambda * (h - h_min)). Zero preserves the old SI weights.
    fb_integral_prior_lambda: float = 0.0
    # Explicitly match Bellman innovations at two independently sampled gammas:
    # [F_gamma B^T - gamma F'_gamma B_target^T]. This regularizes one
    # gamma-conditioned F to represent the same immediate transition measure
    # across horizons.
    fb_gamma_innovation_align: bool = False
    fb_gamma_innovation_align_coef: float = 1.0
    relabel_ratio: float | None = 0.8
    # Optional actor-specific relabel probability. None preserves the original
    # behavior where actor and value updates consume the same relabeled z.
    actor_relabel_ratio: float | None = None
    train_goal_ratio: float = 0.2
    expert_asm_ratio: float = 0.6

    # Rollout-context sampling
    update_z_every_step: int = 100
    use_mix_rollout: bool = True
    rollout_expert_trajectories: bool = True
    rollout_expert_trajectories_length: int = 250
    rollout_expert_trajectories_percentage: float = 0.5
    z_buffer_size: int = 8192
    # --- Analytic z-bar from the linear-W feature decoder (BFM-0.5) -------
    # When >0, a fraction of tracking-rollout z's (and of relabel z's) are built
    # analytically as z_bar_g = W^T c_g from the learned no-bias linear recon
    # head (requires recon_linear + recon_square_augment), instead of B-encoding
    # the expert window. c_g = [2*Lambda*g, -diag(Lambda)] for a diagonal-Lambda
    # tracking reward, g = the W-target features of the goal frame. 0 = off.
    zbar_tracking_ratio: float = 0.0   # frac of tracking envs using z_bar
    zbar_relabel_ratio: float = 0.0    # frac of relabel z drawn as z_bar
    # Per-feature diagonal Lambda (tracking-reward weights). Empty = all-ones
    # over the W head's base feature dim.
    zbar_feature_weights: tuple[float, ...] = ()
    # Transformer actor: H = number of PAST frames in the parallel actor window
    # (window = H+1 incl. current). 0 = MLP actor (no window gathered). Normally
    # left 0 and auto-derived in FBCprAux.__init__ from the policy's
    # actor_history_len when actor_arch=="transformer".
    actor_window_len: int = 0
    # ISOLATION TEST (transformer actor): score the parallel actor loss at the
    # CURRENT token only, while still running the full H+1 transformer forward.
    # Isolates whether the Q_disc/Q_aux runaway comes from the past-token scoring
    # (parallel loss) vs the transformer/frame_norm/window. Default False.
    actor_score_current_only: bool = False
    # Log the Track/global_xy_dev_m + global_yaw_dev_deg deviation metrics during
    # rollout (the robot-vs-reference global-path error). Meaningful only for
    # global-FB / tracking tasks (BFM-Terrain/One); pointless for BFM-Zero/0.5
    # (no global goal). Default True (existing tasks unchanged); set False to skip.
    log_global_track_dev: bool = True
    tracking_T_min: int = 1
    tracking_T_max: int = 16
    # If non-empty, sample T from this discrete set instead of uniformly from
    # [T_min, T_max]. This controls per-env rollout tracking z and is inherited
    # by expert/relabel z unless expert_T_* or disc_fixed_T overrides it.
    # Rollout T is sampled once per tracking episode; expert T once per sequence.
    tracking_T_choices: tuple[int, ...] = ()
    # Per-choice probabilities (must match len(tracking_T_choices) when set).
    # Empty tuple = uniform over choices.
    tracking_T_choice_probs: tuple[float, ...] = ()
    # Optional expert/relabel mean-z horizon sampling. Zeros/empty choices
    # inherit tracking_T_* for backward compatibility. These settings affect
    # the expert z shared by relabeling and discriminator conditioning, not the
    # independently configurable discriminator-positive observation window.
    expert_T_min: int = 0
    expert_T_max: int = 0
    expert_T_choices: tuple[int, ...] = ()
    expert_T_choice_probs: tuple[float, ...] = ()
    # Fixed expert/discriminator z-mean horizon. Positive values decouple the
    # expert z from all sampled ranges. Zero uses expert_T_* when configured,
    # otherwise retaining the legacy tracking_T_* coupling.
    disc_fixed_T: int = 0
    # If True, the discriminator's positive (expert) window is ALWAYS the full
    # seq_length regardless of the per-sequence z-window T — i.e. every frame
    # in the sub-sequence is a valid positive. The z is still computed from the
    # per-T window; only the disc_mask is forced to all-True. If False (default),
    # the positive window matches T (frames 0..T-1 only).
    disc_positive_full_window: bool = False
    # Centered discriminator-positive window around the midpoint of the
    # expert z-mean horizon. Zero preserves the legacy first-T/full-window
    # behavior above. Positive values decouple the style-positive width from T;
    # the sampled expert sequence must be at least this long.
    disc_positive_window: int = 0
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
    # Compile the online forward map when compile_mode is enabled. Large
    # stochastic-integral actor batches may disable this selectively while
    # retaining compilation for B/actor/critics.
    compile_forward_map: bool = True

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
        self._fb_grad_norm_ema_f = 0.0
        self._fb_grad_norm_ema_b = 0.0
        self._fb_grad_norm_ema_steps = 0
        self._fb_tail_diagnostic_step = 0
        replay_sampling_mode = str(
            getattr(cfg, "replay_sampling_mode", "uniform_transition")
        )
        if replay_sampling_mode not in ("uniform_transition", "uniform_trajectory"):
            raise ValueError(
                "replay_sampling_mode must be 'uniform_transition' or "
                f"'uniform_trajectory', got {replay_sampling_mode!r}"
            )
        if float(getattr(cfg, "aux_reward_fixed_scale", 0.0)) < 0.0:
            raise ValueError("aux_reward_fixed_scale must be non-negative")
        if float(cfg.aux_reward_sigma_min) < 0.0:
            raise ValueError("aux_reward_sigma_min must be non-negative")
        for name in ("relabel_ratio", "actor_relabel_ratio"):
            ratio = getattr(cfg, name, None)
            if ratio is not None and not 0.0 <= float(ratio) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1] or None, got {ratio}")
        if bool(getattr(cfg, "fb_grad_spike_clip", False)):
            decay = float(getattr(cfg, "fb_grad_spike_ema_decay", 0.99))
            multiplier = float(getattr(cfg, "fb_grad_spike_multiplier", 5.0))
            warmup = int(getattr(cfg, "fb_grad_spike_warmup_steps", 128))
            if not 0.0 <= decay < 1.0:
                raise ValueError(
                    f"fb_grad_spike_ema_decay must be in [0, 1), got {decay}"
                )
            if multiplier <= 1.0:
                raise ValueError(
                    f"fb_grad_spike_multiplier must be > 1, got {multiplier}"
                )
            if warmup < 0:
                raise ValueError(
                    f"fb_grad_spike_warmup_steps must be non-negative, got {warmup}"
                )
        if int(getattr(cfg, "fb_tail_quantile_every", 1)) <= 0:
            raise ValueError("fb_tail_quantile_every must be positive")

        # Disc batch is sized as disc_num_slices * seq_length (must be a
        # multiple of seq_length for [num_slices, seq_length] reshape).
        # Main cfg.batch_size is independent and stays exact (e.g. 1024).
        seq_length = int(self.policy.seq_length)
        positive_window = int(getattr(cfg, "disc_positive_window", 0))
        if positive_window < 0 or positive_window > seq_length:
            raise ValueError(
                f"disc_positive_window={positive_window} must be in "
                f"[0, seq_length={seq_length}]"
            )
        self._disc_positive_window = positive_window
        disc_num_slices = getattr(cfg, "disc_num_slices", None)
        if disc_num_slices is not None:
            self._disc_batch_size = int(disc_num_slices) * seq_length
        else:
            self._disc_batch_size = max(seq_length, (cfg.batch_size // seq_length) * seq_length)
        self._disc_num_sequences = self._disc_batch_size // seq_length

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

        # LR scaling. Two independently configurable sqrt terms stack:
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
        # Target-network Polyak rates are scaled independently below.
        import math
        REF_BATCH_SIZE = 1024
        ws = (int(torch.distributed.get_world_size())
              if self.is_distributed else 1)
        # LR (and coupled normalizer-EMA) scaling defaults to
        # sqrt(world_size) * sqrt(batch_size / 1024).
        # ``lr_scale_with_world_size`` and ``lr_scale_with_batch_size`` disable
        # either factor independently. This lets parity experiments use an exact
        # base LR even under DDP and with a non-reference local batch.
        if bool(getattr(cfg, "lr_scale_with_batch_size", True)):
            bs_mult = math.sqrt(max(int(cfg.batch_size), 1) / REF_BATCH_SIZE)
        else:
            bs_mult = 1.0
        ws_mult = (
            math.sqrt(max(ws, 1))
            if bool(getattr(cfg, "lr_scale_with_world_size", True))
            else 1.0
        )
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

        # EMA normalizer time-constant scaling. Observation scaling can be
        # disabled independently once DDP uses globally pooled batch moments.
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
            # BatchNorm per-key momentum on _obs_normalizer._normalizers[<k>]._normalizer
            new_obs_moms: Dict[str, float] = {}
            if (
                bool(getattr(cfg, "obs_normalizer_scale_momentum", True))
                and hasattr(self.policy, "_obs_normalizer")
                and hasattr(
                    self.policy._obs_normalizer, "_normalizers"
                )
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

        # Target-network Polyak rates track world-size scaling separately from
        # LR and normalizer EMA scaling. They intentionally do not include the
        # per-rank batch multiplier.
        if bool(getattr(cfg, "target_tau_scale_with_world_size", False)):
            target_tau_ws_cap = int(getattr(cfg, "target_tau_world_size_cap", 0))
            target_tau_ws = (
                min(ws, target_tau_ws_cap)
                if target_tau_ws_cap > 0
                else ws
            )
            target_tau_mult = math.sqrt(max(target_tau_ws, 1))
            cfg.fb_target_tau = min(1.0, float(cfg.fb_target_tau) * target_tau_mult)
            cfg.critic_target_tau = min(
                1.0, float(cfg.critic_target_tau) * target_tau_mult
            )
            print(
                f"[FBCprAux] target tau scaling: world_size={ws}"
                f"->{target_tau_ws} (×{target_tau_mult:.3f})  "
                f"fb={cfg.fb_target_tau:.4g} critic={cfg.critic_target_tau:.4g}",
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
            # entropy_critic / manifold_attractor included so their LRs are also
            # re-applied (with current-cfg DDP scaling) on resume — Adam.load_
            # state_dict would otherwise clobber them with the stale saved LR.
            "entropy_critic": float(cfg.lr_entropy_critic),
            "manifold_attractor": float(cfg.lr_manifold_attractor),
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
            compile_forward_map = bool(getattr(cfg, "compile_forward_map", True))
            if compile_forward_map:
                self.policy._forward_map = torch.compile(self.policy._forward_map, **compile_kwargs)
            self.policy._backward_map = torch.compile(self.policy._backward_map, **compile_kwargs)
            self.policy._actor = torch.compile(self.policy._actor, **compile_kwargs)
            self.policy._critic = torch.compile(self.policy._critic, **compile_kwargs)
            self.policy._aux_critic = torch.compile(self.policy._aux_critic, **compile_kwargs)
            if self.policy._entropy_critic is not None:
                self.policy._entropy_critic = torch.compile(self.policy._entropy_critic, **compile_kwargs)
            # Transformer actor: the outer compile above only wraps __call__/
            # forward (the ROLLOUT path). The TRAINING actor loss calls the
            # custom method ``forward_window`` via ``_unwrap`` (which peels the
            # compile + DDP wrappers), so that path ran EAGER — the dominant cost
            # of a wide transformer actor (eager attention at d_model 2048, x16
            # updates/step -> seconds/iter). Compile ``forward_window`` as a
            # separate bound-method handle stored on the bare wrapper; the
            # training loop uses it when present. This adds an attribute (a
            # compiled callable), NOT a submodule, so the state_dict / checkpoint
            # key layout is unchanged.
            _bare_actor = self._unwrap(self.policy._actor)
            if isinstance(_bare_actor, TransformerActorWrapper):
                _bare_actor._compiled_forward_window = torch.compile(
                    _bare_actor.forward_window, **compile_kwargs)
            # Disc uses autograd.grad(create_graph=True) for WGAN-GP, which
            # is known to hit graph breaks with torch.compile — leave it
            # eager. Target networks never need compile (no backward).
            compiled_nets = (
                "F/B/actor/critic/aux_critic"
                if compile_forward_map
                else "B/actor/critic/aux_critic (F eager)"
            )
            print(f"[FBCprAux] torch.compile mode={compile_mode} applied to "
                  f"{compiled_nets} (disc stays eager)"
                  + ("; actor.forward_window compiled (training path)"
                     if isinstance(_bare_actor, TransformerActorWrapper) else ""),
                  flush=True)

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

    @property
    def training_state_dict(self) -> Dict[str, Any]:
        """Small non-model state needed for an exact training resume."""
        return {
            "fb_grad_norm_ema_f": self._fb_grad_norm_ema_f,
            "fb_grad_norm_ema_b": self._fb_grad_norm_ema_b,
            "fb_grad_norm_ema_steps": self._fb_grad_norm_ema_steps,
        }

    def load_training_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore optional training-only state from a checkpoint."""
        if not state:
            return
        self._fb_grad_norm_ema_f = max(
            float(state.get("fb_grad_norm_ema_f", 0.0)), 0.0
        )
        self._fb_grad_norm_ema_b = max(
            float(state.get("fb_grad_norm_ema_b", 0.0)), 0.0
        )
        self._fb_grad_norm_ema_steps = max(
            int(state.get("fb_grad_norm_ema_steps", 0)), 0
        )

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
        if n >= cap:
            # Sampling from this buffer is unordered, so retaining the newest
            # ``cap`` rows contiguously is equivalent and avoids duplicate CUDA
            # advanced-index writes when n > cap.
            buf.copy_(z[-cap:].detach().to(buf.dtype))
            self._z_buffer_cursor = 0
            self._z_buffer_size = cap
            return
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
        # Anchoring seam (no-op base): re-anchor the shuffled goal's spatial
        # channel to the DESTINATION row's anchor A_i so z[i] and obs_i share a
        # frame (matching deployment). Base leaves ``shuffled`` unchanged.
        shuffled = self._reanchor_goal_z(shuffled, perm)
        goals = self.policy._backward_map(shuffled)
        goals = self.policy.project_z(goals)
        z = torch.where(mix_idxs == 0, goals, z)

        # Expert-encoded z's. Sample with replacement so expert pool size
        # can differ from main batch size.
        n_expert = expert_encodings.shape[0]
        idx = torch.randint(0, n_expert, (batch,), device=self.device)
        # Anchoring seam (no-op base): re-encode the picked expert window under
        # A_i (rebuild its anchored_pose), so the expert-seeded z is in row i's
        # frame too. Base just gathers the precomputed expert_encodings[idx].
        expert_z = self._reanchor_expert_z(expert_encodings, idx)
        z = torch.where(mix_idxs == 1, expert_z, z)

        # Analytic z_bar relabel: for a proportion of rows, overwrite z with the
        # W-decoder task embedding built from a SHUFFLED goal (perm, so each row
        # gets a different goal — same hindsight convention as the goal-z path).
        zbar_ratio = float(getattr(self.cfg, "zbar_relabel_ratio", 0.0))
        zbar = getattr(self, "_relabel_zbar", None)
        if zbar_ratio > 0.0 and zbar is not None:
            zbar_shuf = zbar[perm]
            use_zbar = (torch.rand(batch, 1, device=self.device) < zbar_ratio)
            z = torch.where(use_zbar, zbar_shuf, z)
        return z

    def _reanchor_goal_z(self, shuffled, perm):
        """Goal-z re-anchor seam (no-op base). Returns ``shuffled`` unchanged so
        the base goal-encoded z = B(shuffled next_obs) is byte-identical."""
        return shuffled

    def _reanchor_expert_z(self, expert_encodings, idx):
        """Expert-z re-anchor seam (no-op base). Returns the precomputed
        ``expert_encodings[idx]`` (vanilla expert-encoded z)."""
        return expert_encodings[idx]

    def _sample_expert_T(
        self,
        num_sequences: int,
        seq_length: int,
    ) -> torch.Tensor | None:
        """Sample one expert z-mean horizon per sequence."""
        fixed_disc_T = int(getattr(self.cfg, "disc_fixed_T", 0))
        if fixed_disc_T > 0:
            if fixed_disc_T > seq_length:
                raise ValueError(
                    f"disc_fixed_T={fixed_disc_T} exceeds policy "
                    f"seq_length={seq_length}"
                )
            return torch.full(
                (num_sequences,),
                fixed_disc_T,
                device=self.device,
                dtype=torch.long,
            )

        expert_choices = tuple(
            getattr(self.cfg, "expert_T_choices", ()) or ()
        )
        expert_choice_probs = tuple(
            getattr(self.cfg, "expert_T_choice_probs", ()) or ()
        )
        expert_T_min = int(getattr(self.cfg, "expert_T_min", 0))
        expert_T_max = int(getattr(self.cfg, "expert_T_max", 0))
        if expert_choices or expert_T_min > 0 or expert_T_max > 0:
            choices = expert_choices
            choice_probs = expert_choice_probs
            if expert_choices and expert_T_min == 0 and expert_T_max == 0:
                T_min = min(expert_choices)
                T_max_cfg = max(expert_choices)
            else:
                T_min = expert_T_min
                T_max_cfg = expert_T_max
            if T_min <= 0 or T_max_cfg <= 0:
                raise ValueError(
                    "expert_T_min and expert_T_max must both be positive "
                    "when either expert range endpoint is configured"
                )
        else:
            choices = tuple(
                getattr(self.cfg, "tracking_T_choices", ()) or ()
            )
            choice_probs = tuple(
                getattr(self.cfg, "tracking_T_choice_probs", ()) or ()
            )
            T_min = int(getattr(self.cfg, "tracking_T_min", 1))
            T_max_cfg = int(getattr(self.cfg, "tracking_T_max", 16))
        T_max = min(T_max_cfg, seq_length)
        if T_min < 1 or T_min > T_max:
            raise ValueError(
                f"Invalid expert mean-z horizon range [{T_min}, {T_max_cfg}] "
                f"for seq_length={seq_length}"
            )

        if choices:
            if choice_probs and len(choice_probs) != len(choices):
                raise ValueError(
                    "expert/tracking T choice probabilities must match the "
                    f"number of choices: {len(choice_probs)} != {len(choices)}"
                )
            kept = [
                (choice, choice_probs[i] if choice_probs else 1.0)
                for i, choice in enumerate(choices)
                if T_min <= choice <= T_max
            ]
            if not kept:
                raise ValueError(
                    f"No expert mean-z T choices fit seq_length={seq_length}: "
                    f"{choices}"
                )
            choices_t = torch.tensor(
                [choice for choice, _ in kept],
                device=self.device,
                dtype=torch.long,
            )
            if choice_probs:
                weights = torch.tensor(
                    [prob for _, prob in kept],
                    device=self.device,
                    dtype=torch.float32,
                )
                selection = torch.multinomial(
                    weights, num_sequences, replacement=True
                )
            else:
                selection = torch.randint(
                    0,
                    len(kept),
                    (num_sequences,),
                    device=self.device,
                )
            return choices_t[selection]
        if T_min < T_max:
            return torch.randint(
                T_min,
                T_max + 1,
                (num_sequences,),
                device=self.device,
            )
        if self._disc_positive_window > 0:
            return torch.full(
                (num_sequences,), T_min, device=self.device, dtype=torch.long
            )
        return None

    @staticmethod
    def _permute_obs(
        obs: torch.Tensor | dict[str, torch.Tensor], perm: torch.Tensor
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(obs, dict):
            return {k: v[perm] for k, v in obs.items()}
        return obs[perm]

    @torch.no_grad()
    def encode_expert(
        self,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        T_per_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode expert sub-sequences through B for discriminator training.

        ``disc_fixed_T > 0`` uses that fixed mean horizon. Otherwise expert_T_*
        controls the mean when configured; zero/empty expert settings retain
        the legacy behavior coupled to ``tracking_T_*``.

        Returns:
            z_expert: [batch_size, z_dim] z replicated per frame
            disc_mask: [batch_size] bool, True for frames within the configured
                positive window. None when all sequence frames are positive.
        """
        B_expert = self.policy._backward_map(next_obs).detach()
        seq_length = self.policy.seq_length
        # Use the actual batch returned (may be _disc_batch_size, not cfg.batch_size).
        N = B_expert.shape[0] // seq_length
        B_expert = B_expert.view(N, seq_length, B_expert.shape[-1])
        device = B_expert.device

        positive_window = self._disc_positive_window
        disc_mask: torch.Tensor | None = None
        if T_per_seq is None:
            T_per_seq = self._sample_expert_T(N, seq_length)
        if T_per_seq is not None:
            d = B_expert.shape[-1]
            cumz = torch.cat([torch.zeros(N, 1, d, device=device),
                              torch.cumsum(B_expert, dim=1)], dim=1)  # [N, seq+1, d]
            arange_N = torch.arange(N, device=device)
            # Expert relabel z remains byte-for-byte the first T encoded frames.
            z_sum = cumz[arange_N, T_per_seq]  # [N, d]
            z_expert = z_sum / T_per_seq.float().unsqueeze(-1)
            if positive_window > 0:
                if positive_window == seq_length:
                    disc_mask = None
                else:
                    positive_start = centered_subwindow_start(
                        seq_length, positive_window
                    )
                    positive_end = positive_start + positive_window
                    arange_T = torch.arange(
                        seq_length, device=device
                    )
                    disc_mask = (
                        (arange_T >= positive_start)
                        & (arange_T < positive_end)
                    ).repeat(N)
            else:
                # Legacy: frames 0..T-1 are within the positive window.
                arange_T = torch.arange(
                    seq_length, device=device
                ).unsqueeze(0)
                disc_mask = (
                    arange_T < T_per_seq.unsqueeze(1)
                ).reshape(-1)
                # Optionally use the FULL seq_length as the discriminator
                # positive window while retaining the first-T z mean.
                if bool(getattr(self.cfg, "disc_positive_full_window", False)):
                    disc_mask = None
        else:
            z_expert = B_expert.mean(dim=1)
            if 0 < positive_window < seq_length:
                positive_start = (seq_length - positive_window + 1) // 2
                arange_T = torch.arange(seq_length, device=device)
                disc_mask = (
                    (arange_T >= positive_start)
                    & (arange_T < positive_start + positive_window)
                ).repeat(N)

        if self.cfg.soft_fb:
            norm = z_expert.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            z_expert = z_expert / (norm + 1.0)
        else:
            z_expert = self.policy.project_z(z_expert)

        z_expert = torch.repeat_interleave(z_expert, seq_length, dim=0)
        # Stash window structure for the anchored expert-z re-anchor seam
        # (no-op / unused in the base). ``_expert_next_obs_ref`` is the
        # already-normalized expert body obs; ``_expert_T_per_seq`` the per-
        # sub-sequence T-window; both [N_seq]-aligned with seq_length blocks.
        self._expert_next_obs_ref = next_obs
        self._expert_T_per_seq = T_per_seq
        self._expert_seq_length = seq_length
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
        tracking_fraction = float(self.cfg.rollout_expert_trajectories_percentage)
        if not 0.0 <= tracking_fraction <= 1.0:
            raise ValueError(
                "rollout_expert_trajectories_percentage must be in [0, 1], "
                f"got {tracking_fraction}"
            )
        tracking_enabled = bool(
            self.cfg.rollout_expert_trajectories
            and expert_buffer is not None
            and tracking_fraction > 0.0
        )

        if z is None:
            z = self.policy.sample_z(step_count.shape[0], device=self.device)
            if tracking_enabled:
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
        if tracking_enabled:
            traj_len = int(self.cfg.rollout_expert_trajectories_length)
            if bool(getattr(self.cfg, "rollout_tracking_legacy_schedule", False)):
                idxs = step_count % traj_len
                if bool((idxs == 0).any()):
                    terrain_envs = self._resample_tracking(
                        step_count, expert_buffer, robot_root_xy, robot_root_quat,
                        terrain_z_fn=terrain_z_fn,
                    )
                if getattr(self, "_tracking_env_idx", None) is not None:
                    mod_time = idxs[self._tracking_env_idx].view(-1)
                    mod_time = torch.clamp(
                        mod_time, 0, self._tracking_z.shape[1] - 1
                    )
                    self._tracking_local_phases.copy_(mod_time)
                    n = len(self._tracking_env_idx)
                    z[self._tracking_env_idx] = self._tracking_z[
                        torch.arange(n, device=self.device), mod_time,
                    ]
                return z, terrain_envs

            tracking_phase = int(getattr(self, "_tracking_phase", 0)) + 1
            resampled = False
            if tracking_phase >= traj_len:
                old_tracking_env_idx = getattr(self, "_tracking_env_idx", None)
                if old_tracking_env_idx is not None:
                    old_tracking_env_idx = old_tracking_env_idx.clone()
                terrain_envs = self._resample_tracking(
                    step_count, expert_buffer, robot_root_xy, robot_root_quat,
                    terrain_z_fn=terrain_z_fn,
                )
                # Envs leaving the tracking set must stop using their previous
                # trajectory z immediately. Overlapping/new tracking envs are
                # overwritten with their new tracking z below.
                if old_tracking_env_idx is not None:
                    z[old_tracking_env_idx] = new_z[old_tracking_env_idx]
                tracking_phase = 0
                resampled = True
            self._tracking_phase = tracking_phase
            if getattr(self, "_tracking_env_idx", None) is not None:
                if not resampled:
                    hold = self._tracking_phase_hold_once
                    self._tracking_local_phases = advance_tracking_phases(
                        self._tracking_local_phases,
                        hold,
                        self._tracking_z.shape[1] - 1,
                    )
                    hold.zero_()
                mod_time = self._tracking_local_phases
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
        tracking_fraction = float(self.cfg.rollout_expert_trajectories_percentage)
        n_elem = min(n_envs, max(1, int(tracking_fraction * n_envs)))
        with_replacement = bool(
            getattr(self.cfg, "rollout_tracking_with_replacement", False)
            or getattr(self.cfg, "rollout_tracking_legacy_schedule", False)
        )
        if with_replacement:
            sampled_env_idx = torch.randint(
                0, n_envs, (n_elem,), device=self.device
            )
            # One physical env can execute only one tracking context. CUDA
            # duplicate-index writes do not define which slot wins, so collapse
            # replacement draws before sampling motions/z and make ownership
            # deterministic.
            self._tracking_env_idx = torch.unique(sampled_env_idx)
            n_elem = int(self._tracking_env_idx.numel())
        else:
            self._tracking_env_idx = torch.randperm(
                n_envs, device=self.device
            )[:n_elem]
        # This is a rollout-context clock, deliberately independent of per-env
        # episode ages. Using ``any(step_count % traj_len == 0)`` made one random
        # env reset resample the complete tracking population.
        self._tracking_phase = 0
        self._tracking_resample_count = (
            int(getattr(self, "_tracking_resample_count", 0)) + 1
        )
        traj_len = self.cfg.rollout_expert_trajectories_length
        # Decide global root_h flag BEFORE z encoding.
        grh_prob = getattr(self.cfg, "terrain_variant_root_h_prob", 0.0)
        use_global = torch.rand(n_elem, device=self.device) < grh_prob
        global_rh = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
        global_rh[self._tracking_env_idx] = use_global
        self._tracking_terrain_variant_root_h = global_rh
        # Global FB: sample active mask once per tracking episode.
        global_fb_prob = getattr(self.cfg, "global_fb_zero_prob", 0.5)
        self._tracking_global_fb_active = torch.rand(n_elem, device=self.device) >= global_fb_prob
        # Per-env T for the rollout tracking-z mean. Sampled once here and held
        # for the full tracking episode; it does not affect discriminator T.
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
            self._tracking_T = torch.full(
                (n_elem,), T_min, device=self.device, dtype=torch.long
            )
        # Store robot pose for reference viz.
        if robot_root_xy is not None:
            self._tracking_robot_xy = robot_root_xy[self._tracking_env_idx].to(self.device).clone()
        else:
            self._tracking_robot_xy = None

        # --- Two-frame rollout anchor A_anchor (sim <-> motion) -------------
        # Sample ONE offset A_anchor per tracking window, in the init-LOCAL
        # frame, used on BOTH sides: env ``anchored_pose`` anchor = A_init·
        # A_anchor (sim space, set by the runner), tracking-z encoded under
        # A^m_init·A_anchor (motion space, via sample_tracking_trajectories).
        # Each side's init pose cancels in its own local frame so the obs/z
        # frames coincide, while A_anchor != 0 DISPLACES the spatial goal.
        # Distribution mirrors p_A: prob ``anchor_alpha_gt`` at the origin
        # (A_anchor=0 -> spawn-anchored), else uniform ±range xy / ±π yaw.
        a_alpha = float(getattr(self.cfg, "anchor_alpha_gt", 0.34))
        a_range = float(getattr(self.cfg, "rollout_anchor_xy_range",
                                getattr(self.cfg, "anchor_random_xy_range", 0.0)))
        # No-anchor control: never displace the rollout anchor (A_anchor=0), so
        # tracking-z stays spawn(=origin)-anchored and the robot tracks the
        # reference's intrinsic world path directly from origin/+x.
        if bool(getattr(self.cfg, "anchor_disable", False)):
            a_alpha, a_range = 1.0, 0.0
        is_zero = torch.rand(n_elem, device=self.device) < a_alpha
        aA_xy = (torch.rand(n_elem, 2, device=self.device) * 2 - 1) * a_range
        aA_yaw = (torch.rand(n_elem, device=self.device) * 2 - 1) * math.pi
        aA_xy = torch.where(is_zero.unsqueeze(-1), torch.zeros_like(aA_xy), aA_xy)
        aA_yaw = torch.where(is_zero, torch.zeros_like(aA_yaw), aA_yaw)
        self._tracking_anchor_canon_xy = aA_xy
        self._tracking_anchor_canon_yaw = aA_yaw
        # The sim-space env anchor (A_init·A_anchor in world) is computed by the
        # RUNNER, AFTER any terrain RSI reset settles, from the live (post-reset)
        # robot pose + this canonical offset — see fb_cpr_runner. Computing it
        # here would use the PRE-reset pose and be stale for terrain envs.

        # Sample trajectories first (sets _tracking_motion_ids/starts/lens).
        # Pass A_anchor (canonical frame) so the tracking-z anchored_pose + priv
        # reframe encode under the SAME offset the env applies to the obs.
        batch = expert_buffer.sample_tracking_trajectories(
            n_elem, traj_len,
            anchor_canon_xy=aA_xy, anchor_canon_yaw=aA_yaw)
        self._tracking_motion_ids = batch["motion_ids"].to(self.device)
        self._tracking_starts = batch["starts"].to(self.device)
        self._tracking_motion_lens = batch["motion_lens"].to(self.device)
        early_fraction = float(
            getattr(self.cfg, "tracking_early_termination_fraction", 0.0)
        )
        early_fraction = min(max(early_fraction, 0.0), 1.0)
        n_early = min(n_elem, int(round(early_fraction * n_elem)))
        if (
            n_early > 0
            and getattr(
                expert_buffer, "_tracking_failure_bin_frames", 0
            ) <= 0
        ):
            raise ValueError(
                "tracking_early_termination_fraction requires "
                "tracking_failure_bin_size_s > 0"
            )
        self._tracking_early_termination_count = n_early
        self._tracking_early_termination_mask = torch.zeros(
            n_elem, dtype=torch.bool, device=self.device
        )
        if n_early > 0:
            early_slots = torch.randperm(
                n_elem, device=self.device
            )[:n_early]
            self._tracking_early_termination_mask[early_slots] = True
        self._tracking_early_termination_active = (
            self._tracking_early_termination_mask.clone()
        )
        self._tracking_early_termination_valid = (
            self._tracking_early_termination_mask.clone()
        )
        self._tracking_early_termination_active_count = n_early
        self._tracking_failure_streak = torch.zeros(
            n_elem, dtype=torch.long, device=self.device
        )
        self._tracking_failure_attempt_age = torch.zeros(
            n_elem, dtype=torch.long, device=self.device
        )
        self._tracking_local_phases = torch.zeros(
            n_elem, dtype=torch.long, device=self.device
        )
        self._tracking_phase_hold_once = torch.zeros(
            n_elem, dtype=torch.bool, device=self.device
        )
        rt = batch.get("requires_terrain")
        self._tracking_requires_terrain = rt.to(self.device) if rt is not None else None
        if (
            n_early > 0
            and getattr(
                expert_buffer, "_tracking_failure_bin_frames", 0
            ) > 0
        ):
            self._tracking_attempt_bin_ids = expert_buffer.tracking_bin_ids(
                self._tracking_motion_ids,
                self._tracking_starts,
            ).to(self.device)
            self._tracking_attempt_bin_ends = (
                expert_buffer._tracking_bin_ends[
                    self._tracking_attempt_bin_ids.to(
                        expert_buffer.device
                    )
                ]
                .to(self.device)
            )
        else:
            self._tracking_attempt_bin_ids = None
            self._tracking_attempt_bin_ends = None
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
        # Return env indices for the caller to RSI-reset to the new motion's
        # frame. Normally ONLY terrain-tied motions need a reset (flat motions
        # keep their current physical pose and just swap z). But under
        # ``reset_tracking_on_resample`` we reset EVERY resampled tracking env so
        # the new tracking-z window starts from a clean RSI pose — required for
        # the no-anchor / origin-spawn setup, where the robot has DRIFTED from
        # origin mid-episode but the new origin-anchored z assumes it's back at
        # the motion's frame-0. Without the reset the new z's goal frame and the
        # robot's actual pose would be discontinuously mismatched.
        rt = self._tracking_requires_terrain
        reset_all = bool(getattr(self.cfg, "reset_tracking_on_resample", False))
        if reset_all:
            mask = torch.ones_like(self._tracking_env_idx, dtype=torch.bool)
        elif rt is not None and rt.any():
            mask = rt
        else:
            return None
        return {
            "env_ids": self._tracking_env_idx[mask],
            "motion_ids": self._tracking_motion_ids[mask],
            "starts": self._tracking_starts[mask],
        }

    def _update_tracking_bin_statistics(
        self,
        expert_buffer: Any,
        bin_ids: torch.Tensor,
        succeeded: torch.Tensor,
    ) -> None:
        """Apply completed segment outcomes to the expert-buffer EMA."""
        if (
            bin_ids.numel() == 0
            or not hasattr(
                expert_buffer, "update_tracking_bin_success_statistics"
            )
        ):
            return
        expert_buffer.update_tracking_bin_success_statistics(
            bin_ids,
            succeeded,
            ema_decay=float(
                getattr(
                    self.cfg,
                    "tracking_failure_priority_ema_decay",
                    0.90,
                )
            ),
            priority_scale=float(
                getattr(self.cfg, "tracking_failure_priority_scale", 3.0)
            ),
            max_multiplier=float(
                getattr(
                    self.cfg,
                    "tracking_failure_priority_max_multiplier",
                    4.0,
                )
            ),
        )

    def _resample_failed_tracking_slots(
        self,
        failed_slots: torch.Tensor,
        expert_buffer: Any,
        terrain_z_fn=None,
    ) -> dict[str, torch.Tensor]:
        """Assign failed slots new failure-weighted bins and rebuild their z."""
        num_failed = int(failed_slots.numel())
        sampled = expert_buffer.sample_tracking_failure_bins(num_failed)
        motion_ids = sampled["motion_ids"]
        starts = sampled["starts"]
        anchor_xy = self._tracking_anchor_canon_xy[failed_slots]
        anchor_yaw = self._tracking_anchor_canon_yaw[failed_slots]
        traj_len = int(self.cfg.rollout_expert_trajectories_length)
        batch = expert_buffer.sample_tracking_trajectories(
            num_failed,
            traj_len,
            anchor_canon_xy=anchor_xy.to(expert_buffer.device),
            anchor_canon_yaw=anchor_yaw.to(expert_buffer.device),
            motion_ids=motion_ids,
            starts=starts,
            pad_to_motion_end=True,
        )
        new_z = self._sample_tracking_z(
            expert_buffer,
            num_failed,
            traj_len,
            terrain_variant_root_h=(
                self._tracking_terrain_variant_root_h[
                    self._tracking_env_idx[failed_slots]
                ]
            ),
            terrain_z_fn=terrain_z_fn,
            batch=batch,
            tracking_slots=failed_slots,
        )

        motion_ids_dev = motion_ids.to(self.device)
        starts_dev = starts.to(self.device)
        motion_lens_dev = batch["motion_lens"].to(self.device)
        self._tracking_motion_ids[failed_slots] = motion_ids_dev
        self._tracking_starts[failed_slots] = starts_dev
        self._tracking_motion_lens[failed_slots] = motion_lens_dev
        requires_terrain = batch.get("requires_terrain")
        if requires_terrain is not None:
            self._tracking_requires_terrain[failed_slots] = (
                requires_terrain.to(self.device)
            )
        self._tracking_z[failed_slots] = new_z
        self._tracking_local_phases[failed_slots] = 0
        self._tracking_phase_hold_once[failed_slots] = True
        self._tracking_failure_attempt_age[failed_slots] = 0
        self._tracking_attempt_bin_ids[failed_slots] = sampled[
            "bin_ids"
        ].to(self.device)
        self._tracking_attempt_bin_ends[failed_slots] = sampled[
            "ends"
        ].to(self.device)
        return {
            "motion_ids": motion_ids_dev,
            "reset_frames": starts_dev,
            "reset_offsets": torch.zeros_like(starts_dev),
        }

    @torch.no_grad()
    def get_tracking_failures(
        self,
        live_joint_pos: torch.Tensor,
        ref_joint_pos: torch.Tensor,
        live_priv: torch.Tensor,
        ref_priv: torch.Tensor,
        natural_dones: torch.Tensor,
        expert_buffer: Any,
        terrain_z_fn=None,
    ) -> dict[str, Any] | None:
        """Detect sustained local tracking failures for the enabled cohort.

        Joint positions are synchronized post-step canonical-order tensors.
        Root height is checked separately from ``max_local_self`` because joint
        error cannot detect a coherent whole-body vertical displacement.
        """
        enabled = getattr(self, "_tracking_early_termination_active", None)
        valid = getattr(self, "_tracking_early_termination_valid", None)
        tracking_envs = getattr(self, "_tracking_env_idx", None)
        if (
            enabled is None
            or valid is None
            or tracking_envs is None
            or int(
                getattr(
                    self, "_tracking_early_termination_active_count", 0
                )
            ) == 0
        ):
            return None

        env_ids = tracking_envs
        natural_slots = torch.nonzero(
            enabled & natural_dones[env_ids].bool(),
            as_tuple=False,
        ).squeeze(-1)
        if natural_slots.numel() > 0:
            enabled[natural_slots] = False
            valid[natural_slots] = False
            self._tracking_early_termination_active_count -= int(
                natural_slots.numel()
            )
        enabled_for_metrics = enabled.clone()
        eligible_count = int(
            getattr(self, "_tracking_early_termination_active_count", 0)
        )
        if eligible_count == 0:
            return None

        joint_mae, root_height_error = tracking_failure_metrics(
            live_joint_pos[env_ids],
            ref_joint_pos[env_ids],
            live_priv[env_ids], ref_priv[env_ids]
        )

        failed_frame = (
            joint_mae
            > float(
                getattr(
                    self.cfg,
                    "tracking_failure_joint_mae_threshold_rad",
                    0.20,
                )
            )
        ) | (
            root_height_error
            > float(
                getattr(
                    self.cfg,
                    "tracking_failure_root_height_threshold_m",
                    0.25,
                )
            )
        )
        grace = max(
            int(getattr(self.cfg, "tracking_failure_grace_steps", 10)), 0
        )
        attempt_age = self._tracking_failure_attempt_age
        failed_frame &= attempt_age >= grace
        failed_frame &= enabled

        self._tracking_failure_streak = torch.where(
            failed_frame,
            self._tracking_failure_streak + 1,
            torch.zeros_like(self._tracking_failure_streak),
        )
        consecutive = max(
            int(
                getattr(
                    self.cfg, "tracking_failure_consecutive_steps", 5
                )
            ),
            1,
        )
        failed_slots = torch.nonzero(
            self._tracking_failure_streak >= consecutive,
            as_tuple=False,
        ).squeeze(-1)
        failed_mask = torch.zeros_like(enabled)
        failed_mask[failed_slots] = True

        # A bin succeeds only after an enabled slot reaches its exclusive end
        # without triggering the sustained-error criterion on this step.
        local_t = self._tracking_local_time()
        final_frame = (self._tracking_motion_lens - 1).clamp_min(0)
        post_frames = torch.minimum(
            self._tracking_starts + local_t + 1,
            final_frame,
        )
        completed = (
            enabled
            & ~failed_mask
            & completed_tracking_bins(
                post_frames,
                self._tracking_attempt_bin_ends,
                final_frame,
            )
        )
        completed_slots = torch.nonzero(
            completed, as_tuple=False
        ).squeeze(-1)
        outcome_bins = []
        outcome_success = []
        if completed_slots.numel() > 0:
            outcome_bins.append(
                self._tracking_attempt_bin_ids[completed_slots]
            )
            outcome_success.append(
                torch.ones(
                    completed_slots.numel(),
                    dtype=torch.bool,
                    device=self.device,
                )
            )
            next_frames = torch.minimum(
                post_frames[completed_slots],
                final_frame[completed_slots],
            )
            next_bins = expert_buffer.tracking_bin_ids(
                self._tracking_motion_ids[completed_slots],
                next_frames,
            ).to(self.device)
            self._tracking_attempt_bin_ids[completed_slots] = next_bins
            self._tracking_attempt_bin_ends[completed_slots] = (
                expert_buffer._tracking_bin_ends[
                    next_bins.to(expert_buffer.device)
                ].to(self.device)
            )

            at_motion_end = (
                post_frames[completed_slots]
                >= final_frame[completed_slots]
            )
            ended_slots = completed_slots[at_motion_end]
            if ended_slots.numel() > 0:
                enabled[ended_slots] = False
                valid[ended_slots] = False
                self._tracking_early_termination_active_count -= int(
                    ended_slots.numel()
                )

        if failed_slots.numel() > 0:
            outcome_bins.append(
                self._tracking_attempt_bin_ids[failed_slots]
            )
            outcome_success.append(
                torch.zeros(
                    failed_slots.numel(),
                    dtype=torch.bool,
                    device=self.device,
                )
            )
        if outcome_bins:
            self._update_tracking_bin_statistics(
                expert_buffer,
                torch.cat(outcome_bins),
                torch.cat(outcome_success),
            )

        attempt_age[enabled] += 1
        if failed_slots.numel() == 0:
            return {
                "joint_mae": joint_mae,
                "root_height_error": root_height_error,
                "enabled": enabled_for_metrics,
                "eligible_count": eligible_count,
                "bin_success_count": int(completed_slots.numel()),
            }

        self._tracking_failure_streak[failed_slots] = 0
        reset = self._resample_failed_tracking_slots(
            failed_slots,
            expert_buffer,
            terrain_z_fn=terrain_z_fn,
        )
        return {
            "env_ids": env_ids[failed_slots],
            "slots": failed_slots,
            "motion_ids": reset["motion_ids"],
            "reset_frames": reset["reset_frames"],
            "reset_offsets": reset["reset_offsets"],
            "joint_mae": joint_mae,
            "root_height_error": root_height_error,
            "enabled": enabled_for_metrics,
            "eligible_count": eligible_count,
            "bin_success_count": int(completed_slots.numel()),
        }

    def update_tracking_pose_after_reset(
        self,
        reset_env_ids: torch.Tensor,
        robot_root_xy: torch.Tensor,
        robot_root_quat: torch.Tensor,
        reset_frames: torch.Tensor | None = None,
        tracking_slots: torch.Tensor | None = None,
    ) -> None:
        """Update stored robot pose for terrain-reset envs.

        Called by the runner AFTER ``_reset_idx`` so that reference viz
        uses the post-reset robot position (not the stale pre-reset one).
        """
        if self._tracking_robot_xy is None or self._tracking_env_idx is None:
            return
        if tracking_slots is not None:
            slots = tracking_slots.to(self.device).long()
            mask = torch.zeros_like(self._tracking_env_idx, dtype=torch.bool)
            mask[slots] = True
        else:
            # Vectorized: find tracking slots that correspond to reset envs.
            reset_set = reset_env_ids.to(self.device)
            mask = (
                self._tracking_env_idx.unsqueeze(1)
                == reset_set.unsqueeze(0)
            ).any(dim=1)
            slots = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if not mask.any():
            return
        reset_eids = self._tracking_env_idx[mask]
        robot_xy = robot_root_xy[reset_eids].to(self.device)
        if self._tracking_heading_delta is not None and self._cached_root_quat_dev is not None:
            robot_yaw = self._yaw_from_quat(robot_root_quat[reset_eids].to(self.device))
            if reset_frames is not None:
                frames = reset_frames.to(self.device).long()
                motion_ids = self._tracking_motion_ids[slots]
                current_idx = (
                    self._cached_obs_starts_dev[motion_ids] + frames
                ).long()
                motion_yaw = self._yaw_from_quat(
                    self._cached_root_quat_dev[current_idx]
                )
                heading_delta = robot_yaw - motion_yaw
                anchor_idx = self._tracking_anchor_global_idx()[slots]
                anchor_xy = self._cached_root_pos_dev[anchor_idx, :2]
                motion_xy = self._cached_root_pos_dev[current_idx, :2]
                delta_xy = motion_xy - anchor_xy
                cos_d = torch.cos(heading_delta)
                sin_d = torch.sin(heading_delta)
                dx, dy = delta_xy[:, 0], delta_xy[:, 1]
                rotated_delta = torch.stack(
                    (
                        cos_d * dx - sin_d * dy,
                        sin_d * dx + cos_d * dy,
                    ),
                    dim=-1,
                )
                self._tracking_robot_xy[slots] = (
                    robot_xy - rotated_delta
                )
                self._tracking_heading_delta[slots] = heading_delta
                return
            anchor_idx = self._tracking_anchor_global_idx()[mask]
            motion_yaw = self._yaw_from_quat(
                self._cached_root_quat_dev[anchor_idx]
            )
            self._tracking_heading_delta[mask] = robot_yaw - motion_yaw
        self._tracking_robot_xy[mask] = robot_xy

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
        final_frame = (self._tracking_motion_lens - 1).clamp_min(0)
        frame0 = torch.minimum(self._tracking_starts, final_frame)
        return (self._cached_obs_starts_dev[self._tracking_motion_ids] + frame0).long()

    def _tracking_local_time(self) -> torch.Tensor:
        """Current per-slot frame offset, including failure rollbacks."""
        local = getattr(self, "_tracking_local_phases", None)
        if local is not None:
            return local
        phase = int(getattr(self, "_tracking_phase", 0))
        return torch.full_like(self._tracking_env_idx, phase)

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
        local_t = self._tracking_local_time()
        final_frame = (self._tracking_motion_lens - 1).clamp_min(0)
        frame = torch.minimum(
            self._tracking_starts + local_t.view(-1) + 1,
            final_frame,
        )
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

        local_t = self._tracking_local_time()
        final_frame = (self._tracking_motion_lens - 1).clamp_min(0)
        frame = torch.minimum(
            self._tracking_starts + local_t.view(-1) + 1,
            final_frame,
        )
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
        local_t = self._tracking_local_time()
        final_frame = (self._tracking_motion_lens - 1).clamp_min(0)
        frame = torch.minimum(
            self._tracking_starts + local_t.view(-1) + 1,
            final_frame,
        )
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
        local_t = self._tracking_local_time()
        final_frame = (self._tracking_motion_lens - 1).clamp_min(0)
        frame = torch.minimum(
            self._tracking_starts + local_t.view(-1) + 1,
            final_frame,
        )
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

    def _zbar_lambda(self) -> torch.Tensor | None:
        """Diagonal Lambda (per-feature tracking weight) for the z_bar map, sized
        to the recon head's base feature dim. Cfg ``zbar_feature_weights`` or
        all-ones. Cached. Returns None if no square-augmented linear recon head."""
        cached = getattr(self, "_zbar_lambda_cache", "unset")
        if cached != "unset":
            return cached
        head = getattr(self.policy, "_reconstruction_head", None)
        lam = None
        if head is not None and getattr(head, "square_augment", False):
            n = int(head.base_dim)
            w = tuple(getattr(self.cfg, "zbar_feature_weights", ()) or ())
            if w:
                assert len(w) == n, f"zbar_feature_weights len {len(w)} != base_dim {n}"
                lam = torch.tensor(w, dtype=torch.float32, device=self.device)
            else:
                lam = torch.ones(n, dtype=torch.float32, device=self.device)
        self._zbar_lambda_cache = lam
        return lam

    @torch.no_grad()
    def _zbar_from_obs(self, obs: dict) -> torch.Tensor | None:
        """Build projected z_bar for goals given by the W-target features.

        CRITICAL — the W head is trained to reconstruct the W-target slices from
        the NORMALIZED goal obs (backward_fb's recon_loss runs gather_target on
        ``fb_goal = train_next_obs`` AFTER the obs-normalizer). So the goal
        features ``g`` fed into z_bar = W^T c_g MUST be in the SAME normalized
        space, or c_g (raw) and W (normalized) are unit-inconsistent and the
        z_bar direction is wrong (project_z only fixes norm, not direction).
        We therefore NORMALIZE the obs here before slicing the goal features.
        ``obs`` arrives RAW (callers pass pre-normalization obs)."""
        head = getattr(self.policy, "_reconstruction_head", None)
        lam = self._zbar_lambda()
        if head is None or lam is None or not isinstance(obs, dict):
            return None
        obs_n = self.policy._normalize(obs)   # match W's training space
        try:
            goal_x = head.gather_base_target(obs_n)  # [B, n] normalized features
        except KeyError:
            return None
        return self.policy.zbar_from_goal(goal_x, lam)

    @torch.no_grad()
    def _sample_tracking_z(
        self,
        expert_buffer: Any,
        batch_dim: int,
        traj_length: int,
        terrain_variant_root_h: torch.Tensor | None = None,
        terrain_z_fn=None,
        batch: dict | None = None,
        tracking_slots: torch.Tensor | None = None,
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
        motion_ids = batch["motion_ids"].to(self.device)
        starts = batch["starts"].to(self.device)
        motion_lens = batch["motion_lens"].to(self.device)
        requires_terrain = batch.get("requires_terrain")
        if requires_terrain is not None:
            requires_terrain = requires_terrain.to(self.device)
        if tracking_slots is None:
            tracking_T = getattr(self, "_tracking_T", None)
            tracking_robot_xy = getattr(self, "_tracking_robot_xy", None)
            tracking_heading_delta = getattr(
                self, "_tracking_heading_delta", None
            )
        else:
            tracking_slots = tracking_slots.to(self.device).long()
            tracking_T_all = getattr(self, "_tracking_T", None)
            tracking_T = (
                tracking_T_all[tracking_slots]
                if tracking_T_all is not None
                else None
            )
            tracking_robot_xy_all = getattr(
                self, "_tracking_robot_xy", None
            )
            tracking_robot_xy = (
                tracking_robot_xy_all[tracking_slots]
                if tracking_robot_xy_all is not None
                else None
            )
            tracking_heading_delta_all = getattr(
                self, "_tracking_heading_delta", None
            )
            tracking_heading_delta = (
                tracking_heading_delta_all[tracking_slots]
                if tracking_heading_delta_all is not None
                else None
            )
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
            raw_frame = starts.unsqueeze(1) + arange
            final_frame = (motion_lens - 1).clamp_min(0).unsqueeze(1)
            frame_nxt = torch.minimum(raw_frame + 1, final_frame)
            is_t = requires_terrain
            obs_starts = self._cached_obs_starts_dev[motion_ids]
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
                    anchor_frame = torch.minimum(
                        starts, (motion_lens - 1).clamp_min(0)
                    )
                    anchor_idx = (
                        self._cached_obs_starts_dev[motion_ids]
                        + anchor_frame
                    ).long()
                    anchor_xy = self._cached_root_pos_dev[anchor_idx, :2]  # [B, 2]
                    anchor_xy_flat = anchor_xy.unsqueeze(1).expand(-1, traj_length, -1).reshape(-1, 2)
                    delta_xy = root_pos_xy - anchor_xy_flat
                    if tracking_robot_xy is not None and tracking_heading_delta is not None:
                        hd = tracking_heading_delta.unsqueeze(1).expand(-1, traj_length).reshape(-1)
                        cos_d = torch.cos(hd)
                        sin_d = torch.sin(hd)
                        dx, dy = delta_xy[:, 0], delta_xy[:, 1]
                        rot_xy = torch.stack([cos_d * dx - sin_d * dy,
                                              sin_d * dx + cos_d * dy], dim=-1)
                        rxy_flat = tracking_robot_xy.unsqueeze(1).expand(-1, traj_length, -1).reshape(-1, 2)
                        world_xy = rxy_flat + rot_xy
                    else:
                        world_xy = root_pos_xy
                    tz = terrain_z_fn(world_xy[nt_grh])
                    new_root_h[nt_grh] = new_root_h[nt_grh] + tz
            # Apply: replace priv[:, 0] where terrain_variant_root_h is set.
            priv = priv.clone()
            priv[grh_flat, 0] = new_root_h[grh_flat]
            next_obs["privileged_state"] = priv
        # Analytic z_bar goal features: capture the W-target features from the
        # RAW (pre-normalization) next_obs so g is in real feature units, matching
        # what the W head reconstructs. Built into per-frame z_bar below.
        _zbar_flat = None
        if float(getattr(self.cfg, "zbar_tracking_ratio", 0.0)) > 0.0:
            _zbar_flat = self._zbar_from_obs(next_obs)  # [B*T, d] or None
        next_obs = self.policy._normalize(next_obs)
        z = self.policy._backward_map(next_obs)
        z = z.view(batch_dim, traj_length, z.shape[-1])

        # Variable-T rolling mean: per-env window from self._tracking_T
        T_per_env = tracking_T
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

        # Analytic z_bar: for a ratio of tracking ENVS, replace the B-encoded
        # tracking-z with the per-frame z_bar built from the goal features +
        # learned W decoder. Per-env (all frames of a chosen env use z_bar) so the
        # whole rollout window is consistently driven by the analytic embedding.
        ratio = float(getattr(self.cfg, "zbar_tracking_ratio", 0.0))
        if ratio > 0.0 and _zbar_flat is not None:
            zbar = _zbar_flat.view(batch_dim, traj_length, -1)
            use_zbar = torch.rand(batch_dim, device=z.device) < ratio  # [B]
            z = torch.where(use_zbar.view(batch_dim, 1, 1), zbar, z)
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

    def _extra_phase1_reduce_nets(self):
        """Extra non-DDP-wrapped nets to include in the merged phase-1 grad
        all-reduce (no-op base). Subclasses (anchored: the spatial
        discriminator) return modules whose grads must reduce in the SAME
        collective, in rank-consistent order, rather than via a separate reduce
        that could race the phase-1 streams.

        The reconstruction head (sibling of B, trained by the folded recon_loss
        inside backward_fb) is included HERE when merge is on — otherwise its
        standalone reduce in step_fb races the phase-1 stream collectives and
        desyncs NCCL op order across ranks (XL-only first-backward hang)."""
        nets = []
        rh = getattr(self.policy, "_reconstruction_head", None)
        if rh is not None:
            nets.append(rh)
        return nets

    # --- update surface ----------------------------------------------------- #

    def broadcast_parameters(self) -> None:
        """Broadcast ALL policy parameters + buffers from rank 0 to every rank.

        Called once by the runner at init so every rank starts from the
        same weights. Also includes the EMA / BatchNorm running stats.
        """
        if not self.is_distributed:
            return
        # broadcast_object_list, but with the payload moved to CPU on the source
        # FIRST. The stock path pickles rank-0's state_dict WITH each tensor's
        # device (cuda:0); every receiving rank deserializes via torch.load,
        # restoring to that SAVED device -> all N ranks-per-node materialize the
        # payload on cuda:0 -> OOM on GPU 0 (observed at XL: one ~103 GB process
        # + 7 ranks piling ~10 GB each on cuda:0). Moving the state_dict to CPU
        # before the broadcast makes the pickle carry device='cpu', so the
        # deserialize lands in host RAM; load_state_dict then copies into each
        # rank's own (correct-device) params. ONE collective (fast, can't
        # desync) + no GPU pileup. Verified with a 4-rank gloo test.
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            cpu_sd = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                      for k, v in self.policy.state_dict().items()}
            objs = [cpu_sd]
        else:
            objs = [None]
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
        # The transformer actor owns its OWN frame BatchNorm (frame_norm), used
        # both at rollout (eval-mode act -> fills the per-rank replay) and train.
        # It's outside _obs_normalizer and DDP broadcast_buffers=False, so sync
        # its running stats here too (else ranks' running_mean/var drift).
        _actor = self._unwrap(self.policy._actor)
        if hasattr(_actor, "frame_norm"):
            all_bufs += list(_actor.frame_norm.buffers())

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

    @torch.no_grad()
    def _update_obs_running_stats(
        self,
        train_obs: dict[str, torch.Tensor],
        train_next_obs: dict[str, torch.Tensor],
    ) -> None:
        """Update observation moments from the training observations."""
        normalizer = self.policy._obs_normalizer
        if (
            not self.is_distributed
            or not bool(getattr(self.cfg, "obs_normalizer_global_moments", True))
        ):
            normalizer(train_obs)
            normalizer(train_next_obs)
            return

        entries: list[tuple[torch.nn.BatchNorm1d, torch.Tensor]] = []
        packed_parts: list[torch.Tensor] = []
        for obs in (train_obs, train_next_obs):
            for key, module in normalizer._normalizers.items():
                if key not in obs:
                    continue
                x = obs[key].detach()
                if x.ndim != 2:
                    raise ValueError(
                        f"Observation normalizer key {key!r} expected a 2D tensor, "
                        f"got shape {tuple(x.shape)}"
                    )
                bn = module._normalizer
                x64 = x.to(torch.float64)
                packed_parts.extend(
                    (
                        x64.sum(dim=0),
                        x64.square().sum(dim=0),
                        torch.tensor([x.shape[0]], device=x.device, dtype=torch.float64),
                    )
                )
                entries.append((bn, x))

        if not packed_parts:
            return
        packed = torch.cat(packed_parts)
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)

        offset = 0
        for bn, x in entries:
            width = x.shape[1]
            total = packed[offset: offset + width]
            offset += width
            total_sq = packed[offset: offset + width]
            offset += width
            count = packed[offset]
            offset += 1

            mean = total / count
            # BatchNorm stores the unbiased batch variance in running_var.
            var = (
                (total_sq - total.square() / count) / (count - 1.0)
            ).clamp_min_(0.0)
            if bn.num_batches_tracked is not None:
                bn.num_batches_tracked.add_(1)
            if bn.momentum is not None:
                momentum = float(bn.momentum)
            else:
                batches = (
                    int(bn.num_batches_tracked.item())
                    if bn.num_batches_tracked is not None
                    else 1
                )
                momentum = 1.0 / float(batches)
            bn.running_mean.lerp_(mean.to(bn.running_mean.dtype), momentum)
            bn.running_var.lerp_(var.to(bn.running_var.dtype), momentum)

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
        self._cached_target_action_dist = None

        expert_T_per_seq = None
        expert_sample_kwargs = {}
        if self._disc_positive_window > 0:
            expert_sampler = replay_buffer[self._EXPERT_KEY]
            expert_T_per_seq = (
                expert_sampler.peek("_mean_widths")
                if hasattr(expert_sampler, "peek")
                else None
            )
            if expert_T_per_seq is None:
                expert_T_per_seq = self._sample_expert_T(
                    self._disc_num_sequences,
                    int(self.policy.seq_length),
                )
            expert_sample_kwargs = {
                "seq_length": int(self.policy.seq_length),
                "mean_widths": expert_T_per_seq,
            }
        expert_batch = replay_buffer[self._EXPERT_KEY].sample(
            self._disc_batch_size,
            **expert_sample_kwargs,
        )
        train_batch = replay_buffer[self._REPLAY_KEY].sample(self.cfg.batch_size)

        train_obs = self._to_device(train_batch["observation"])
        train_next_obs = self._to_device(train_batch["next"]["observation"])
        train_action = train_batch["action"].to(self.device, non_blocking=True)
        # Transformer actor: stash the H+1 timestep window (per-position obs + z +
        # valid mask) for the parallel actor loss in backward_actor. None for MLP.
        self._train_actor_window = train_batch.get("actor_window", None)
        train_terminated = train_batch["next"]["terminated"].to(self.device, non_blocking=True)
        not_term = (~train_terminated.bool()).float()
        # --- gamma-conditioned F: sample a per-row gamma for the FB TD update ---
        # h = -log(1-gamma) ~ Uniform[h_S, h_L]; gamma = 1-exp(-h). The SAME
        # per-row gamma feeds F's conditioning input AND the TD-target discount,
        # so F(.,gamma) fits the successor measure at that horizon. fb_gamma is
        # None when the feature is off (F stays plain, discount = scalar).
        self._fb_gamma = None
        self._fb_gamma_alt = None
        self._fb_not_term = not_term
        if bool(getattr(self.cfg, "fb_gamma_conditioned", False)):
            g_l = float(self.cfg.discount)
            g_s = float(self.cfg.actor_gamma_short)
            # rand_like(not_term) so _fb_gamma matches not_term's shape ([B] or
            # [B,1]) — a [B] vs [B,1] mismatch would broadcast discount to [B,B].
            self._fb_gamma = sample_log_horizon_gamma(not_term, g_s, g_l)
            if bool(getattr(self.cfg, "fb_gamma_innovation_align", False)):
                self._fb_gamma_alt = sample_log_horizon_gamma(not_term, g_s, g_l)
            discount = self._fb_gamma * not_term
        else:
            discount = self.cfg.discount * not_term
        # Separate aux/disc discounts; default to main discount when None.
        _disc_aux = self.cfg.discount if self.cfg.discount_aux is None else self.cfg.discount_aux
        _disc_disc = self.cfg.discount if self.cfg.discount_disc is None else self.cfg.discount_disc
        discount_aux = float(_disc_aux) * not_term
        discount_disc = float(_disc_disc) * not_term

        expert_obs = self._to_device(expert_batch["observation"])
        expert_next_obs = self._to_device(expert_batch["next"]["observation"])

        # Anchor-frame body-pose reframe (no-op base). Runs on the RAW priv —
        # BEFORE the normalizer — so BN running-stats are accumulated on the
        # reframed (anchor-frame) privileged_state, not the heading frame. The
        # anchored subclass reframes train + expert priv body POS/ROT6D into a
        # per-row anchor A_i (sampled here and reused by the anchored_pose
        # preamble below), keeping B/F/critic/disc on one consistent frame.
        train_obs, train_next_obs, expert_obs, expert_next_obs = (
            self._anchor_priv_pre_normalize(
                train_batch, expert_batch,
                train_obs, train_next_obs, expert_obs, expert_next_obs))

        # Analytic z_bar relabel: capture the goal z_bar from the RAW (pre-
        # normalization) train_next_obs features, so a proportion of the relabel
        # z can be the W-decoder task embedding. Stashed for sample_mixed_z.
        self._relabel_zbar = None
        if float(getattr(self.cfg, "zbar_relabel_ratio", 0.0)) > 0.0:
            self._relabel_zbar = self._zbar_from_obs(train_next_obs)  # [batch, d] or None

        # Stash the RAW (pre-normalization) next_obs for the transformer actor's
        # TD-target next_action. The transformer actor is a RAW-obs consumer (it
        # owns its own frame BatchNorm), so feeding it the per-key-NORMALIZED
        # next_obs (below) would DOUBLE-normalize -> wrong next_action poisoning
        # every FB/critic/aux/entropy TD target. MLP actor ignores this (its
        # target path normalizes itself). Shallow-copied dict of the raw tensors.
        self._raw_train_next_obs = dict(train_next_obs)

        # Update from globally pooled sufficient statistics. Updating local
        # BatchNorm EMAs for all agent steps and averaging only afterward loses
        # between-rank variance and makes each rank normalize against stale,
        # rank-local moments during the update burst.
        self._update_obs_running_stats(train_obs, train_next_obs)

        # Freeze normalizer momentum for downstream passes.
        with torch.no_grad(), eval_mode(self.policy._obs_normalizer):
            train_obs = self.policy._obs_normalizer(train_obs)
            train_next_obs = self.policy._obs_normalizer(train_next_obs)
            expert_obs = self.policy._obs_normalizer(expert_obs)
            expert_next_obs = self.policy._obs_normalizer(expert_next_obs)

        # Anchoring preamble (Global-through-Anchoring): no-op in the base.
        # The anchored subclass overwrites the ``anchored_pose`` obs of
        # train_obs / train_next_obs under a per-row anchor A BEFORE z is built,
        # so sample_mixed_z (B(train_next_obs)) and the FB loss all see the SAME
        # anchored frame — the whole objective is then anchor-equivariant.
        train_obs, train_next_obs = self._anchor_obs_preamble(
            train_batch, train_obs, train_next_obs)

        # Stash the expert's RAW canonical next-pose (x,y,yaw) for the anchored
        # expert-z re-anchor seam (base ignores it). Sits at next.canon_pose
        # (the normalizer drops non-registered obs keys, so the buffer rides it
        # at the top level of ``next``).
        _ecp = expert_batch.get("next", {}).get("canon_pose", None)
        self._expert_canon_pose = (
            _ecp.to(self.device, non_blocking=True) if _ecp is not None else None
        )

        # Encode expert → z_expert (+ disc validity mask for variable T)
        expert_z, expert_disc_mask = self.encode_expert(
            next_obs=expert_next_obs,
            T_per_seq=expert_T_per_seq,
        )
        stored_train_z = train_batch["z"].to(self.device, non_blocking=True)

        # BFM order: disc sees ORIGINAL train_z (from rollout), THEN relabel.
        # The discriminator must train on the actual (s, z) pairs from the
        # replay — not freshly sampled z's that were never rolled out.
        #
        # Anchored seam: the rollout z was encoded under the SPAWN anchor, but
        # expert_z is encoded under the buffer's random p_A anchor — so the
        # disc could shortcut on the (anchor-correlated) z distribution instead
        # of judging motion style. The anchored subclass re-encodes the policy
        # disc-z as B(train_next_obs), which the preamble already anchored under
        # a per-row RANDOM A_i ~ p_A — matching expert_z's random anchor so the
        # anchor component is i.i.d. for both. Base = identity (rollout z).
        disc_train_z = self._disc_train_z(train_next_obs, stored_train_z)

        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self._zbuf_add(z)
        value_relabel_ratio = self.cfg.relabel_ratio
        if value_relabel_ratio is None:
            train_z = stored_train_z
            value_relabel_mask = torch.zeros(
                (stored_train_z.shape[0], 1),
                device=stored_train_z.device,
                dtype=torch.bool,
            )
        else:
            train_z, value_relabel_mask = sample_relabel_z(
                stored_train_z,
                z,
                float(value_relabel_ratio),
            )

        # FB and all critics retain the value-side relabel distribution above.
        # The actor may independently trade hindsight/expert z for paired
        # rollout z without changing its total batch size. None shares the
        # value-side tensor exactly for backward compatibility.
        actor_relabel_ratio = getattr(self.cfg, "actor_relabel_ratio", None)
        if actor_relabel_ratio is None:
            actor_train_z = None
            actor_relabel_mask = value_relabel_mask
        else:
            actor_train_z, actor_relabel_mask = sample_relabel_z(
                stored_train_z,
                z,
                float(actor_relabel_ratio),
            )

        # --- Anchoring seam (Global-through-Anchoring) -----------------
        # Default is identity: ``fb_goal`` is the transition's own next obs and
        # obs/z pass through unchanged, so non-anchored tasks are byte-identical.
        # The anchored subclass overrides ``_anchor_relabel`` to: sample a
        # coordinate anchor A, inject the anchored pose A^-1 g into
        # obs/next_obs, hindsight-relabel z's spatial block = B_spatial(task
        # goal s_h ~ p_goal). NOTE: ``fb_goal`` stays = next_obs — the FB loss
        # is a batch-matrix contrastive (Ms = F @ B.T) whose DIAGONAL is the
        # actual-transition reward (needs B(next_obs)) and whose OFF-DIAGONAL
        # rows are the independent rho/successor negatives. Passing a separate
        # query as goal breaks the diagonal and F never learns.
        fb_goal = train_next_obs
        train_obs, train_next_obs, fb_goal, train_z = self._anchor_relabel(
            train_batch=train_batch,
            train_obs=train_obs,
            train_next_obs=train_next_obs,
            train_z=train_z,
            mixed_z=z,
            expert_z=expert_z,
        )
        if actor_train_z is None:
            actor_train_z = train_z

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
        # Always update EMA moments for diagnostics. A positive fixed scale
        # replaces adaptive normalization while retaining those diagnostics.
        aux_reward_ema = self.policy._aux_reward_normalizer(aux_reward)
        aux_reward_sigma_min = (
            self.cfg.aux_reward_sigma_min
            if self.cfg.aux_actor_denormalize_q
            else 0.0
        )
        aux_reward = aux_reward_for_critic(
            aux_reward,
            aux_reward_ema,
            self.cfg.aux_reward_fixed_scale,
            self.policy._aux_reward_normalizer.S,
            aux_reward_sigma_min,
        )

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
            merged_nets = [
                self.policy._discriminator,  # not DDP-wrapped; use manually
                # For DDP-wrapped nets iterate .module so we target the real
                # parameters (grads live on the same tensors either way, but
                # .module.parameters() is the clean path).
                self._unwrap(self.policy._forward_map),
                self._unwrap(self.policy._backward_map),
                self._unwrap(self.policy._aux_critic),
            ]
            # Subclasses (anchored: spatial discriminator) add their own
            # phase-1 nets so they reduce in the SAME merged collective, in a
            # rank-consistent order — NOT a separate ad-hoc reduce that can
            # race the phase-1 streams / desync the NCCL op order.
            merged_nets += self._extra_phase1_reduce_nets()
            merged_handle = reduce_gradients_merged_async(merged_nets)
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
        metrics["Relabel/value_fraction"] = value_relabel_mask.float().mean()
        metrics["Relabel/actor_fraction"] = actor_relabel_mask.float().mean()

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
            z=actor_train_z,
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

    def _disc_train_z(self, train_next_obs, train_z):
        """Policy disc-z seam (no-op base = the rollout z). The anchored subclass
        re-encodes B(train_next_obs) so the disc's policy-z is drawn under the
        SAME random p_A anchor as expert_z — preventing the disc from shortcutting
        on the anchor-correlated z distribution."""
        return train_z

    def _anchor_priv_pre_normalize(self, train_batch, expert_batch,
                                   train_obs, train_next_obs,
                                   expert_obs, expert_next_obs):
        """Anchor-frame body-pose reframe (no-op base). Runs on RAW priv before
        the normalizer. The anchored subclass reframes train + expert
        ``privileged_state`` body POS/ROT6D into a per-row anchor A_i. Returns
        the (possibly reframed) ``(train_obs, train_next_obs, expert_obs,
        expert_next_obs)`` unchanged in the base."""
        return train_obs, train_next_obs, expert_obs, expert_next_obs

    def _anchor_obs_preamble(self, train_batch, train_obs, train_next_obs):
        """Anchoring preamble (no-op base). Runs BEFORE z is built. The anchored
        subclass overwrites the ``anchored_pose`` obs under a per-row anchor A so
        z (sample_mixed_z on B(train_next_obs)) and the FB loss share the frame.
        Returns the (possibly relabeled) ``(train_obs, train_next_obs)``."""
        return train_obs, train_next_obs

    def _anchor_relabel(self, train_batch, train_obs, train_next_obs, train_z,
                        mixed_z, expert_z):
        """Anchoring seam (no-op in the base FB-CPR-Aux). The base returns
        ``(obs, next_obs, fb_goal=next_obs, z)`` unchanged so non-anchored
        tasks are byte-identical. (Anchoring is now done in
        ``_anchor_obs_preamble`` before z is built; this seam is kept for
        back-compat / additional relabels.)"""
        return train_obs, train_next_obs, train_next_obs, train_z

    @torch.no_grad()
    def _target_next_action(self, next_obs, z):
        """Sample the TD-target next_action from the actor (clip=stddev_clip).

        MLP actor: call directly on the (already per-key-normalized) ``next_obs``.
        Transformer actor: it is a RAW-obs consumer (owns its frame BatchNorm), so
        feed the RAW pre-normalization next_obs (stashed as _raw_train_next_obs)
        through policy.actor() — which routes raw obs to the transformer — under
        eval_mode so frame_norm does NOT update its running stats from the target
        pass. This avoids the double-normalization that would otherwise corrupt
        every FB/critic/aux/entropy TD target.
        """
        p = self.policy
        dist = getattr(self, "_cached_target_action_dist", None)
        if dist is None:
            if isinstance(self._unwrap(p._actor), TransformerActorWrapper):
                raw = getattr(self, "_raw_train_next_obs", None)
                src = raw if raw is not None else next_obs
                with eval_mode(p._actor):
                    dist = p.actor(src, z, p.actor_std)
            else:
                dist = p._actor(next_obs, z, p.actor_std)
            # FB, critic and aux-critic use the same actor distribution but
            # independently sample from it. Caching removes two identical actor
            # forwards without correlating their target-smoothing noise.
            self._cached_target_action_dist = dist
        return dist.sample(clip=self.cfg.stddev_clip)

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
        # gamma-conditioned F: pass the per-row gamma sampled in update() to BOTH
        # the online and target F. Same gamma the TD target is discounted by
        # (``discount`` was set from _fb_gamma in update()). When innovation
        # alignment is enabled, batch the primary and independent alternate gamma
        # into one online-F call and one target-F call.
        fb_gamma = getattr(self, "_fb_gamma", None)
        fb_gamma_alt = getattr(self, "_fb_gamma_alt", None)
        normalized_forward = (
            fb_gamma is not None
            and bool(getattr(p, "forward_gamma_normalized_output", False))
        )
        align_innovations = (
            fb_gamma is not None
            and fb_gamma_alt is not None
            and bool(getattr(self.cfg, "fb_gamma_innovation_align", False))
        )

        def _cat_batch(x):
            if isinstance(x, dict):
                return {key: torch.cat((value, value), dim=0) for key, value in x.items()}
            return torch.cat((x, x), dim=0)

        with torch.no_grad():
            # next_action via actor (raw obs for the transformer actor — see
            # _target_next_action; avoids double-normalization in the TD target)
            next_action = self._target_next_action(next_obs, z)
            if align_innovations:
                batch_size = z.shape[0]
                gamma_pair = torch.cat((fb_gamma, fb_gamma_alt), dim=0)
                target_Fs_pair = p._target_forward_map(
                    _cat_batch(next_obs),
                    torch.cat((z, z), dim=0),
                    torch.cat((next_action, next_action), dim=0),
                    gamma_pair,
                )
                target_Fs, target_Fs_alt = target_Fs_pair.split(batch_size, dim=1)
            else:
                f_args = (fb_gamma,) if fb_gamma is not None else ()
                target_Fs = p._target_forward_map(next_obs, z, next_action, *f_args)
                target_Fs_alt = None
            if normalized_forward:
                target_Fs = gamma_forward_output_to_raw(target_Fs, fb_gamma)
                if target_Fs_alt is not None:
                    target_Fs_alt = gamma_forward_output_to_raw(
                        target_Fs_alt, fb_gamma_alt
                    )
            target_B = p._target_backward_map(goal)  # (B, d)
            target_Ms = torch.matmul(target_Fs, target_B.T)  # (num_par, B, B)
            _, _, target_M = self._pessimistic_value(target_Ms, self.cfg.fb_pessimism_penalty)
            if target_Fs_alt is not None:
                target_Ms_alt = torch.matmul(target_Fs_alt, target_B.T)
                _, _, target_M_alt = self._pessimistic_value(
                    target_Ms_alt, self.cfg.fb_pessimism_penalty
                )
            else:
                target_M_alt = None

        if align_innovations:
            Fs_pair = p._forward_map(
                _cat_batch(obs),
                torch.cat((z, z), dim=0),
                torch.cat((action, action), dim=0),
                torch.cat((fb_gamma, fb_gamma_alt), dim=0),
            )
            Fs, Fs_alt = Fs_pair.split(z.shape[0], dim=1)
        else:
            f_args = (fb_gamma,) if fb_gamma is not None else ()
            Fs = p._forward_map(obs, z, action, *f_args)
            Fs_alt = None
        if normalized_forward:
            Fs = gamma_forward_output_to_raw(Fs, fb_gamma)
            if Fs_alt is not None:
                Fs_alt = gamma_forward_output_to_raw(Fs_alt, fb_gamma_alt)
        B = p._backward_map(goal)
        Ms = torch.matmul(Fs, B.T)

        fb_diff = Ms - discount.view(-1, 1) * target_M
        fb_offdiag_sq = 0.5 * (fb_diff * self._off_diag).pow(2)
        gamma_loss_weights = None
        if bool(getattr(self.cfg, "fb_gamma_loss_weighting", False)):
            if fb_gamma is None:
                raise ValueError(
                    "fb_gamma_loss_weighting requires fb_gamma_conditioned=True"
                )
            gamma_loss_weights = normalized_gamma_loss_weights(
                fb_gamma.view(-1),
                float(self.cfg.actor_gamma_short),
                float(self.cfg.discount),
                float(getattr(self.cfg, "fb_gamma_loss_weight_power", 2.0)),
            )
            row_weights = gamma_loss_weights.view(1, -1, 1)
            fb_offdiag = (
                fb_offdiag_sq * row_weights
            ).sum() / self._off_diag_sum
            fb_diag_values = torch.diagonal(fb_diff, dim1=1, dim2=2)
            fb_diag = -(
                fb_diag_values * gamma_loss_weights.view(1, -1)
            ).mean() * Ms.shape[0]
        else:
            fb_offdiag = fb_offdiag_sq.sum() / self._off_diag_sum
            fb_diag = -torch.diagonal(
                fb_diff, dim1=1, dim2=2
            ).mean() * Ms.shape[0]
        fb_loss = fb_offdiag + fb_diag

        innovation_align_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        if Fs_alt is not None and target_M_alt is not None:
            # Keep alignment from moving the shared B space; B remains trained
            # by the main FB and orthogonality objectives.
            B_align_t = B.detach().T
            Ms_align = torch.matmul(Fs, B_align_t)
            Ms_alt = torch.matmul(Fs_alt, B_align_t)
            not_term = getattr(self, "_fb_not_term", torch.ones_like(fb_gamma_alt))
            discount_alt = fb_gamma_alt * not_term
            diff_align = Ms_align - discount.view(-1, 1) * target_M
            diff_alt = Ms_alt - discount_alt.view(-1, 1) * target_M_alt
            innovation_align_loss = innovation_alignment_loss(diff_align, diff_alt)
            fb_loss = fb_loss + (
                float(getattr(self.cfg, "fb_gamma_innovation_align_coef", 1.0))
                * innovation_align_loss
            )

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

        # Reconstruction regulariser: decode the configured slices of ``goal``
        # from ``z = B(goal)`` and minimise MSE. Pushes B to preserve task-
        # relevant state info that the bare FB loss may collapse out of z.
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

        # FB-side extra-loss seam (no-op base). The anchored subclass puts its
        # two-anchor VALUE-consistency here — it depends only on F (Q=<F,z> on
        # a detached action), so it MUST be optimized in the F update, not the
        # actor update (which only steps the actor optimizer).
        fb_extra, fb_extra_logs = self._fb_extra_loss(obs, z)
        fb_loss = fb_loss + fb_extra

        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        # DDP on F / B already fired async all_reduce INSIDE backward via
        # bucket hooks; nothing to fire here. Leave handles as None so
        # ``step_fb`` becomes a plain opt.step().
        F_handle = None
        B_handle = None

        with torch.no_grad():
            # One bad source row affects every goal column; one bad B(goal)
            # column affects every source row. Keep tail diagnostics separate
            # from the dense mean so random spike batches are attributable.
            batch_size = fb_diff.shape[-1]
            offdiag_count = max(batch_size - 1, 1)
            row_energy = fb_offdiag_sq.sum(dim=(0, 2)) / offdiag_count
            col_energy = fb_offdiag_sq.sum(dim=(0, 1)) / offdiag_count
            tail_quantile_every = int(
                getattr(self.cfg, "fb_tail_quantile_every", 1)
            )
            log_tail_quantiles = (
                self._fb_tail_diagnostic_step % tail_quantile_every == 0
            )
            self._fb_tail_diagnostic_step += 1
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
                "fb_offdiag_row_max": row_energy.max(),
                "fb_offdiag_col_max": col_energy.max(),
                "fb_offdiag_top_row_share": (
                    row_energy.max() / row_energy.sum().clamp_min(1e-12)
                ),
                "fb_innovation_align_loss": innovation_align_loss,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "q_loss": q_loss,
                "recon_loss": recon_loss,
            }
            if log_tail_quantiles:
                out["fb_offdiag_row_p99"] = torch.quantile(row_energy, 0.99)
                out["fb_offdiag_col_p99"] = torch.quantile(col_energy, 0.99)
            if fb_gamma is not None:
                gamma_flat = fb_gamma.view(-1)
                worst_row = row_energy.argmax()
                out["MutableGamma/fb_gamma_mean"] = gamma_flat.mean()
                out["MutableGamma/fb_gamma_max"] = gamma_flat.max()
                out["MutableGamma/fb_gamma_at_row_max"] = gamma_flat[worst_row]
                out["MutableGamma/fb_gamma_gt_0975"] = (
                    gamma_flat > 0.975
                ).float().mean()
                if gamma_loss_weights is not None:
                    out["MutableGamma/fb_loss_weight_mean"] = (
                        gamma_loss_weights.mean()
                    )
                    out["MutableGamma/fb_loss_weight_min"] = (
                        gamma_loss_weights.min()
                    )
                    out["MutableGamma/fb_loss_weight_max"] = (
                        gamma_loss_weights.max()
                    )
                high_gamma = gamma_flat > 0.975
                low_gamma = gamma_flat < 0.9
                out["MutableGamma/fb_offdiag_gt_0975"] = (
                    row_energy * high_gamma
                ).sum() / high_gamma.sum().clamp_min(1)
                out["MutableGamma/fb_offdiag_lt_09"] = (
                    row_energy * low_gamma
                ).sum() / low_gamma.sum().clamp_min(1)
                out["MutableGamma/fb_offdiag_gt_0975_max"] = torch.where(
                    high_gamma.any(),
                    row_energy.masked_fill(~high_gamma, -torch.inf).max(),
                    torch.zeros((), device=row_energy.device),
                )
                if fb_gamma_alt is not None:
                    gamma_alt_flat = fb_gamma_alt.view(-1)
                    h = -torch.log1p(-gamma_flat)
                    h_alt = -torch.log1p(-gamma_alt_flat)
                    out["MutableGamma/fb_gamma_alt_mean"] = gamma_alt_flat.mean()
                    out["MutableGamma/fb_gamma_h_gap_mean"] = (
                        h - h_alt
                    ).abs().mean()
            out.update(fb_extra_logs)
        return out, F_handle, B_handle

    def _fb_extra_loss(self, obs, z):
        """FB-side extra-loss seam (no-op base). Returns ``(loss, logs)``.
        Subclasses add F-dependent regularisers (e.g. anchor value
        consistency) here so they're optimized in the F update."""
        return torch.zeros((), device=self.device), {}

    def step_fb(
        self, F_handle: Any, B_handle: Any, clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        p = self.policy
        finish_async_reduce(F_handle)
        finish_async_reduce(B_handle)
        # DDP: the reconstruction head (linear W) is a sibling module, NOT inside
        # the DDP-wrapped _backward_map, so its grads (from the recon_loss folded
        # into fb_loss) are LOCAL. Reduce them so W (and the analytic z_bar=W^T c_g
        # it produces) stays consistent across ranks. backward_optimizer owns both
        # B and the recon head; reduce before its step.
        # NOTE: when merge_phase1_reduce is on (stream-parallel phase 1), the recon
        # head is instead folded into the merged collective via
        # _extra_phase1_reduce_nets() — a SEPARATE reduce here would race the
        # phase-1 streams and desync NCCL op order (XL-only first-backward hang).
        rh = getattr(p, "_reconstruction_head", None)
        if self.is_distributed and rh is not None and not getattr(self, "_merge_phase1_reduce", False):
            from ..utils import reduce_gradients
            reduce_gradients(rh)

        f_params = list(p._forward_map.parameters())
        b_params = list(p._backward_map.parameters())
        gn_f = _grad_norm_without_clipping(f_params)
        gn_b = _grad_norm_without_clipping(b_params)
        nonfinite = (~torch.isfinite(gn_f) | ~torch.isfinite(gn_b)).to(torch.int32)
        # F/B gradients have already been globally reduced by DDP hooks or the
        # merged phase-1 reducer, so this predicate is identical on every rank.
        # A second scalar collective here only adds one synchronization per
        # optimizer update.
        if bool(nonfinite.item()):
            self.forward_optimizer.zero_grad(set_to_none=True)
            self.backward_optimizer.zero_grad(set_to_none=True)
            return {
                "grad_norm/forward_map": gn_f.detach(),
                "grad_norm/backward_map": gn_b.detach(),
                "grad_nonfinite/fb_skipped": torch.ones((), device=gn_f.device),
            }

        spike_metrics: Dict[str, torch.Tensor] = {}
        dynamic_f_limit = None
        dynamic_b_limit = None
        if bool(getattr(self.cfg, "fb_grad_spike_clip", False)):
            decay = float(getattr(self.cfg, "fb_grad_spike_ema_decay", 0.99))
            multiplier = float(getattr(self.cfg, "fb_grad_spike_multiplier", 5.0))
            warmup = int(getattr(self.cfg, "fb_grad_spike_warmup_steps", 128))
            norm_f = float(gn_f.detach().item())
            norm_b = float(gn_b.detach().item())
            steps = self._fb_grad_norm_ema_steps

            ema_f = norm_f if steps == 0 else self._fb_grad_norm_ema_f
            ema_b = norm_b if steps == 0 else self._fb_grad_norm_ema_b
            next_ema_f, threshold_f, spike_f = ema_grad_spike_state(
                norm_f, ema_f, steps, decay, multiplier, warmup
            )
            next_ema_b, threshold_b, spike_b = ema_grad_spike_state(
                norm_b, ema_b, steps, decay, multiplier, warmup
            )
            if spike_f:
                dynamic_f_limit = max(ema_f, 1e-12)
            if spike_b:
                dynamic_b_limit = max(ema_b, 1e-12)

            # Winsorize the EMA observation at the pre-update threshold. A
            # single spike therefore cannot raise the baseline to its own size,
            # while a sustained scale change can still be followed gradually.
            self._fb_grad_norm_ema_f = next_ema_f
            self._fb_grad_norm_ema_b = next_ema_b
            self._fb_grad_norm_ema_steps = steps + 1

            spike_keys = (
                "grad_spike/forward_map",
                "grad_spike/backward_map",
                "grad_spike/forward_map_ema",
                "grad_spike/backward_map_ema",
                "grad_spike/forward_map_threshold",
                "grad_spike/backward_map_threshold",
                "grad_spike/forward_map_clip_scale",
                "grad_spike/backward_map_clip_scale",
            )
            spike_values = gn_f.new_tensor((
                float(spike_f),
                float(spike_b),
                ema_f,
                ema_b,
                threshold_f,
                threshold_b,
                min(1.0, max(ema_f, 1e-12) / max(norm_f, 1e-12))
                if spike_f else 1.0,
                min(1.0, max(ema_b, 1e-12) / max(norm_b, 1e-12))
                if spike_b else 1.0,
            ))
            spike_metrics = dict(zip(spike_keys, spike_values.unbind()))

        static_limit = (
            float(clip_grad_norm) if clip_grad_norm is not None else float("inf")
        )
        f_limit = min(
            static_limit,
            dynamic_f_limit if dynamic_f_limit is not None else float("inf"),
        )
        b_limit = min(
            static_limit,
            dynamic_b_limit if dynamic_b_limit is not None else float("inf"),
        )
        if math.isfinite(f_limit):
            torch.nn.utils.clip_grad_norm_(f_params, f_limit)
        if math.isfinite(b_limit):
            torch.nn.utils.clip_grad_norm_(b_params, b_limit)
        self.forward_optimizer.step()
        self.backward_optimizer.step()
        metrics = {
            "grad_norm/forward_map": gn_f.detach(),
            "grad_norm/backward_map": gn_b.detach(),
            "grad_nonfinite/fb_skipped": torch.zeros((), device=gn_f.device),
        }
        metrics.update(spike_metrics)
        return metrics

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
            next_action = self._target_next_action(next_obs, z)
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
            next_action = self._target_next_action(next_obs, z)
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
                "aux_reward_sigma_ema": p._aux_reward_normalizer.S.sqrt(),
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
            # Transformer actor consumes RAW obs (owns its frame BatchNorm); feed
            # the raw next_obs under eval_mode to avoid double-normalization +
            # stat pollution. MLP actor uses the normalized next_obs directly.
            if isinstance(self._unwrap(p._actor), TransformerActorWrapper):
                _raw = getattr(self, "_raw_train_next_obs", None)
                with eval_mode(p._actor):
                    next_dist = p.actor(_raw if _raw is not None else next_obs, z, p.actor_std)
            else:
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

    def _aux_q_for_actor(self, q_aux: torch.Tensor) -> torch.Tensor:
        """Optionally restore normalized Q_aux to current reward-scale units."""
        return aux_q_for_actor(
            q_aux,
            self.policy._aux_reward_normalizer.S,
            self.cfg.aux_actor_denormalize_q,
            self.cfg.aux_reward_fixed_scale,
            self.cfg.aux_reward_sigma_min,
        )

    def _backward_actor_transformer(self, win: dict, z: torch.Tensor):
        """Actor loss for the RoPE transformer actor — ALL H+1 TOKENS (temporal-parallel).

        The actor runs ONCE over the H+1 frame window and emits H+1 action
        distributions (one per timestep token). Every VALID position p is scored
        with the SAME FB actor objective (-Q_fb - reg*Q_disc - reg_aux*Q_aux) using
        that position's own state s_{t-p} (from the window) and the SHARED latent
        z=z_t. One backward; H+1 gradient signals per sample (more actor gradient).

        Two bugs that the previous version of this path had are kept fixed:
          * All three per-position Q's are reshaped to [BL] before combining, so
            the masked-mean is a true MEAN, not a sum-over-positions (the old
            [BL] vs [BL,1] broadcast that produced the "-2e6" explosion).
          * Invalid (zero-padded, post-reset) frames are excluded as attention
            KEYS (inside forward_window) and zeroed in the BatchNorm-stat path.
        CAVEAT (accepted): a past token p attends to a TRUNCATED causal context
        [t-H..t-p] that does not occur at rollout (where each step sees a full
        window), so past-token actions are mildly off-distribution.
        """
        p = self.policy
        actor = self._unwrap(p._actor)  # TransformerActorWrapper
        dev = self.device
        win_obs = {k: v.to(dev, non_blocking=True) for k, v in win["obs"].items()}
        valid = win["valid"].to(dev)                              # [B, L]
        B, L = valid.shape                                        # L = H+1

        # Raw per-position frames [B, L, 93] = [state | last_action] (training
        # order matches the wrapper's _current_frame). Zero the invalid positions
        # so the BatchNorm-stat path matches rollout (the attention key-mask in
        # forward_window additionally excludes them from attention).
        frames = torch.cat([win_obs["state"], win_obs["last_action"]], dim=-1)
        frames = frames * valid.unsqueeze(-1).to(frames.dtype)

        # Actor forward over the window with the SINGLE shared z_t token ->
        # H+1 action means -> sample. valid masks invalid frames out of both the
        # frame BatchNorm stats and the attention keys. Use the COMPILED
        # forward_window when available (set up in __init__ under compile_mode) —
        # the eager wide-transformer forward here is the dominant per-step cost.
        fw = getattr(actor, "_compiled_forward_window", None) or actor.forward_window
        dist = fw(frames, z, p.actor_std, valid=valid)
        sampled_action = dist.sample(clip=self.cfg.stddev_clip)   # [B, L, A]

        # Flatten (B*L) to score every position with the single-step F/critic.
        # z_t is repeated to every position for the per-position FB-Q evaluation.
        BL = B * L
        flat_obs_raw = {k: v.reshape(BL, *v.shape[2:]) for k, v in win_obs.items()}
        flat_obs = p._normalize(flat_obs_raw)
        flat_z = z.unsqueeze(1).expand(B, L, z.shape[-1]).reshape(BL, z.shape[-1])
        flat_a = sampled_action.reshape(BL, sampled_action.shape[-1])
        pol_cfg = self._unwrap(p).cfg

        # All three per-position Q's MUST be the SAME 1-D shape [BL]; otherwise the
        # combination below right-aligns [BL]+[BL,1] -> [BL,BL] and the masked-mean
        # becomes a SUM over positions (the "-2e6" explosion). Flatten each to [BL].
        Qs_disc = p._critic(flat_obs, flat_z, flat_a)
        if bool(getattr(pol_cfg, "critic_distributional", False)):
            Qs_disc = Qs_disc.mean(dim=-1, keepdim=True)
        _, _, Q_disc = self._pessimistic_value(Qs_disc, self.cfg.actor_pessimism_penalty)
        Q_disc = Q_disc.reshape(BL)
        Qs_aux = p._aux_critic(flat_obs, flat_z, flat_a)
        if bool(getattr(pol_cfg, "aux_critic_distributional", False)):
            Qs_aux = Qs_aux.mean(dim=-1, keepdim=True)
        _, _, Q_aux = self._pessimistic_value(Qs_aux, self.cfg.actor_pessimism_penalty)
        Q_aux = self._aux_q_for_actor(Q_aux.reshape(BL))
        # gamma-conditioned F: main term at gamma_L, plus a gamma_S short-horizon
        # term (mirrors the MLP actor path). f_gc guards non-conditioned F.
        _f_gc = bool(getattr(self.cfg, "fb_gamma_conditioned", False)) and \
            getattr(p, "forward_gamma_conditioned", False)
        _f_normalized = _f_gc and bool(
            getattr(p, "forward_gamma_normalized_output", False)
        )
        _fbshort_flat = None
        if _f_gc:
            _gL = torch.full((flat_z.shape[0],), float(self.cfg.discount), device=flat_z.device)
            _gS = torch.full((flat_z.shape[0],), float(self.cfg.actor_gamma_short), device=flat_z.device)
            _FsS = p._forward_map(flat_obs, flat_z, flat_a, _gS)
            _, _, _QfbS = self._pessimistic_value((_FsS * flat_z).sum(dim=-1), self.cfg.actor_pessimism_penalty)
            _fbshort_flat = _QfbS.reshape(BL)
            Fs = p._forward_map(flat_obs, flat_z, flat_a, _gL)
        else:
            Fs = p._forward_map(flat_obs, flat_z, flat_a)
        Qs_fb = (Fs * flat_z).sum(dim=-1)
        _, _, Q_fb = self._pessimistic_value(Qs_fb, self.cfg.actor_pessimism_penalty)
        Q_fb = Q_fb.reshape(BL)
        if _fbshort_flat is not None:
            _gL = float(self.cfg.discount); _gS = float(self.cfg.actor_gamma_short)
            _sc = float(self.cfg.actor_gamma_short_alpha)
            if not _f_normalized:
                _sc *= (1.0 - _gS) / max(1.0 - _gL, 1e-6)
            Q_fb = Q_fb + _sc * _fbshort_flat  # fold short-horizon into the FB Q used below
        Q_fb = float(getattr(self.cfg, "actor_fb_scale", 1.0)) * Q_fb

        # Per-position weight (scale_reg) over VALID positions only, then a
        # valid-masked MEAN over positions (not a sum).
        vmask = valid.reshape(BL).float()
        nval = vmask.sum().clamp_min(1.0)
        weight = (Q_fb.abs() * vmask).sum().detach() / nval if self.cfg.scale_reg else 1.0

        per_pos = -(Q_fb + self.cfg.reg_coeff * weight * Q_disc
                    + self.cfg.reg_coeff_aux * weight * Q_aux)    # [BL]
        actor_loss = (per_pos * vmask).sum() / nval

        self.actor_optimizer.zero_grad(set_to_none=True)
        # Only actor parameters are optimized in this phase. Restricting the
        # backward leaves still propagates through F/critic/aux-critic to the
        # sampled action, but it cannot accumulate their parameter gradients or
        # trigger their DDP reducer hooks.
        actor_loss.backward(inputs=tuple(actor.parameters()))
        # DDP: the transformer actor runs through the UNWRAPPED module
        # (forward_window is a custom method, not DDP.forward), so DDP's bucket
        # hooks never fire and the grads are LOCAL. Manually all-reduce them so
        # ranks stay in sync (the MLP-actor path reduces via DDP.__call__).
        # step_actor finishes the handle before optimizer.step.
        handle = reduce_gradients_async(actor) if self.is_distributed else None
        with torch.no_grad():
            out = {
                "actor_loss": actor_loss.detach(),
                "Q_fb": (Q_fb * vmask).sum().detach() / nval,
                "Q_discriminator": (Q_disc * vmask).sum().detach() / nval,
                "Q_aux": (Q_aux * vmask).sum().detach() / nval,
                "actor_tf/valid_frac": vmask.mean().detach(),
                "act_loc/abs_mean": dist.loc.abs().mean().detach(),
            }
        return out, handle

    def backward_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Any]:
        # The actor loss backpropagates through F / critic / aux_critic to the
        # sampled action, but their optimizers have already stepped and only the
        # actor optimizer steps below. Suppress those three DDP reducers during
        # their forwards; _backward_actor_impl also restricts backward leaves to
        # actor parameters so their hooks cannot participate in this phase.
        #
        # The MLP actor remains DDP-synchronized by its own hooks. The transformer
        # actor bypasses DDP.forward and is synchronized explicitly in
        # _backward_actor_transformer, so both paths issue exactly one actor
        # gradient synchronization in this phase.
        if self.is_distributed and self._is_ddp_wrapped:
            with contextlib.ExitStack() as actor_value_ctx:
                for net in (
                    self.policy._forward_map,
                    self.policy._critic,
                    self.policy._aux_critic,
                ):
                    actor_value_ctx.enter_context(net.no_sync())
                return self._backward_actor_impl(obs, action, z)
        return self._backward_actor_impl(obs, action, z)

    def _backward_actor_impl(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Any]:
        # Transformer actor: parallel FB -Q over all H+1 timestep positions.
        win = getattr(self, "_train_actor_window", None)
        if win is not None and isinstance(self._unwrap(self.policy._actor), TransformerActorWrapper):
            return self._backward_actor_transformer(win, z)
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
        Q_aux = self._aux_q_for_actor(Q_aux)
        # Q from FB (implicit Q = F·z). With gamma-conditioning, query F at the
        # LONG horizon gamma_L (= cfg.discount) for the main term, and add a
        # short-horizon term at gamma_S:
        #   Q_fb_L + (1-gamma_S)/(1-gamma_L) * alpha * Q_fb_S
        # The (1-gamma_S)/(1-gamma_L) factor rescales the short-horizon successor
        # measure (which sums ~1/(1-gamma) fewer steps) onto the long-horizon scale.
        fb_gc = bool(getattr(self.cfg, "fb_gamma_conditioned", False)) and \
            getattr(p, "forward_gamma_conditioned", False)
        fb_si = fb_gc and bool(getattr(self.cfg, "fb_stochastic_integral", False))
        normalized_forward = fb_gc and bool(
            getattr(p, "forward_gamma_normalized_output", False)
        )
        Bsz = z.shape[0]
        gL_args = ()
        Q_fb_short = None
        fb_short_term = torch.zeros((), device=z.device, dtype=z.dtype)
        self._q_fb_short_log = None  # reset each update (only set when conditioned)
        self._q_fb_L_log = None
        self._q_fb_S_log = None
        self._q_fb_integral_log = None
        self._q_fb_w_entropy_log = None
        self._q_fb_w_entropy_frac_log = None
        self._q_fb_w_top_log = None
        self._q_fb_w_profile = None
        self._q_fb_w_argmax_frac = None
        self._q_fb_w_tau_log = None

        if fb_si:
            # --- STOCHASTIC-INTEGRAL FB objective over the horizon -----------
            # Stratified-sample K horizons h_i in [h_lo, h_hi] (K even grids, one
            # uniform draw per grid, per row), gamma_i = 1-exp(-h_i). Batch-forward
            # F(s, pi, z, gamma_i), take the NORMALIZED per-step value
            # N_i = (1-gamma_i)*<F,z>. Integral weights w_i = softmax over horizons
            # of the TARGET-F normalized values (max-subtracted, no-grad, EMA-
            # stable); integrate the ONLINE N:  Q_final = sum_i w_i * N_i.
            K = int(getattr(self.cfg, "fb_integral_K", 8))
            gL = float(self.cfg.discount); gS = float(self.cfg.actor_gamma_short)
            h_lo = -math.log(max(1.0 - gS, 1e-6)); h_hi = -math.log(max(1.0 - gL, 1e-6))
            edges = torch.linspace(h_lo, h_hi, K + 1, device=z.device)          # [K+1]
            u = torch.rand(Bsz, K, device=z.device)
            hs = edges[:-1].view(1, K) + u * (edges[1:] - edges[:-1]).view(1, K)  # [B,K]
            gammas_k = 1.0 - torch.exp(-hs)                                       # [B,K]
            # Vectorized: fold the K horizons into the batch dim -> one forward
            # over B*K, then reshape back. obs is a dict; tile each key K-fold
            # (repeat_interleave so row order is [b0g0..b0gK-1, b1g0..]).
            def _tileK(t):
                return t.repeat_interleave(K, dim=0)
            obs_bk = {k: _tileK(v) for k, v in obs.items()} if isinstance(obs, dict) else _tileK(obs)
            z_bk = _tileK(z)                                                     # [B*K, d]
            a_bk = _tileK(sampled_action)                                        # [B*K, A]
            g_bk = gammas_k.reshape(-1)                                          # [B*K]
            F_bk = p._forward_map(obs_bk, z_bk, a_bk, g_bk)                      # [par, B*K, d]
            _, _, Q_bk = self._pessimistic_value((F_bk * z_bk).sum(dim=-1),
                                                 self.cfg.actor_pessimism_penalty)  # [B*K]
            N = normalized_forward_value(
                Q_bk, g_bk, normalized_forward
            ).reshape(Bsz, K)
            # Integral weights from the TARGET forward map (EMA), not the online
            # F — stabler weighting that doesn't chase the fast-moving online net.
            # The integrated N still uses the ONLINE F (gradient flows through N);
            # w is target-derived AND detached, so it's a fixed importance weight.
            with torch.no_grad():
                Ft_bk = p._target_forward_map(obs_bk, z_bk, a_bk, g_bk)          # [par, B*K, d]
                _, _, Qt_bk = self._pessimistic_value((Ft_bk * z_bk).sum(dim=-1),
                                                      self.cfg.actor_pessimism_penalty)
                Nt = normalized_forward_value(
                    Qt_bk, g_bk, normalized_forward
                ).reshape(Bsz, K)
                adaptive_tau = bool(getattr(self.cfg, "fb_integral_adaptive_tau", False))
                prior_lambda = float(getattr(self.cfg, "fb_integral_prior_lambda", 0.0))
                w, tau = stochastic_integral_weights(
                    Nt, hs, h_lo, prior_lambda, adaptive_tau
                )
                if adaptive_tau:
                    self._q_fb_w_tau_log = tau.mean()
                # Detached, scalar-only diagnostics. The per-grid profile is
                # retained here as [K], then emitted below as K separate scalars.
                wd = w.detach()
                ent = -(wd.clamp_min(1e-12) * wd.clamp_min(1e-12).log()).sum(dim=1)
                self._q_fb_w_entropy_log = ent.mean()
                self._q_fb_w_entropy_frac_log = ent.mean() / max(math.log(K), 1e-12)
                self._q_fb_w_top_log = wd.max(dim=1).values.mean()
                self._q_fb_w_profile = wd.mean(dim=0)
                self._q_fb_w_argmax_frac = (
                    wd.argmax(dim=1).float().mean() / max(K - 1, 1)
                )
            # Alignment scale: the integral is a per-step (normalized) value; the
            # standard-gamma FB Q lives at ~1/(1-gamma) magnitude. Multiply by
            # 1/(1-gamma_align) (default gamma_align=0.98 -> 50) so Q_final sits on
            # the same scale as the rest of the actor objective.
            g_align = float(getattr(self.cfg, "fb_integral_align_gamma", 0.98))
            align = 1.0 / max(1.0 - g_align, 1e-6)
            Q_integral = (w * N).sum(dim=1)                                      # [B] per-step
            Q_final = align * Q_integral                                        # aligned objective
            Q_fb = Q_final
            # Use the same aligned units for scale_reg so Q_disc/Q_aux preserve
            # their relative balance with the 50x-aligned FB objective.
            Q_fb_combined = Q_final
            # split-log needs an Fs; reuse the last grid's long-horizon slice
            # (F_bk row for the K-th sub-sample of each row ~ near gamma_L).
            Fs = F_bk[:, K - 1::K, :]                                            # [par, B, d]
            self._q_fb_integral_log = Q_final.mean().detach()
        else:
            if fb_gc:
                gL = torch.full((Bsz,), float(self.cfg.discount), device=z.device)
                gS = torch.full((Bsz,), float(self.cfg.actor_gamma_short), device=z.device)
                gL_args = (gL,)
                Fs_S = p._forward_map(obs, z, sampled_action, gS)
                _, _, Q_fb_short = self._pessimistic_value((Fs_S * z).sum(dim=-1),
                                                           self.cfg.actor_pessimism_penalty)
            Fs = p._forward_map(obs, z, sampled_action, *gL_args)
            Qs_fb = (Fs * z).sum(dim=-1)
            _, _, Q_fb = self._pessimistic_value(Qs_fb, self.cfg.actor_pessimism_penalty)
            # Short-horizon FB term added to the actor objective (0 when off).
            # ``Q_fb_combined`` [B] is the FULL FB objective the actor maximizes:
            #   Q_fb_L + (1-gamma_S)/(1-gamma_L) * alpha * Q_fb_S
            # It also drives the scale_reg ``weight`` (so the aux/disc reg terms
            # track the magnitude of the SUM of both scaled FB Qs, not just Q_fb_L).
            Q_fb_combined = Q_fb
            if Q_fb_short is not None:
                gL = float(self.cfg.discount); gS = float(self.cfg.actor_gamma_short)
                fb_short_scale = float(self.cfg.actor_gamma_short_alpha)
                if not normalized_forward:
                    fb_short_scale *= (1.0 - gS) / max(1.0 - gL, 1e-6)
                Q_fb_combined = Q_fb + fb_short_scale * Q_fb_short          # [B]
                fb_short_term = fb_short_scale * Q_fb_short.mean()
                # gamma-NORMALIZED logs: (1-gamma)*Q ~ per-step value, so the two
                # horizons are on a comparable scale (raw Q ~ 1/(1-gamma)).
                if normalized_forward:
                    self._q_fb_short_log = (
                        Q_fb_short.mean() / max(1.0 - gS, 1e-6)
                    ).detach()
                    self._q_fb_L_log = Q_fb.mean().detach()
                    self._q_fb_S_log = Q_fb_short.mean().detach()
                else:
                    self._q_fb_short_log = Q_fb_short.mean().detach()
                    self._q_fb_L_log = ((1.0 - gL) * Q_fb.mean()).detach()
                    self._q_fb_S_log = ((1.0 - gS) * Q_fb_short.mean()).detach()
        # Align the actor-side FB objective without changing F's TD targets.
        # Q_fb_combined also drives scale_reg, so Q_disc/Q_aux retain their
        # relative balance with the scaled direct FB term.
        actor_fb_scale = float(getattr(self.cfg, "actor_fb_scale", 1.0))
        Q_fb = actor_fb_scale * Q_fb
        Q_fb_combined = actor_fb_scale * Q_fb_combined
        fb_short_term = actor_fb_scale * fb_short_term
        if self._q_fb_integral_log is not None:
            self._q_fb_integral_log = actor_fb_scale * self._q_fb_integral_log
        # Optional per-block Q split (anchored variant: local vs spatial z).
        # No-op in the base. Stash for the metrics dict below.
        self._q_fb_split_logs = self._q_fb_split(Fs, z)

        if self.cfg.soft_fb:
            R = 1.0  # soft FB uses unit ball
            z_norms = z.norm(dim=-1)

        # scale_reg weight tracks the magnitude of the FULL FB objective
        # (Q_fb_combined = scaled sum of Q_fb_L and Q_fb_S). == Q_fb when
        # gamma-conditioning is off, so non-conditioned behavior is unchanged.
        if self.cfg.soft_fb and self.cfg.scale_reg:
            z_norm_clamped = z_norms.clamp(min=0.1 * R)
            Q_fb_normalized = Q_fb_combined * (R / z_norm_clamped)
            weight = Q_fb_normalized.abs().mean().detach()
        elif self.cfg.scale_reg:
            weight = Q_fb_combined.abs().mean().detach()
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
                - fb_short_term
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
                - fb_short_term
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
        # Preserve dQ/da through the value networks while excluding their
        # parameter leaves. Only the actor DDP reducer may run in this phase.
        actor_loss.backward(inputs=tuple(p._actor.parameters()))
        # DDP handled reduce inside backward.
        handle = None

        with torch.no_grad():
            out = {
                "actor_loss": actor_loss.detach(),
                "Q_discriminator": Q_discriminator.mean(),
                "Q_aux": Q_aux.mean(),
                "Q_fb": Q_fb.mean(),
            }
            # gamma-conditioned F: log BOTH horizons under the MutableGamma/
            # category. Raw Q ~ 1/(1-gamma), so also log gamma-NORMALIZED
            # (per-step) values which are comparable across horizons. Q_fb
            # (above) is the long-horizon Q_fb_L.
            if getattr(self, "_q_fb_short_log", None) is not None:
                out["MutableGamma/Q_fb_L_raw"] = Q_fb.mean()
                out["MutableGamma/Q_fb_S_raw"] = self._q_fb_short_log
                out["MutableGamma/Q_fb_L"] = self._q_fb_L_log   # (1-gamma_L)*Q_fb_L
                out["MutableGamma/Q_fb_S"] = self._q_fb_S_log   # (1-gamma_S)*Q_fb_S
            if getattr(self, "_q_fb_integral_log", None) is not None:
                out["MutableGamma/Q_fb_integral"] = self._q_fb_integral_log
                out["MutableGamma/w_entropy"] = self._q_fb_w_entropy_log
                out["MutableGamma/w_entropy_frac"] = self._q_fb_w_entropy_frac_log
                out["MutableGamma/w_top"] = self._q_fb_w_top_log
                out["MutableGamma/w_argmax_frac"] = self._q_fb_w_argmax_frac
                if self._q_fb_w_tau_log is not None:
                    out["MutableGamma/w_tau"] = self._q_fb_w_tau_log
                if self._q_fb_w_profile is not None:
                    for gi in range(self._q_fb_w_profile.numel()):
                        out[f"MutableGamma/w_grid{gi}"] = self._q_fb_w_profile[gi]
            out.update(act_stats)
            out.update(extra_logs)
            if getattr(self, "_q_fb_split_logs", None):
                out.update(self._q_fb_split_logs)
                self._q_fb_split_logs = None
            if hasattr(self, "_soft_fb_actor_logs"):
                out.update(self._soft_fb_actor_logs)
                del self._soft_fb_actor_logs
        return out, handle

    def _q_fb_split(self, Fs, z):
        """Per-block Q_fb split seam (no-op base). The anchored variant logs
        the local (<F,z>_local) and spatial (<F,z>_spatial) contributions to
        the implicit Q separately."""
        return {}

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
            "training_state": self.training_state_dict,
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
        self.load_training_state_dict(state.get("training_state", {}))


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
