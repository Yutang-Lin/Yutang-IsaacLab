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

    # Discriminator
    grad_penalty_discriminator: float = 10.0

    # Reg-coeffs in the actor objective
    reg_coeff: float = 0.05       # weight on Q_discriminator inside actor loss
    reg_coeff_aux: float = 0.02   # weight on Q_aux_critic inside actor loss
    scale_reg: bool = True         # multiply regs by |Q_fb|.abs().mean().detach()

    # Mixed-z sampling (at training time and for the in-rollout ZBuffer)
    batch_size: int = 1024
    discount: float = 0.98
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

    # AMP (bf16) — kept False by default; enable only if you know why.
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
        # Discriminator special case: at large effective batch + cleaner
        # gradient the BCE classifier saturates very fast (decision margin
        # widens within a few hundred updates), which makes
        # ``log(D/(1-D))`` bimodal/heavy-tailed and turns the critic's TD
        # target into a hard, discontinuous regression problem. We dampen
        # disc's scaling to 0.25× of the other branches, clamped at a 1x
        # floor so single-rank / batch==1024 stays unchanged.
        import math
        REF_BATCH_SIZE = 1024
        DISC_SCALING_FACTOR = 0.25
        ws = (int(torch.distributed.get_world_size())
              if self.is_distributed else 1)
        bs_mult = math.sqrt(max(int(cfg.batch_size), 1) / REF_BATCH_SIZE)
        ws_mult = math.sqrt(max(ws, 1))
        combined_mult = ws_mult * bs_mult
        # Damped disc multiplier: keep 1x as floor so we never go BELOW
        # the single-rank base LR.
        disc_mult = max(1.0, DISC_SCALING_FACTOR * combined_mult)
        if combined_mult != 1.0:
            cfg.lr_actor = float(cfg.lr_actor) * combined_mult
            cfg.lr_critic = float(cfg.lr_critic) * combined_mult
            cfg.lr_aux_critic = float(cfg.lr_aux_critic) * combined_mult
            cfg.lr_f = float(cfg.lr_f) * combined_mult
            cfg.lr_b = float(cfg.lr_b) * combined_mult
            cfg.lr_discriminator = float(cfg.lr_discriminator) * disc_mult
            print(
                f"[FBCprAux] LR scaling: world_size={ws} (×{ws_mult:.3f})  "
                f"batch_size={cfg.batch_size}/{REF_BATCH_SIZE} (×{bs_mult:.3f})  "
                f"combined ×{combined_mult:.3f}  disc ×{disc_mult:.3f} "
                f"(0.25× combined, floored at 1)",
                flush=True,
            )
            print(
                f"[FBCprAux] scaled LRs: "
                f"actor={cfg.lr_actor:.3g} critic={cfg.lr_critic:.3g} "
                f"aux_critic={cfg.lr_aux_critic:.3g} F={cfg.lr_f:.3g} "
                f"B={cfg.lr_b:.3g} disc={cfg.lr_discriminator:.3g}",
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
            self._is_ddp_wrapped = True
            print(f"[FBCprAux] DDP-wrapped F/B/actor/critic/aux_critic "
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
        self.backward_optimizer = torch.optim.Adam(
            p._backward_map.parameters(),
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
            p._discriminator.parameters(),
            lr=cfg.lr_discriminator,
            weight_decay=cfg.weight_decay_discriminator,
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

    @property
    def optimizer_dict(self) -> Dict[str, Any]:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "backward_optimizer": self.backward_optimizer.state_dict(),
            "forward_optimizer": self.forward_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "aux_critic_optimizer": self.aux_critic_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
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

        # Expert-encoded z's
        perm = torch.randperm(batch, device=self.device)
        z = torch.where(mix_idxs == 1, expert_encodings[perm], z)
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
    ) -> torch.Tensor:
        """Encode expert sub-sequences through B, average over seq_length, project."""
        B_expert = self.policy._backward_map(next_obs).detach()
        seq_length = self.policy.seq_length
        assert self.cfg.batch_size % seq_length == 0, (
            f"batch_size ({self.cfg.batch_size}) must be divisible by seq_length ({seq_length})"
        )
        N = self.cfg.batch_size // seq_length
        B_expert = B_expert.view(N, seq_length, B_expert.shape[-1])
        z_expert = B_expert.mean(dim=1)
        z_expert = self.policy.project_z(z_expert)
        # Repeat-interleave back to [batch, d] so every row in the batch has
        # the sequence-level z for its parent sequence.
        z_expert = torch.repeat_interleave(z_expert, seq_length, dim=0)
        return z_expert

    @torch.no_grad()
    def maybe_update_rollout_context(
        self,
        z: torch.Tensor | None,
        step_count: torch.Tensor,
        expert_buffer: Any | None = None,
    ) -> torch.Tensor:
        """Update the rollout-time z context.

        Mirrors BFM-Zero's ``FBAgent.maybe_update_rollout_context``. Called once
        per env step by the runner with the current per-env ``step_count``.
        """
        if z is None:
            z = self.policy.sample_z(step_count.shape[0], device=self.device)
            if self.cfg.rollout_expert_trajectories and expert_buffer is not None:
                n_elem = max(
                    1,
                    int(
                        self.cfg.rollout_expert_trajectories_percentage
                        * step_count.shape[0]
                    ),
                )
                self._env_idx_with_expert_rollout = torch.randint(
                    0, step_count.shape[0], size=(n_elem,), device=self.device
                )
                self._tracking_z = self._sample_tracking_z(
                    expert_buffer,
                    n_elem,
                    self.cfg.rollout_expert_trajectories_length,
                )
                z[self._env_idx_with_expert_rollout] = self._tracking_z[:, 0]
            else:
                self._env_idx_with_expert_rollout = None
            return z

        # existing z — periodic refresh
        mask_reset_z = (step_count % self.cfg.update_z_every_step == 0).view(-1, 1)
        if self.cfg.use_mix_rollout and not self._zbuf_empty():
            new_z = self._zbuf_sample(z.shape[0])
        else:
            new_z = self.policy.sample_z(z.shape[0], device=self.device)
        z = torch.where(mask_reset_z, new_z, z.to(self.device))

        if self.cfg.rollout_expert_trajectories and expert_buffer is not None:
            idxs = step_count % self.cfg.rollout_expert_trajectories_length
            if bool((idxs == 0).any()):
                n_elem = max(
                    1,
                    int(
                        self.cfg.rollout_expert_trajectories_percentage
                        * step_count.shape[0]
                    ),
                )
                self._env_idx_with_expert_rollout = torch.randint(
                    0, step_count.shape[0], size=(n_elem,), device=self.device
                )
                self._tracking_z = self._sample_tracking_z(
                    expert_buffer,
                    n_elem,
                    self.cfg.rollout_expert_trajectories_length,
                )
            if getattr(self, "_env_idx_with_expert_rollout", None) is not None:
                mod_time = idxs[self._env_idx_with_expert_rollout].view(-1)
                T = self._tracking_z.shape[1]
                mod_time = torch.clamp(mod_time, 0, T - 1)
                z[self._env_idx_with_expert_rollout] = self._tracking_z[
                    torch.arange(len(self._env_idx_with_expert_rollout), device=self.device),
                    mod_time,
                ]
        return z

    @torch.no_grad()
    def _sample_tracking_z(
        self,
        expert_buffer: Any,
        batch_dim: int,
        traj_length: int,
    ) -> torch.Tensor:
        """Sample contiguous expert sub-trajectories and encode via B, with the
        BFM-Zero seq_length-window rolling mean."""
        seq_length = self.policy.seq_length
        batch = expert_buffer.sample(batch_dim * traj_length, seq_length=traj_length)
        next_obs = batch["next"]["observation"]
        next_obs = self._to_device(next_obs)
        z = self.policy._backward_map(next_obs)
        z = z.view(batch_dim, traj_length, z.shape[-1])
        for step in range(traj_length):
            end_idx = min(step + seq_length, traj_length)
            z[:, step] = z[:, step:end_idx].mean(dim=1)
        return self.policy.project_z(z)

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

        expert_batch = replay_buffer[self._EXPERT_KEY].sample(self.cfg.batch_size)
        train_batch = replay_buffer[self._REPLAY_KEY].sample(self.cfg.batch_size)

        train_obs = self._to_device(train_batch["observation"])
        train_next_obs = self._to_device(train_batch["next"]["observation"])
        train_action = train_batch["action"].to(self.device, non_blocking=True)
        train_terminated = train_batch["next"]["terminated"].to(self.device, non_blocking=True)
        discount = self.cfg.discount * (~train_terminated.bool()).float()

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

        # Encode expert → z_expert
        expert_z = self.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"].to(self.device, non_blocking=True)

        # Mixed-z sampling + optional relabel. Needed before any backward
        # since fb/critic/aux/actor all read train_z.
        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self._zbuf_add(z)
        if self.cfg.relabel_ratio is not None:
            mask = torch.rand(
                (self.cfg.batch_size, 1), device=self.device
            ) <= self.cfg.relabel_ratio
            train_z = torch.where(mask, z, train_z)

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
                        train_z=train_z,
                        grad_penalty=self.cfg.grad_penalty_discriminator
                        if self.cfg.grad_penalty_discriminator > 0
                        else None,
                    )
            else:
                disc_metrics, disc_handle = self.backward_discriminator(
                    expert_obs=expert_obs,
                    expert_z=expert_z,
                    train_obs=train_obs,
                    train_z=train_z,
                    grad_penalty=self.cfg.grad_penalty_discriminator
                    if self.cfg.grad_penalty_discriminator > 0
                    else None,
                )
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
                        goal=train_next_obs,
                        z=train_z,
                        q_loss_coef=q_loss_coef,
                    )
                with torch.cuda.stream(self._phase1_stream_aux):
                    aux_metrics, _ = self.backward_aux_critic(
                        obs=train_obs,
                        action=train_action,
                        discount=discount,
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
                    goal=train_next_obs,
                    z=train_z,
                    q_loss_coef=q_loss_coef,
                )
                aux_metrics, _ = self.backward_aux_critic(
                    obs=train_obs,
                    action=train_action,
                    discount=discount,
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
            discount=discount,
            next_obs=train_next_obs,
            z=train_z,
        )
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
    ) -> Tuple[Dict[str, torch.Tensor], Any]:
        """Compute disc loss, backward, fire async reduce. Returns (metrics, reduce_handle)."""
        disc = self.policy._discriminator
        expert_logits = disc.compute_logits(expert_obs, expert_z)
        unlabeled_logits = disc.compute_logits(train_obs, train_z)
        expert_loss = -F.logsigmoid(expert_logits)
        unlabeled_loss = F.softplus(unlabeled_logits)
        loss = torch.mean(expert_loss + unlabeled_loss)

        wgan_gp = None
        if grad_penalty is not None:
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

        # Orthonormality loss on B.
        Cov = torch.matmul(B, B.T)
        orth_loss_diag = -Cov.diag().mean()
        orth_loss_offdiag = 0.5 * (Cov * self._off_diag).pow(2).sum() / self._off_diag_sum
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss = fb_loss + self.cfg.ortho_coef * orth_loss

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
        with torch.no_grad():
            reward = p._discriminator.compute_reward(obs, z)
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
                "act_loc/abs_p95": loc.abs().float().quantile(0.95).detach(),
                "act_loc/frac_gt_0_9": (loc.abs() > 0.9).float().mean().detach(),
                "act_loc/frac_gt_0_99": (loc.abs() > 0.99).float().mean().detach(),
                "act_sample/abs_mean": sampled_action.abs().mean().detach(),
                "act_sample/frac_clamped": (
                    sampled_action.abs() >= (1.0 - 2.0 * dist.eps)
                ).float().mean().detach(),
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

        weight = Q_fb.abs().mean().detach() if self.cfg.scale_reg else 1.0
        actor_loss = (
            -Q_discriminator.mean() * self.cfg.reg_coeff * weight
            - Q_aux.mean() * self.cfg.reg_coeff_aux * weight
            - Q_fb.mean()
        )

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
        return out, handle

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
        self.policy.load_state_dict(state["policy"])
        optim = state.get("optimizers", {})
        for name, sd in optim.items():
            opt = getattr(self, name, None)
            if opt is not None:
                opt.load_state_dict(sd)
