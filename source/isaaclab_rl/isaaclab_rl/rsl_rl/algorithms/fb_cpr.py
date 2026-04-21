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
from ..utils import reduce_gradients

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

        # DDP LR scaling: ``lr_*  *= sqrt(world_size)`` for the fast-moving
        # branches (actor, critic, aux_critic, F). The slow branches (B,
        # discriminator) stay at BFM's 1e-5 — BFM's design intent is that
        # the z-encoder and style discriminator move slowly relative to
        # the actor/critics, and this holds regardless of global batch size.
        if self.is_distributed:
            import math
            ws = int(torch.distributed.get_world_size())
            s = math.sqrt(max(ws, 1))
            cfg.lr_actor = float(cfg.lr_actor) * s
            cfg.lr_critic = float(cfg.lr_critic) * s
            cfg.lr_aux_critic = float(cfg.lr_aux_critic) * s
            cfg.lr_f = float(cfg.lr_f) * s
            print(f"[FBCprAux] DDP world_size={ws}, sqrt-scaled LRs: "
                  f"actor={cfg.lr_actor:.3g} critic={cfg.lr_critic:.3g} "
                  f"aux_critic={cfg.lr_aux_critic:.3g} F={cfg.lr_f:.3g} "
                  f"(B={cfg.lr_b}, disc={cfg.lr_discriminator} unchanged)",
                  flush=True)

        # Put the policy on device. The policy holds *all* networks, including
        # obs normalizer + aux reward normalizer + target networks.
        self.policy.to(self.device)

        # Initialize + prepare for training.
        self.policy.train(True)
        self.policy.requires_grad_(True)
        self.policy.apply(weight_init)
        self.policy._prepare_for_train()

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
        self.backward_optimizer = torch.optim.Adam(
            p._backward_map.parameters(), lr=cfg.lr_b, weight_decay=cfg.weight_decay
        )
        self.forward_optimizer = torch.optim.Adam(
            p._forward_map.parameters(), lr=cfg.lr_f, weight_decay=cfg.weight_decay
        )
        self.actor_optimizer = torch.optim.Adam(
            p._actor.parameters(), lr=cfg.lr_actor, weight_decay=cfg.weight_decay
        )
        self.critic_optimizer = torch.optim.Adam(
            p._critic.parameters(), lr=cfg.lr_critic, weight_decay=cfg.weight_decay
        )
        self.aux_critic_optimizer = torch.optim.Adam(
            p._aux_critic.parameters(), lr=cfg.lr_aux_critic, weight_decay=cfg.weight_decay
        )
        self.discriminator_optimizer = torch.optim.Adam(
            p._discriminator.parameters(),
            lr=cfg.lr_discriminator,
            weight_decay=cfg.weight_decay_discriminator,
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
        """
        if not self.is_distributed:
            return
        world = float(torch.distributed.get_world_size())
        for buf in list(self.policy._obs_normalizer.buffers()) + list(self.policy._aux_reward_normalizer.buffers()):
            if buf is None or buf.dtype == torch.long:
                # Integer counters: take max (all ranks advanced together).
                if buf is not None and buf.dtype == torch.long:
                    b32 = buf.to(torch.float32)
                    torch.distributed.all_reduce(b32, op=torch.distributed.ReduceOp.MAX)
                    buf.copy_(b32.to(buf.dtype))
                continue
            torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
            buf.div_(world)

    def update(self, replay_buffer: Dict[str, Any], step: int) -> Dict[str, torch.Tensor]:
        """One full FB-CPR-Aux update step.

        Expects ``replay_buffer`` to contain:
          - ``"train"``: main replay, with ``.sample(batch_size)`` returning a
            dict like the one produced by the runner (`observation`, `action`,
            `z`, `next`, `aux_rewards`, etc.).
          - ``"expert_slicer"``: expert buffer, with ``.sample(batch_size)``
            returning at least `observation` and `next.observation`.
        """
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

        # 1) Discriminator update
        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=train_obs,
            train_z=train_z,
            grad_penalty=self.cfg.grad_penalty_discriminator
            if self.cfg.grad_penalty_discriminator > 0
            else None,
        )

        # 2) Mixed-z sampling + optional relabel
        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self._zbuf_add(z)
        if self.cfg.relabel_ratio is not None:
            mask = torch.rand(
                (self.cfg.batch_size, 1), device=self.device
            ) <= self.cfg.relabel_ratio
            train_z = torch.where(mask, z, train_z)

        q_loss_coef = self.cfg.q_loss_coef if self.cfg.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.clip_grad_norm if self.cfg.clip_grad_norm > 0 else None

        # 3) F-B update
        metrics.update(
            self.update_fb(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                goal=train_next_obs,
                z=train_z,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )
        # 4) Critic update
        metrics.update(
            self.update_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                z=train_z,
            )
        )
        # 5) Aux-critic update: assemble scaled aux reward
        aux_reward = torch.zeros(
            (self.cfg.batch_size, 1), device=self.device, dtype=torch.float32
        )
        aux_batch = train_batch.get("aux_rewards", None)
        if aux_batch is not None and len(self.cfg.aux_rewards_scaling) > 0:
            for name, scale in self.cfg.aux_rewards_scaling.items():
                if name not in aux_batch:
                    continue
                vals = aux_batch[name].to(self.device, non_blocking=True).view(-1, 1)
                metrics[f"aux_rew/{name}"] = vals.mean().detach()
                aux_reward = aux_reward + scale * vals
        # Pass through EMA reward normalizer (BFM's `RewardNormalizer(scale=True)`).
        aux_reward = self.policy._aux_reward_normalizer(aux_reward)
        metrics.update(
            self.update_aux_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                aux_reward=aux_reward,
                next_obs=train_next_obs,
                z=train_z,
            )
        )
        # 6) Actor update
        metrics.update(
            self.update_actor(
                obs=train_obs,
                action=train_action,
                z=train_z,
                clip_grad_norm=clip_grad_norm,
            )
        )

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

        # Sync running stats (BatchNorm running_mean/var, EMA moments) across
        # ranks so every rank's ``_obs_normalizer`` and ``_aux_reward_normalizer``
        # stay consistent. Cheap (just a handful of small buffers).
        if self.is_distributed:
            self._sync_running_stats()


        return metrics

    # --- individual update blocks ------------------------------------------ #

    def update_discriminator(
        self,
        expert_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        train_obs: torch.Tensor | dict[str, torch.Tensor],
        train_z: torch.Tensor,
        grad_penalty: float | None,
    ) -> Dict[str, torch.Tensor]:
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
        if self.is_distributed:
            reduce_gradients(self.policy._discriminator)
        self.discriminator_optimizer.step()

        with torch.no_grad():
            out = {
                "disc_loss": loss.detach(),
                "disc_expert_loss": expert_loss.detach().mean(),
                "disc_train_loss": unlabeled_loss.detach().mean(),
            }
            if wgan_gp is not None:
                out["disc_wgan_gp_loss"] = wgan_gp.detach()
        return out

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

    def update_fb(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        goal: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        q_loss_coef: float | None,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
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
        if self.is_distributed:
            reduce_gradients(p._forward_map)
            reduce_gradients(p._backward_map)
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(p._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(p._backward_map.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

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
        return out

    def update_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        p = self.policy
        num_parallel = p._critic.num_parallel
        with torch.no_grad():
            reward = p._discriminator.compute_reward(obs, z)
            dist = p._actor(next_obs, z, p.actor_std)
            next_action = dist.sample(clip=self.cfg.stddev_clip)
            next_Qs = p._target_critic(next_obs, z, next_action)
            Q_mean, Q_unc, next_V = self._pessimistic_value(
                next_Qs, self.cfg.critic_pessimism_penalty
            )
            target_Q = reward + discount.view(-1, 1) * next_V
            expanded = target_Q.expand(num_parallel, -1, -1)

        Qs = p._critic(obs, z, action)
        critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.is_distributed:
            reduce_gradients(p._critic)
        self.critic_optimizer.step()

        with torch.no_grad():
            out = {
                "target_Q": target_Q.mean(),
                "Q1": Qs.mean(),
                "mean_next_Q": Q_mean.mean(),
                "unc_Q": Q_unc.mean(),
                "critic_loss": critic_loss.mean(),
                "mean_disc_reward": reward.mean(),
            }
        return out

    def update_aux_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        aux_reward: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        p = self.policy
        num_parallel = p._aux_critic.num_parallel
        with torch.no_grad():
            dist = p._actor(next_obs, z, p.actor_std)
            next_action = dist.sample(clip=self.cfg.stddev_clip)
            next_Qs = p._target_aux_critic(next_obs, z, next_action)
            Q_mean, Q_unc, next_V = self._pessimistic_value(
                next_Qs, self.cfg.aux_critic_pessimism_penalty
            )
            target_Q = aux_reward + discount.view(-1, 1) * next_V
            expanded = target_Q.expand(num_parallel, -1, -1)

        Qs = p._aux_critic(obs, z, action)
        aux_critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded)

        self.aux_critic_optimizer.zero_grad(set_to_none=True)
        aux_critic_loss.backward()
        if self.is_distributed:
            reduce_gradients(p._aux_critic)
        self.aux_critic_optimizer.step()

        with torch.no_grad():
            out = {
                "target_auxQ": target_Q.mean(),
                "auxQ1": Qs.mean(),
                "mean_next_auxQ": Q_mean.mean(),
                "unc_auxQ": Q_unc.mean(),
                "aux_critic_loss": aux_critic_loss.mean(),
                "mean_aux_reward": aux_reward.mean(),
            }
        return out

    def update_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        p = self.policy
        dist = p._actor(obs, z, p.actor_std)
        sampled_action = dist.sample(clip=self.cfg.stddev_clip)

        # Q from discriminator-reward critic
        Qs_disc = p._critic(obs, z, sampled_action)
        _, _, Q_discriminator = self._pessimistic_value(
            Qs_disc, self.cfg.actor_pessimism_penalty
        )
        # Q from aux-reward critic
        Qs_aux = p._aux_critic(obs, z, sampled_action)
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
        if self.is_distributed:
            reduce_gradients(p._actor)
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(p._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            out = {
                "actor_loss": actor_loss.detach(),
                "Q_discriminator": Q_discriminator.mean(),
                "Q_aux": Q_aux.mean(),
                "Q_fb": Q_fb.mean(),
            }
        return out

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
