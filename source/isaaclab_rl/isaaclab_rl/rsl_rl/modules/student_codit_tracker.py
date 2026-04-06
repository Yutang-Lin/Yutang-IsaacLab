# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CoDiT-Track: Condition-Denoising Distillation for Multi-Modal Humanoid Tracking.

A single-stage Transformer-based distillation policy that:
  - Uses fixed-length future condition tokens with stochastic corruption
  - Decomposes action into condition-invariant base + condition-dependent residual
  - Trains with two independently corrupted views per sample
  - Explicitly denoises future conditions as an auxiliary task

Design principles:
  - Fixed T future tokens, no horizon truncation
  - Corruption represents sparse / unreliable future information
  - Sparse keypoint control and sparse future control are unified under corruption
  - History token encodes condition-invariant dynamics context (cannot see future)
  - Base action is produced from the history token
  - Conditional action is produced from the proprio token after interacting with future
  - Future tokens must explicitly denoise their own clean values
  - Base branch is trained to explain the shared part across two corruption paths
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .actor_critic_transformer import ActorCriticTransformer
from .actor_critic_tf_recurrent import ActorCriticTFRecurrent
from .actor_critic_tf_recurrent_ml import ActorCriticTFRecurrentML
from .actor_critic_tf_recurrent_ll import ActorCriticTFRecurrentLL
from .actor_critic_tf_recurrent_latent import ActorCriticTFRecurrentLatent
from .actor_critic_transformer_latent import ActorCriticTransformerLatent
from .actor_critic_transformer_residual import ActorCriticTransformerResidual

from isaaclab_rl.rsl_rl.networks.codit_networks import FutureCorruptor, CoDiTTransformer
from isaaclab_rl.rsl_rl.networks.cvae_tracker_networks import _build_mlp


class StudentCoDiTTracker(nn.Module):
    """CoDiT-Track student policy for distillation.

    Uses a Condition-Denoising Transformer with:
    - MLP history encoder on env-provided history-stacked proprio obs
    - Two-level stochastic corruption of future keypoint conditions
    - T+2 token transformer with masked attention (history blocked from future)
    - Dual action heads: base (history) + conditional (proprio)
    - Future denoising auxiliary head
    - Two-view corruption training with shared explainability loss

    Follows the StudentTeacher contract for use with the Distillation algorithm.
    """

    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_policy_cfg,
        teacher_policy_ckpt,
        student_obs_meta,
        teacher_obs_meta,
        init_noise_std=0.1,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentCoDiTTracker.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.loaded_teacher = False

        if isinstance(num_student_obs, tuple):
            num_student_obs, _ = num_student_obs

        # -- Resolve obs meta: split into history proprio, current proprio, and condition indices --
        if "actor_obs_meta" in student_obs_meta:
            obs_meta = student_obs_meta["actor_obs_meta"]
        else:
            obs_meta = student_obs_meta
        self.student_obs_meta = obs_meta
        history_proprio_ids, current_proprio_ids, condition_ids = (
            self._resolve_obs_meta(num_student_obs, obs_meta)
        )
        self.register_buffer("history_proprio_ids", history_proprio_ids)
        self.register_buffer("current_proprio_ids", current_proprio_ids)
        self.register_buffer("condition_ids", condition_ids)

        history_proprio_dim = history_proprio_ids.shape[0]
        current_proprio_dim = current_proprio_ids.shape[0]
        cond_dim = condition_ids.shape[0]

        # -- Extract hyperparams from student_policy_cfg --
        cfg = dict(student_policy_cfg)
        cfg.pop("class_name", None)
        activation_name = cfg.pop("activation", "elu")
        activation = resolve_nn_activation(activation_name)

        # History encoder
        history_hidden_dims = cfg.pop("history_hidden_dims", [512, 256])

        # Keypoint geometry
        self.num_keypoints = cfg.pop("num_keypoints", 6)
        self.dims_per_keypoint = cfg.pop("dims_per_keypoint", 9)
        self.num_future_frames = cfg.pop("num_future_frames", 5)
        expected_cond_dim = self.num_future_frames * self.num_keypoints * self.dims_per_keypoint
        assert cond_dim == expected_cond_dim, (
            f"Condition dim mismatch: got {cond_dim}, expected T*K*D = "
            f"{self.num_future_frames}*{self.num_keypoints}*{self.dims_per_keypoint} = {expected_cond_dim}"
        )

        # Corruption config (flow-matching time range)
        t_range = tuple(cfg.pop("t_range", [0.0, 1.0]))

        # Transformer config
        tf_d_model = cfg.pop("tf_d_model", 256)
        tf_num_heads = cfg.pop("tf_num_heads", 4)
        tf_num_layers = cfg.pop("tf_num_layers", 3)
        tf_hidden_dim = cfg.pop("tf_hidden_dim", 512)
        tf_dropout = cfg.pop("tf_dropout", 0.0)
        tf_activation_name = cfg.pop("tf_activation", "gelu")
        if tf_activation_name == "gelu":
            tf_activation = nn.GELU(approximate="tanh")
        else:
            tf_activation = resolve_nn_activation(tf_activation_name)

        # Loss weights
        self.lambda_future = cfg.pop("lambda_future", 1.0)

        # Rollout corruption: per-episode parameters sampled at reset
        # Each keypoint independently: clean (t=0) or noisy (t sampled from rollout_t_range)
        self.rollout_t_range = tuple(cfg.pop("rollout_t_range", [0.0, 1.0]))
        self.rollout_p_clean_range = tuple(cfg.pop("rollout_p_clean_range", [0.0, 1.0]))

        self.num_actions = num_actions

        if cfg:
            print(f"StudentCoDiTTracker: unused config keys: {list(cfg.keys())}")

        # -- Build sub-networks --
        # History encoder: MLP on history-stacked proprio
        d_history = history_hidden_dims[-1] if history_hidden_dims else history_proprio_dim
        self.history_encoder = _build_mlp(history_proprio_dim, history_hidden_dims, d_history, activation)

        # Future corruption module (no learnable params)
        self.corruptor = FutureCorruptor(
            num_keypoints=self.num_keypoints,
            dims_per_keypoint=self.dims_per_keypoint,
            t_range=t_range,
        )

        # CoDiT Transformer
        self.transformer = CoDiTTransformer(
            proprio_dim=current_proprio_dim,
            history_dim=d_history,
            num_keypoints=self.num_keypoints,
            dims_per_keypoint=self.dims_per_keypoint,
            num_future_frames=self.num_future_frames,
            d_model=tf_d_model,
            num_heads=tf_num_heads,
            hidden_dim=tf_hidden_dim,
            num_layers=tf_num_layers,
            num_actions=num_actions,
            dropout=tf_dropout,
            activation=tf_activation,
        )

        # -- Load frozen teacher --
        teacher_ckpt = torch.load(teacher_policy_ckpt, map_location="cpu", weights_only=False)
        if "obs_norm_state_dict" in teacher_ckpt:
            self.obs_norm_state_dict = teacher_ckpt["obs_norm_state_dict"]
        else:
            self.obs_norm_state_dict = None

        teacher_policy_cfg = teacher_ckpt["policy_cfg"]
        teacher_policy_class = eval(teacher_policy_cfg.pop("class_name"))
        teacher_policy_args = teacher_policy_cfg.pop("_args")
        assert num_teacher_obs == teacher_policy_args[0], (
            f"Mismatch in teacher obs: {num_teacher_obs} vs {teacher_policy_args[0]}"
        )
        assert num_actions == teacher_policy_args[2], (
            f"Mismatch in actions: {num_actions} vs {teacher_policy_args[2]}"
        )
        self.teacher: ActorCritic = teacher_policy_class(*teacher_policy_args, **teacher_policy_cfg)
        self.teacher.load_state_dict(teacher_ckpt["model_state_dict"], strict=True)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.loaded_teacher = True

        print(f"StudentCoDiTTracker obs split: history_proprio={history_proprio_dim}, "
              f"current_proprio={current_proprio_dim}, conditions={cond_dim}")
        print(f"StudentCoDiTTracker keypoint geometry: T={self.num_future_frames}, "
              f"K={self.num_keypoints}, D={self.dims_per_keypoint}")
        print(f"StudentCoDiTTracker networks:")
        print(f"  History Encoder: {self.history_encoder}")
        print(f"  Transformer: {self.transformer}")
        print(f"  Teacher: {self.teacher}")

        # -- Action noise --
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

        # -- State --
        self._save_dict = {}
        self._save_log_dict = {}
        self._training_mode = False
        self._cached = {}
        # Per-env episode corruption state (lazily initialized on first act() call)
        # Precomputed at reset, held constant for entire episode (only ε changes per step)
        self._ep_t_combined: torch.Tensor | None = None  # [N, T, K]

    @property
    def student(self):
        """Return self so Distillation.broadcast_parameters() syncs trainable weights."""
        return self

    def _resolve_obs_meta(self, num_obs: int, obs_meta: dict):
        """Resolve observation metadata into history proprio, current proprio, and condition index tensors.

        CoDiT uses 3 groups (no posterior_conditions unlike CVAE):
          - conditions: future keypoint conditions (y_t)
          - current_proprio: current-frame proprio (o_t)
          - everything else: history-stacked proprio (H_t)
        """
        all_obs = torch.arange(num_obs)
        history_mask = torch.ones(num_obs, dtype=torch.bool)
        condition_obs = []
        current_proprio_obs = []

        if "conditions" in obs_meta:
            for seg in obs_meta["conditions"]:
                condition_obs.append(all_obs[seg["start"]:seg["end"]].clone())
                history_mask[seg["start"]:seg["end"]] = False

        if "current_proprio" in obs_meta:
            for seg in obs_meta["current_proprio"]:
                current_proprio_obs.append(all_obs[seg["start"]:seg["end"]].clone())
                history_mask[seg["start"]:seg["end"]] = False

        # Also exclude posterior_conditions from history if present (for compatibility)
        if "posterior_conditions" in obs_meta:
            for seg in obs_meta["posterior_conditions"]:
                history_mask[seg["start"]:seg["end"]] = False

        history_proprio_ids = all_obs[history_mask].clone().contiguous()
        current_proprio_ids = torch.cat(current_proprio_obs).contiguous() if current_proprio_obs else torch.tensor([], dtype=torch.long)
        condition_ids = torch.cat(condition_obs).contiguous() if condition_obs else torch.tensor([], dtype=torch.long)

        return history_proprio_ids, current_proprio_ids, condition_ids

    def _split_obs(self, obs: torch.Tensor):
        """Split observations into history proprio, current proprio, and conditions."""
        hp_t = obs[..., self.history_proprio_ids].contiguous()
        o_t = obs[..., self.current_proprio_ids].contiguous()
        y_t = obs[..., self.condition_ids].contiguous()
        return hp_t, o_t, y_t

    def _reshape_conditions(self, y_flat: torch.Tensor) -> torch.Tensor:
        """Reshape flat condition vector to structured [B, T, K, D].

        The obs function produces data in frame-major order:
        [frame_0_kp_0, frame_0_kp_1, ..., frame_0_kp_K, frame_1_kp_0, ...]
        """
        B = y_flat.shape[0]
        return y_flat.view(B, self.num_future_frames, self.num_keypoints, self.dims_per_keypoint)

    def _forward_clean(self, o_t: torch.Tensor, h_t: torch.Tensor, y_clean: torch.Tensor):
        """Forward pass with zero corruption (for deployment inference).

        Args:
            o_t: [B, proprio_dim]
            h_t: [B, d_history]
            y_clean: [B, T, K, D]

        Returns:
            a_total: [B, num_actions]
        """
        y_clean_out, tau = self.corruptor.no_corrupt(y_clean)
        y_flat = y_clean_out.flatten(start_dim=2)  # [B, T, K*D]
        a_base, a_cond, _ = self.transformer(o_t, h_t, y_flat, tau)
        return a_base + a_cond

    def _forward_corrupted(self, o_t: torch.Tensor, h_t: torch.Tensor, y_clean: torch.Tensor):
        """Forward pass with stochastic corruption (for training).

        Returns:
            a_base: [B, num_actions]
            a_cond: [B, num_actions]
            a_total: [B, num_actions]
            y_hat: [B, T, K*D] denoised future predictions
            tau: [B, T, K] corruption state (for logging)
        """
        y_corrupted, tau = self.corruptor.corrupt(y_clean)
        y_corrupted_flat = y_corrupted.flatten(start_dim=2)  # [B, T, K*D]
        a_base, a_cond, y_hat = self.transformer(o_t, h_t, y_corrupted_flat, tau)
        return a_base, a_cond, a_base + a_cond, y_hat, tau

    def forward(self):
        raise NotImplementedError

    def _sample_episode_corruption(self, env_ids: torch.Tensor, device: torch.device):
        """Sample and precompute per-keypoint corruption state for given env indices.

        Per env: each keypoint independently clean (t=0) or noisy (t sampled).
        Same mask across all T frames, held constant for the episode.
        """
        n = env_ids.shape[0]
        T, K = self.num_future_frames, self.num_keypoints

        # Per-env: sample noise level and clean probability
        t_level = torch.empty(n, device=device).uniform_(*self.rollout_t_range)
        p_clean = torch.empty(n, device=device).uniform_(*self.rollout_p_clean_range)

        # Per-keypoint Bernoulli: clean or noisy (same across all frames)
        kp_noisy = torch.rand(n, 1, K, device=device) > p_clean[:, None, None]  # [n, 1, K]
        t_kp = kp_noisy.float() * t_level[:, None, None]  # [n, 1, K]
        t_combined = t_kp.expand(n, T, K).contiguous()  # [n, T, K]

        # Snap t > 0.75 to 1.0 (pure noise)
        t_combined = torch.where(t_combined > 0.75, torch.ones_like(t_combined), t_combined)

        self._ep_t_combined[env_ids] = t_combined

    def act(self, observations, *args, **kwargs):
        """Rollout: forward with per-episode fixed corruption, sample with action noise.

        Obs from env contains clean y_t. We corrupt internally using precomputed
        per-env episode corruption (t_combined, tau_fixed). Only the Gaussian ε
        changes each step. The obs tensor is NOT modified, so the rollout buffer
        stores clean conditions for training.
        """
        hp_t, o_t, y_flat = self._split_obs(observations)
        h_t = self.history_encoder(hp_t)
        y_clean = self._reshape_conditions(y_flat)

        # Lazily init per-env corruption state
        B = observations.shape[0]
        T, K = self.num_future_frames, self.num_keypoints
        if self._ep_t_combined is None or self._ep_t_combined.shape[0] != B:
            # Create outside inference mode — these buffers are mutated in reset()
            with torch.inference_mode(False):
                self._ep_t_combined = torch.zeros(B, T, K, device=observations.device)
                self._sample_episode_corruption(torch.arange(B, device=observations.device), observations.device)

        # Corrupt with precomputed episode t (only ε is fresh)
        y_corrupted, tau = self.corruptor.corrupt_rollout(y_clean, self._ep_t_combined)
        y_corrupted_flat = y_corrupted.flatten(start_dim=2)
        a_base, a_cond, _ = self.transformer(o_t, h_t, y_corrupted_flat, tau)
        action_mean = a_base + a_cond

        std = self.std.expand_as(action_mean)
        self.distribution = Normal(action_mean, std)
        return self.distribution.sample()

    def evaluate(self, teacher_observations, *args, **kwargs):
        """Get teacher actions (frozen)."""
        with torch.no_grad():
            actions = self.teacher.act_inference(teacher_observations, *args, **kwargs)
        return actions

    def act_inference(self, observations, *args, **kwargs):
        """Training: single corrupted forward. Inference: clean forward.

        During training (_training_mode=True):
          - Applies fresh stochastic corruption to clean conditions
          - Runs one corrupted forward, caches for extra_loss (future denoising)
          - Each call samples new corruption, so each epoch sees different noise
          - Returns total action (a_base + a_cond) for standard behavior_loss

        During inference:
          - Forward with zero corruption (clean conditions, tau=0)
          - Returns a_base + a_cond
        """
        hp_t, o_t, y_flat = self._split_obs(observations)
        h_t = self.history_encoder(hp_t)
        y_clean = self._reshape_conditions(y_flat)

        if self._training_mode:
            a_base, a_cond, a_total, y_hat, tau = self._forward_corrupted(o_t, h_t, y_clean)

            # Cache for extra_loss (future denoising only)
            y_clean_flat = y_clean.flatten(start_dim=2)  # [B, T, K*D]
            self._cached = {
                "y_hat": y_hat,
                "y_clean": y_clean_flat,
                "tau": tau,
                "a_base": a_base,
                "a_cond": a_cond,
            }
            return a_total
        else:
            return self._forward_clean(o_t, h_t, y_clean)

    def extra_loss(self, **kwargs):
        """Compute future denoising auxiliary loss.

        Also logs action norms and corruption statistics.
        """
        if not self._cached:
            return {}, {}

        c = self._cached
        loss_dict = {}
        log_dict = {}

        # -- L_future: denoising loss --
        l_future = F.mse_loss(c["y_hat"], c["y_clean"])
        loss_dict["codit_future"] = l_future * self.lambda_future
        log_dict["codit_future"] = l_future.item()

        # -- Logging --
        with torch.no_grad():
            log_dict["codit_a_base_norm"] = c["a_base"].norm(dim=-1).mean().item()
            log_dict["codit_a_cond_norm"] = c["a_cond"].norm(dim=-1).mean().item()

            tau = c["tau"]
            log_dict["codit_t_mean"] = tau.mean().item()

            y_err = (c["y_hat"] - c["y_clean"]).pow(2).mean(dim=(0, 2))  # [T]
            log_dict["codit_future_err_mean"] = y_err.mean().item()

        self._cached = {}
        return dict(loss_dict), dict(log_dict)

    def pre_train(self):
        self._training_mode = True

    def after_train(self):
        self._training_mode = False
        self._cached = {}

    def reset(self, dones=None, hidden_states=None):
        """Resample per-episode corruption state for reset envs."""
        if dones is not None and self._ep_t_combined is not None:
            env_ids = dones.bool().flatten().nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() > 0:
                with torch.inference_mode(False):
                    self._sample_episode_corruption(env_ids, self._ep_t_combined.device)

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    def load_state_dict(self, state_dict, strict=True):
        """Load only student (non-teacher) parameters."""
        student_keys = [key for key in state_dict.keys() if not key.startswith("teacher")]
        student_params = {key: value for key, value in state_dict.items() if key in student_keys}
        super().load_state_dict(student_params, strict=False)
        return True

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
