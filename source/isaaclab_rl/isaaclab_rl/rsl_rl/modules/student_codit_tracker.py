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

        # Corruption config (flow-matching time ranges)
        t_keypoint_range = tuple(cfg.pop("t_keypoint_range", [0.0, 0.5]))
        t_frame_range = tuple(cfg.pop("t_frame_range", [0.0, 0.3]))

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
        self.lambda_shared = cfg.pop("lambda_shared", 0.5)
        self.lambda_base_cons = cfg.pop("lambda_base_cons", 0.1)
        self.lambda_act_view2 = cfg.pop("lambda_act_view2", 1.0)

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
            t_keypoint_range=t_keypoint_range,
            t_frame_range=t_frame_range,
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
        """Forward pass with zero corruption (for rollout and inference).

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
            tau: [B, T, K+1] corruption state (for logging)
        """
        y_corrupted, tau = self.corruptor.corrupt(y_clean)
        y_corrupted_flat = y_corrupted.flatten(start_dim=2)  # [B, T, K*D]
        a_base, a_cond, y_hat = self.transformer(o_t, h_t, y_corrupted_flat, tau)
        return a_base, a_cond, a_base + a_cond, y_hat, tau

    def forward(self):
        raise NotImplementedError

    def act(self, observations, *args, **kwargs):
        """Rollout: forward with clean conditions (no corruption), sample with noise."""
        hp_t, o_t, y_flat = self._split_obs(observations)
        h_t = self.history_encoder(hp_t)
        y_clean = self._reshape_conditions(y_flat)

        # No corruption during rollout — clean conditions, tau=0
        action_mean = self._forward_clean(o_t, h_t, y_clean)

        std = self.std.expand_as(action_mean)
        self.distribution = Normal(action_mean, std)
        return self.distribution.sample()

    def evaluate(self, teacher_observations, *args, **kwargs):
        """Get teacher actions (frozen)."""
        with torch.no_grad():
            actions = self.teacher.act_inference(teacher_observations, *args, **kwargs)
        return actions

    def act_inference(self, observations, *args, **kwargs):
        """Training: two-view corrupted forward. Inference: clean forward.

        During training (_training_mode=True):
          - Applies fresh stochastic corruption to clean conditions
          - Runs two independent corrupted forwards
          - Caches all intermediates for extra_loss computation
          - Each call samples new corruption, so each epoch sees different views
          - Returns view 1's total action for standard behavior_loss

        During inference:
          - Forward with zero corruption (clean conditions, tau=0)
          - Returns a_base + a_cond
        """
        hp_t, o_t, y_flat = self._split_obs(observations)
        h_t = self.history_encoder(hp_t)
        y_clean = self._reshape_conditions(y_flat)

        if self._training_mode:
            # Two independent corrupted views of the same clean future
            a_base_1, a_cond_1, a_1, y_hat_1, tau_1 = self._forward_corrupted(o_t, h_t, y_clean)
            a_base_2, a_cond_2, a_2, y_hat_2, tau_2 = self._forward_corrupted(o_t, h_t, y_clean)

            # Cache for extra_loss
            y_clean_flat = y_clean.flatten(start_dim=2)  # [B, T, K*D]
            self._cached = {
                "a_1": a_1, "a_2": a_2,
                "a_base_1": a_base_1, "a_base_2": a_base_2,
                "a_cond_1": a_cond_1, "a_cond_2": a_cond_2,
                "y_hat_1": y_hat_1, "y_hat_2": y_hat_2,
                "y_clean": y_clean_flat,
                "tau_1": tau_1, "tau_2": tau_2,
            }
            return a_1  # view 1's action for standard behavior_loss
        else:
            return self._forward_clean(o_t, h_t, y_clean)

    def extra_loss(self, **kwargs):
        """Compute CoDiT auxiliary losses from cached two-view training results.

        Losses:
          - L_act_view2: action distillation for view 2 (if teacher actions available)
          - L_future: future denoising loss (both views must reconstruct clean future)
          - L_shared: shared explainability (base branch explains other view's total action)
          - L_base_cons: weak base action consistency across views

        Also logs norms, variances, and corruption statistics.
        """
        if not self._cached:
            return {}, {}

        c = self._cached
        loss_dict = {}
        log_dict = {}

        # -- L_future: denoising loss for both views --
        # Each future token must reconstruct its own clean future value
        l_future_1 = F.mse_loss(c["y_hat_1"], c["y_clean"])
        l_future_2 = F.mse_loss(c["y_hat_2"], c["y_clean"])
        l_future = l_future_1 + l_future_2
        loss_dict["codit_future"] = l_future * self.lambda_future
        log_dict["codit_future"] = l_future.item()

        # -- L_shared: shared explainability loss --
        # Base branch should explain the shared, corruption-invariant part of the teacher-consistent action
        # Asymmetric through stop-gradient to prevent collapse
        l_shared = (
            F.mse_loss(c["a_base_1"], c["a_2"].detach()) +
            F.mse_loss(c["a_base_2"], c["a_1"].detach())
        )
        loss_dict["codit_shared"] = l_shared * self.lambda_shared
        log_dict["codit_shared"] = l_shared.item()

        # -- L_base_cons: weak base action consistency --
        l_base_cons = F.mse_loss(c["a_base_1"], c["a_base_2"])
        loss_dict["codit_base_cons"] = l_base_cons * self.lambda_base_cons
        log_dict["codit_base_cons"] = l_base_cons.item()

        # -- L_act view 2: action distillation for second corrupted view --
        privileged_actions = kwargs.get("privileged_actions_batch", None)
        if privileged_actions is not None:
            l_act_2 = F.mse_loss(c["a_2"], privileged_actions)
            loss_dict["codit_act_view2"] = l_act_2 * self.lambda_act_view2
            log_dict["codit_act_view2"] = l_act_2.item()

        # -- Logging: norms and variances --
        with torch.no_grad():
            log_dict["codit_a_base_norm"] = c["a_base_1"].norm(dim=-1).mean().item()
            log_dict["codit_a_cond_norm"] = c["a_cond_1"].norm(dim=-1).mean().item()
            log_dict["codit_a_base_var"] = c["a_base_1"].var(dim=0).mean().item()
            log_dict["codit_a_cond_var"] = c["a_cond_1"].var(dim=0).mean().item()

            # Corruption statistics (tau: [B, T, K+1])
            tau_1 = c["tau_1"]
            log_dict["codit_t_kp_mean"] = tau_1[:, :, :-1].mean().item()
            log_dict["codit_t_frame_mean"] = tau_1[:, :, -1].mean().item()

            # Per-timestep future prediction error
            y_err = (c["y_hat_1"] - c["y_clean"]).pow(2).mean(dim=(0, 2))  # [T]
            log_dict["codit_future_err_mean"] = y_err.mean().item()

        self._cached = {}
        return dict(loss_dict), dict(log_dict)

    def pre_train(self):
        self._training_mode = True

    def after_train(self):
        self._training_mode = False
        self._cached = {}

    def reset(self, dones=None, hidden_states=None):
        pass  # No recurrence, no temporal buffers

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
