# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CoDiT-MF: CoDiT-Track with MeanFlow denoising and contrastive features.

Extends CoDiT-Track with:
  - MeanFlow velocity prediction (replaces direct denoising)
  - JVP-based self-consistency training (75% instantaneous, 25% propagation)
  - Contrastive feature regularization on future token representations
  - Action heads unchanged: a_total = a_base + a_cond
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.func import jvp as func_jvp

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

from isaaclab_rl.rsl_rl.networks.codit_networks import FutureCorruptor
from isaaclab_rl.rsl_rl.networks.codit_mf_networks import CoDiTMFTransformer
from isaaclab_rl.rsl_rl.networks.cvae_tracker_networks import _build_mlp


class StudentCoDiTMFTracker(nn.Module):
    """CoDiT-MF student policy for distillation.

    Same architecture as StudentCoDiTTracker but with:
    - MeanFlow velocity prediction instead of direct denoising
    - JVP-based self-consistency loss (MeanFlow objective)
    - Contrastive feature regularization on future tokens
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
                "StudentCoDiTMFTracker.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.loaded_teacher = False

        if isinstance(num_student_obs, tuple):
            num_student_obs, _ = num_student_obs

        # -- Resolve obs meta --
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

        # -- Extract hyperparams --
        cfg = dict(student_policy_cfg)
        cfg.pop("class_name", None)
        activation_name = cfg.pop("activation", "elu")
        activation = resolve_nn_activation(activation_name)

        history_hidden_dims = cfg.pop("history_hidden_dims", [512, 256])

        self.num_keypoints = cfg.pop("num_keypoints", 6)
        self.dims_per_keypoint = cfg.pop("dims_per_keypoint", 9)
        self.num_future_frames = cfg.pop("num_future_frames", 5)
        expected_cond_dim = self.num_future_frames * self.num_keypoints * self.dims_per_keypoint
        assert cond_dim == expected_cond_dim, (
            f"Condition dim mismatch: got {cond_dim}, expected T*K*D = "
            f"{self.num_future_frames}*{self.num_keypoints}*{self.dims_per_keypoint} = {expected_cond_dim}"
        )

        # Corruption config
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
        self.lambda_mf = cfg.pop("lambda_mf", 1.0)
        self.lambda_contrast = cfg.pop("lambda_contrast", 0.01)
        self.mf_propagation_ratio = cfg.pop("mf_propagation_ratio", 0.25)

        # Rollout corruption
        self.rollout_t_range = tuple(cfg.pop("rollout_t_range", [0.0, 1.0]))
        self.rollout_p_clean_range = tuple(cfg.pop("rollout_p_clean_range", [0.0, 1.0]))

        self.num_actions = num_actions

        if cfg:
            print(f"StudentCoDiTMFTracker: unused config keys: {list(cfg.keys())}")

        # -- Build sub-networks --
        d_history = history_hidden_dims[-1] if history_hidden_dims else history_proprio_dim
        self.history_encoder = _build_mlp(history_proprio_dim, history_hidden_dims, d_history, activation)

        self.corruptor = FutureCorruptor(
            num_keypoints=self.num_keypoints,
            dims_per_keypoint=self.dims_per_keypoint,
            t_range=t_range,
        )

        self.transformer = CoDiTMFTransformer(
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
        assert num_teacher_obs == teacher_policy_args[0]
        assert num_actions == teacher_policy_args[2]
        self.teacher: ActorCritic = teacher_policy_class(*teacher_policy_args, **teacher_policy_cfg)
        self.teacher.load_state_dict(teacher_ckpt["model_state_dict"], strict=True)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.loaded_teacher = True

        print(f"StudentCoDiTMFTracker obs split: history_proprio={history_proprio_dim}, "
              f"current_proprio={current_proprio_dim}, conditions={cond_dim}")
        print(f"StudentCoDiTMFTracker: lambda_mf={self.lambda_mf}, lambda_contrast={self.lambda_contrast}, "
              f"propagation_ratio={self.mf_propagation_ratio}")

        # -- Action noise --
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

        # -- State --
        self._training_mode = False
        self._cached = {}
        self._ep_t_combined: torch.Tensor | None = None

    @property
    def student(self):
        return self

    def _resolve_obs_meta(self, num_obs: int, obs_meta: dict):
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

        if "posterior_conditions" in obs_meta:
            for seg in obs_meta["posterior_conditions"]:
                history_mask[seg["start"]:seg["end"]] = False

        history_proprio_ids = all_obs[history_mask].clone().contiguous()
        current_proprio_ids = torch.cat(current_proprio_obs).contiguous() if current_proprio_obs else torch.tensor([], dtype=torch.long)
        condition_ids = torch.cat(condition_obs).contiguous() if condition_obs else torch.tensor([], dtype=torch.long)
        return history_proprio_ids, current_proprio_ids, condition_ids

    def _split_obs(self, obs):
        hp_t = obs[..., self.history_proprio_ids].contiguous()
        o_t = obs[..., self.current_proprio_ids].contiguous()
        y_t = obs[..., self.condition_ids].contiguous()
        return hp_t, o_t, y_t

    def _reshape_conditions(self, y_flat):
        B = y_flat.shape[0]
        return y_flat.view(B, self.num_future_frames, self.num_keypoints, self.dims_per_keypoint)

    def _forward_clean(self, o_t, h_t, y_clean):
        """Inference: r=0, t=0 → clean conditions, velocity head unused for action."""
        B, T, K, D = y_clean.shape
        y_flat = y_clean.flatten(start_dim=2)  # [B, T, K*D]
        t = torch.zeros(B, T, K, device=y_clean.device)
        r = torch.zeros(B, T, K, device=y_clean.device)
        a_base, a_cond, _, _ = self.transformer(o_t, h_t, y_flat, t, r)
        return a_base + a_cond

    def _forward_corrupted_mf(self, o_t, h_t, y_clean):
        """Training forward with MeanFlow.

        Optimizations vs naive implementation:
        1. Single full forward for actions + u + features (no redundant pass)
        2. torch.func.jvp (forward-mode AD) instead of backward-mode simulation
        3. JVP only on propagation subset (r≠t), skip for instantaneous (r=t)

        Cost: ~1 full forward + ~0.25 * 2x JVP on subset + 1 contrastive forward ≈ 2.5x
        (vs 4x before: JVP backward-mode 2x + full forward 1x + contrastive 1x)
        """
        B, T, K, D = y_clean.shape
        device = y_clean.device

        # Sample t ~ U(0, 1) per keypoint per frame
        t = torch.empty(B, T, K, device=device).uniform_(
            self.corruptor.t_lo, self.corruptor.t_hi)

        # Sample r: per-sample decision (instantaneous vs propagation)
        prop_mask = torch.rand(B, device=device) < self.mf_propagation_ratio  # [B]
        r = t.clone()
        prop_indices = prop_mask.nonzero(as_tuple=True)[0]
        if prop_indices.numel() > 0:
            r[prop_indices] = torch.rand_like(t[prop_indices]) * t[prop_indices]

        # Corrupt
        eps = torch.randn(B, T, K, D, device=device)
        y_t = (1.0 - t).unsqueeze(-1) * y_clean + t.unsqueeze(-1) * eps
        v_t = eps - y_clean  # instantaneous velocity

        y_flat = y_t.flatten(start_dim=2)  # [B, T, K*D]
        v_flat = v_t.flatten(start_dim=2)  # [B, T, K*D]

        # 1. Full forward on entire batch → actions, u, features
        a_base, a_cond, u, features1 = self.transformer(o_t, h_t, y_flat, t, r)

        # 2. JVP only on propagation subset (r≠t) for du_dt
        # For instantaneous samples (r=t), t_minus_r=0 so du_dt doesn't matter
        du_dt = torch.zeros_like(u)
        if prop_indices.numel() > 0:
            from torch.nn.attention import sdpa_kernel, SDPBackend

            # Extract subset
            y_sub = y_flat[prop_indices]
            r_sub = r[prop_indices]
            t_sub = t[prop_indices]
            v_sub = v_flat[prop_indices]
            # Pre-compute constant tokens for subset
            tok_p_sub = self.transformer.proprio_proj(o_t[prop_indices]) + self.transformer.proprio_embed
            tok_h_sub = self.transformer.history_proj(h_t[prop_indices]) + self.transformer.history_embed

            def denoise_fn(y, r_, t_):
                return self.transformer.denoise_only(y, t_, r_, tok_p_sub, tok_h_sub)

            with sdpa_kernel(SDPBackend.MATH):
                _, du_dt_sub = func_jvp(
                    denoise_fn,
                    (y_sub, r_sub, t_sub),
                    (v_sub, torch.zeros_like(r_sub), torch.ones_like(t_sub)),
                )
            du_dt[prop_indices] = du_dt_sub

        # MeanFlow target: u_target = v_flat - (t - r) * du_dt
        t_minus_r = (t - r).unsqueeze(-1).expand(B, T, K, D).flatten(start_dim=2)
        u_target = v_flat - t_minus_r * du_dt

        # 3. Contrastive: second corruption with different eps
        eps2 = torch.randn(B, T, K, D, device=device)
        y_t2 = (1.0 - t).unsqueeze(-1) * y_clean + t.unsqueeze(-1) * eps2
        _, _, _, features2 = self.transformer(o_t, h_t, y_t2.flatten(start_dim=2), t, r)

        a_total = a_base + a_cond

        self._cached = {
            "u": u,
            "u_target": u_target,
            "features1": features1,
            "features2": features2,
            "a_base": a_base,
            "a_cond": a_cond,
            "t": t,
        }
        return a_total

    def forward(self):
        raise NotImplementedError

    def _sample_episode_corruption(self, env_ids, device):
        n = env_ids.shape[0]
        T, K = self.num_future_frames, self.num_keypoints
        t_level = torch.empty(n, device=device).uniform_(*self.rollout_t_range)
        p_clean = torch.empty(n, device=device).uniform_(*self.rollout_p_clean_range)
        kp_noisy = torch.rand(n, 1, K, device=device) > p_clean[:, None, None]
        t_kp = kp_noisy.float() * t_level[:, None, None]
        t_combined = t_kp.expand(n, T, K).contiguous()
        t_combined = torch.where(t_combined > 0.75, torch.ones_like(t_combined), t_combined)
        self._ep_t_combined[env_ids] = t_combined

    def act(self, observations, *args, **kwargs):
        hp_t, o_t, y_flat = self._split_obs(observations)
        h_t = self.history_encoder(hp_t)
        y_clean = self._reshape_conditions(y_flat)

        B = observations.shape[0]
        T, K = self.num_future_frames, self.num_keypoints
        if self._ep_t_combined is None or self._ep_t_combined.shape[0] != B:
            with torch.inference_mode(False):
                self._ep_t_combined = torch.zeros(B, T, K, device=observations.device)
                self._sample_episode_corruption(torch.arange(B, device=observations.device), observations.device)

        y_corrupted, tau = self.corruptor.corrupt_rollout(y_clean, self._ep_t_combined)
        y_corrupted_flat = y_corrupted.flatten(start_dim=2)
        r = torch.zeros_like(self._ep_t_combined)
        a_base, a_cond, _, _ = self.transformer(o_t, h_t, y_corrupted_flat, self._ep_t_combined, r)
        action_mean = a_base + a_cond

        std = self.std.expand_as(action_mean)
        self.distribution = Normal(action_mean, std)
        return self.distribution.sample()

    def evaluate(self, teacher_observations, *args, **kwargs):
        with torch.no_grad():
            actions = self.teacher.act_inference(teacher_observations, *args, **kwargs)
        return actions

    def act_inference(self, observations, *args, **kwargs):
        hp_t, o_t, y_flat = self._split_obs(observations)
        h_t = self.history_encoder(hp_t)
        y_clean = self._reshape_conditions(y_flat)

        if self._training_mode:
            return self._forward_corrupted_mf(o_t, h_t, y_clean)
        else:
            return self._forward_clean(o_t, h_t, y_clean)

    def extra_loss(self, **kwargs):
        if not self._cached:
            return {}, {}

        c = self._cached
        loss_dict = {}
        log_dict = {}

        # -- L_meanflow: MeanFlow self-consistency --
        l_mf = F.mse_loss(c["u"], c["u_target"].detach())
        loss_dict["codit_mf"] = l_mf * self.lambda_mf
        log_dict["codit_mf"] = l_mf.item()

        # -- L_contrast: contrastive feature regularization --
        h1 = c["features1"].flatten(1, 2)  # [B, T*d_model]
        h2 = c["features2"].flatten(1, 2)  # [B, T*d_model]

        # Attract: same y_clean, different eps → pull together
        l_attract = ((h1 - h2) ** 2).sum(dim=-1).mean()

        # Repel: maximize feature variance across batch
        l_repel = -h1.var(dim=0).mean()

        l_contrast = l_attract + l_repel
        loss_dict["codit_contrast"] = l_contrast * self.lambda_contrast
        log_dict["codit_contrast"] = l_contrast.item()
        log_dict["codit_attract"] = l_attract.item()
        log_dict["codit_repel"] = l_repel.item()

        # -- Logging --
        with torch.no_grad():
            log_dict["codit_a_base_norm"] = c["a_base"].norm(dim=-1).mean().item()
            log_dict["codit_a_cond_norm"] = c["a_cond"].norm(dim=-1).mean().item()
            log_dict["codit_t_mean"] = c["t"].mean().item()

            # Feature variance at high t
            t_flat = c["t"].mean(dim=(1, 2))  # per-sample mean t
            high_t_mask = t_flat > 0.8
            if high_t_mask.any():
                log_dict["codit_feat_var_high_t"] = h1[high_t_mask].var(dim=0).mean().item()

            # Positive pair distance
            log_dict["codit_pair_dist"] = (h1 - h2).norm(dim=-1).mean().item()

        self._cached = {}
        return dict(loss_dict), dict(log_dict)

    def pre_train(self):
        self._training_mode = True

    def after_train(self):
        self._training_mode = False
        self._cached = {}

    def reset(self, dones=None, hidden_states=None):
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
