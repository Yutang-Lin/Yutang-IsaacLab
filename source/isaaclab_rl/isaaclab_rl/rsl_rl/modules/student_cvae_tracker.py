# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
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

from isaaclab_rl.rsl_rl.networks.cvae_tracker_networks import (
    CVAEPrior,
    CVAEPosterior,
    CVAEActionDecoder,
    _build_mlp,
)


class StudentCVAETracker(nn.Module):
    """CVAE-based student tracking policy for distillation.

    Uses a Conditional VAE with:
    - MLP history encoder on env-provided history-stacked proprio obs (via wrap_with_history)
    - Prior network p(z_t | h_t, y_t) for inference-time latent prediction
    - Low-rank posterior correction from motion_keybody observations
    - Transformer action decoder with o_t, y_t, z_t as separate tokens

    No recurrence — the env provides history-stacked observations and the prior
    does not condition on previous z.

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
                "StudentCVAETracker.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.loaded_teacher = False

        if isinstance(num_student_obs, tuple):
            num_student_obs, _ = num_student_obs

        # -- Resolve obs meta: split into history proprio, current proprio, condition, and keybody indices --
        # Unwrap nesting: runner passes {actor_obs_meta: {...}, critic_obs_meta: {...}}
        if "actor_obs_meta" in student_obs_meta:
            obs_meta = student_obs_meta["actor_obs_meta"]
        else:
            obs_meta = student_obs_meta
        self.student_obs_meta = obs_meta
        history_proprio_ids, current_proprio_ids, condition_ids, keybody_ids = (
            self._resolve_obs_meta(num_student_obs, obs_meta)
        )
        # Register as buffers so they move with .to(device)
        self.register_buffer("history_proprio_ids", history_proprio_ids)
        self.register_buffer("current_proprio_ids", current_proprio_ids)
        self.register_buffer("condition_ids", condition_ids)
        self.register_buffer("keybody_ids", keybody_ids)

        history_proprio_dim = history_proprio_ids.shape[0]
        current_proprio_dim = current_proprio_ids.shape[0]
        cond_dim = condition_ids.shape[0]
        keybody_dim = keybody_ids.shape[0]

        # -- Extract CVAE hyperparams from student_policy_cfg --
        cfg = dict(student_policy_cfg)  # copy to avoid mutating
        cfg.pop("class_name", None)
        activation_name = cfg.pop("activation", "elu")
        activation = resolve_nn_activation(activation_name)

        latent_dim = cfg.pop("latent_dim", 32)
        corr_rank = cfg.pop("corr_rank", 8)
        history_hidden_dims = cfg.pop("history_hidden_dims", [512, 256])
        prior_hidden_dims = cfg.pop("prior_hidden_dims", [256, 128])
        posterior_hidden_dims = cfg.pop("posterior_hidden_dims", [256, 128])
        decoder_hidden_dims = cfg.pop("decoder_hidden_dims", [256])
        tf_d_model = cfg.pop("tf_d_model", 256)
        tf_num_heads = cfg.pop("tf_num_heads", 4)
        tf_num_layers = cfg.pop("tf_num_layers", 2)
        tf_hidden_dim = cfg.pop("tf_hidden_dim", 512)
        tf_dropout = cfg.pop("tf_dropout", 0.0)
        tf_activation_name = cfg.pop("tf_activation", "gelu")
        if tf_activation_name == "gelu":
            tf_activation = nn.GELU(approximate="tanh")
        else:
            tf_activation = resolve_nn_activation(tf_activation_name)
        self.corr_kl_coef = cfg.pop("corr_kl_coef", 1e-3)
        self.latent_kl_coef = cfg.pop("latent_kl_coef", 1e-3)
        # Temporal smoothness coefficients (first and second order)
        self.h_smooth_1st_coef = cfg.pop("h_smooth_1st_coef", 2e-4)
        self.h_smooth_2nd_coef = cfg.pop("h_smooth_2nd_coef", 2e-4)
        self.z_smooth_1st_coef = cfg.pop("z_smooth_1st_coef", 2e-4)
        self.z_smooth_2nd_coef = cfg.pop("z_smooth_2nd_coef", 2e-4)
        self.c_smooth_1st_coef = cfg.pop("c_smooth_1st_coef", 2e-4)
        self.c_smooth_2nd_coef = cfg.pop("c_smooth_2nd_coef", 2e-4)
        # Prior output regularization (L2 penalty to keep mu/logvar small)
        self.prior_mu_reg_coef = cfg.pop("prior_mu_reg_coef", 1e-5)
        self.prior_logvar_reg_coef = cfg.pop("prior_logvar_reg_coef", 1e-5)

        # Episodic binary keypoint masking for rollout
        # Keypoint order in 6-keypoint obs: [lw, rw, la, ra, head, pelvis] × T frames × D dims
        self.rollout_mask_num_keypoints = cfg.pop("rollout_mask_num_keypoints", 0)  # 0 = disabled
        self.rollout_mask_dims_per_keypoint = cfg.pop("rollout_mask_dims_per_keypoint", 9)
        self.rollout_mask_num_frames = cfg.pop("rollout_mask_num_frames", 5)
        self.rollout_mask_p_clean_range = tuple(cfg.pop("rollout_mask_p_clean_range", [0.2, 1.0]))

        self.latent_dim = latent_dim
        self.corr_rank = corr_rank
        self.num_actions = num_actions

        if cfg:
            print(f"StudentCVAETracker: unused config keys: {list(cfg.keys())}")

        # -- Build sub-networks --
        # History encoder: MLP on history-stacked proprio (env provides history via wrap_with_history)
        history_output_dim = history_hidden_dims[-1] if history_hidden_dims else history_proprio_dim
        self.history_encoder = _build_mlp(history_proprio_dim, history_hidden_dims, history_output_dim, activation)

        self.prior = CVAEPrior(
            h_dim=history_output_dim,
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            hidden_dims=prior_hidden_dims,
            activation=activation,
        )

        self.posterior = CVAEPosterior(
            keybody_dim=keybody_dim,
            corr_rank=corr_rank,
            latent_dim=latent_dim,
            hidden_dims=posterior_hidden_dims,
            activation=activation,
        )

        self.action_decoder = CVAEActionDecoder(
            proprio_dim=current_proprio_dim,
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            num_actions=num_actions,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            tf_d_model=tf_d_model,
            tf_num_heads=tf_num_heads,
            tf_num_layers=tf_num_layers,
            tf_hidden_dim=tf_hidden_dim,
            tf_dropout=tf_dropout,
            tf_activation=tf_activation,
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

        print(f"StudentCVAETracker obs split: history_proprio={history_proprio_dim}, "
              f"current_proprio={current_proprio_dim}, cond={cond_dim}, keybody={keybody_dim}")
        print(f"StudentCVAETracker networks:")
        print(f"  History Encoder: {self.history_encoder}")
        print(f"  Prior: {self.prior}")
        print(f"  Posterior: {self.posterior}")
        print(f"  Action Decoder: {self.action_decoder}")
        print(f"  Teacher: {self.teacher}")

        # -- Action noise --
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

        # -- State --
        self._save_dict = {}
        self._save_log_dict = {}
        self.compute_latent_loss = False

        # -- Episodic keypoint masking state (lazily initialized) --
        self._ep_kp_mask: torch.Tensor | None = None  # [N, K] bool, True=visible

        # -- Temporal smoothness buffers (populated lazily in act_inference) --
        self._prev_h: torch.Tensor | None = None
        self._prev_prev_h: torch.Tensor | None = None
        self._prev_z: torch.Tensor | None = None
        self._prev_prev_z: torch.Tensor | None = None
        self._prev_c: torch.Tensor | None = None
        self._prev_prev_c: torch.Tensor | None = None

    @property
    def student(self):
        """Return self so Distillation.broadcast_parameters() syncs trainable weights."""
        return self

    def _resolve_obs_meta(self, num_obs: int, obs_meta: dict):
        """Resolve observation metadata into history proprio, current proprio, condition, and keybody index tensors.

        Returns:
            history_proprio_ids: indices of history-stacked proprio obs (input to history encoder)
            current_proprio_ids: indices of current-frame proprio obs (o_t token for action decoder)
            condition_ids: indices of condition obs (y_t)
            keybody_ids: indices of posterior condition obs (r_t)
        """
        all_obs = torch.arange(num_obs)
        history_mask = torch.ones(num_obs, dtype=torch.bool)
        condition_obs = []
        keybody_obs = []
        current_proprio_obs = []

        if "conditions" in obs_meta:
            for seg in obs_meta["conditions"]:
                condition_obs.append(all_obs[seg["start"]:seg["end"]].clone())
                history_mask[seg["start"]:seg["end"]] = False

        if "posterior_conditions" in obs_meta:
            for seg in obs_meta["posterior_conditions"]:
                keybody_obs.append(all_obs[seg["start"]:seg["end"]].clone())
                history_mask[seg["start"]:seg["end"]] = False

        if "current_proprio" in obs_meta:
            for seg in obs_meta["current_proprio"]:
                current_proprio_obs.append(all_obs[seg["start"]:seg["end"]].clone())
                history_mask[seg["start"]:seg["end"]] = False

        history_proprio_ids = all_obs[history_mask].clone().contiguous()
        current_proprio_ids = torch.cat(current_proprio_obs).contiguous() if current_proprio_obs else torch.tensor([], dtype=torch.long)
        condition_ids = torch.cat(condition_obs).contiguous() if condition_obs else torch.tensor([], dtype=torch.long)
        keybody_ids = torch.cat(keybody_obs).contiguous() if keybody_obs else torch.tensor([], dtype=torch.long)

        return history_proprio_ids, current_proprio_ids, condition_ids, keybody_ids

    def _split_obs(self, obs: torch.Tensor):
        """Split observations into history proprio, current proprio, condition, and keybody components."""
        hp_t = obs[..., self.history_proprio_ids].contiguous()
        o_t = obs[..., self.current_proprio_ids].contiguous()
        y_t = obs[..., self.condition_ids].contiguous()
        r_t = obs[..., self.keybody_ids].contiguous()
        return hp_t, o_t, y_t, r_t

    _LOGVAR_CLAMP = (-20.0, 10.0)  # prevents exp() overflow in sampling and KL

    def _encode_history(self, hp_t: torch.Tensor) -> torch.Tensor:
        """Encode history-stacked proprio and L2-normalize to unit sphere."""
        h_t = self.history_encoder(hp_t)
        h_t = torch.nn.functional.normalize(h_t, dim=-1)
        return h_t

    def _compute_prior(self, hp_t: torch.Tensor, y_t: torch.Tensor):
        """Encode history-stacked proprio, compute prior distribution."""
        h_t = self._encode_history(hp_t)
        mu_prior, logvar_prior = self.prior(h_t, y_t)
        logvar_prior = logvar_prior.clamp(*self._LOGVAR_CLAMP)
        return mu_prior, logvar_prior

    @staticmethod
    def _sample_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    @staticmethod
    def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor,
                       mu_target: torch.Tensor | None = None,
                       logvar_target: torch.Tensor | None = None) -> torch.Tensor:
        """KL(N(mu, sigma) || N(mu_target, sigma_target)).

        If targets are None, computes KL against N(0, I).
        """
        if mu_target is None and logvar_target is None:
            return 0.5 * (logvar.exp() + mu.pow(2) - 1 - logvar).mean()
        else:
            logvar_t = logvar_target
            mu_t = mu_target
            kl = 0.5 * (
                (logvar_t - logvar)
                + (logvar.exp() + (mu - mu_t).pow(2)) / logvar_t.exp()
                - 1
            )
            return kl.mean()

    def forward(self):
        raise NotImplementedError

    def _sample_episode_kp_mask(self, env_ids: torch.Tensor, device: torch.device):
        """Sample per-env binary keypoint mask for rollout. True=visible, False=masked."""
        n = env_ids.shape[0]
        K = self.rollout_mask_num_keypoints
        p_clean = torch.empty(n, device=device).uniform_(*self.rollout_mask_p_clean_range)
        mask = torch.rand(n, K, device=device) < p_clean[:, None]  # [n, K]
        self._ep_kp_mask[env_ids] = mask

    def _apply_kp_mask(self, y_t: torch.Tensor) -> torch.Tensor:
        """Apply per-env binary keypoint mask to condition vector.

        y_t layout: [B, T*K*D] where T=num_frames, K=num_keypoints, D=dims_per_keypoint.
        Masked keypoints are zeroed out.
        """
        K = self.rollout_mask_num_keypoints
        D = self.rollout_mask_dims_per_keypoint
        T = self.rollout_mask_num_frames
        B = y_t.shape[0]
        # Reshape to [B, T, K, D], apply mask, flatten back
        y = y_t.view(B, T, K, D)
        mask = self._ep_kp_mask[:B, None, :, None].expand_as(y)  # [B, 1, K, 1] → [B, T, K, D]
        y = y * mask.float()
        return y.flatten(start_dim=1)

    def act(self, observations, *args, **kwargs):
        """Rollout: use prior only, sample with noise. Apply episodic keypoint masking."""
        hp_t, o_t, y_t, r_t = self._split_obs(observations)

        # Episodic binary keypoint masking (if enabled)
        if self.rollout_mask_num_keypoints > 0:
            B = observations.shape[0]
            K = self.rollout_mask_num_keypoints
            if self._ep_kp_mask is None or self._ep_kp_mask.shape[0] != B:
                with torch.inference_mode(False):
                    self._ep_kp_mask = torch.ones(B, K, dtype=torch.bool, device=observations.device)
                    self._sample_episode_kp_mask(torch.arange(B, device=observations.device), observations.device)
            y_t = self._apply_kp_mask(y_t)

        mu_prior, logvar_prior = self._compute_prior(hp_t, y_t)
        z_t = self._sample_gaussian(mu_prior, logvar_prior)

        action_mean = self.action_decoder(o_t, y_t, z_t)

        std = self.std.expand_as(action_mean)
        self.distribution = Normal(action_mean, std)
        return self.distribution.sample()

    def evaluate(self, teacher_observations, *args, **kwargs):
        """Get teacher actions (frozen)."""
        with torch.no_grad():
            actions = self.teacher.act_inference(teacher_observations, *args, **kwargs)
        return actions

    def _smoothness_loss(self, x: torch.Tensor, prev: torch.Tensor | None, prev_prev: torch.Tensor | None):
        """Compute first and second order temporal smoothness losses.

        Returns (first_order, second_order) as scalars, or (None, None) if not enough history.
        """
        first = None
        second = None
        if prev is not None:
            first = (x - prev).pow(2).mean()
            if prev_prev is not None:
                second = (x - 2 * prev + prev_prev).pow(2).mean()
        return first, second

    def _random_kp_mask(self, y_t: torch.Tensor) -> torch.Tensor:
        """Apply fresh random binary keypoint mask for training.

        Each call samples a new mask — different from the episode-fixed rollout mask.
        """
        K = self.rollout_mask_num_keypoints
        D = self.rollout_mask_dims_per_keypoint
        T = self.rollout_mask_num_frames
        B = y_t.shape[0]
        p_clean = torch.empty(B, device=y_t.device).uniform_(*self.rollout_mask_p_clean_range)
        mask = (torch.rand(B, K, device=y_t.device) < p_clean[:, None])  # [B, K] bool
        y = y_t.view(B, T, K, D)
        y = y * mask[:, None, :, None].float()
        return y.flatten(start_dim=1)

    def act_inference(self, observations, *args, **kwargs):
        """Training update or deployment inference.

        During training (compute_latent_loss=True): uses posterior correction and stores KL losses.
        During inference: uses prior mean only.
        """
        hp_t, o_t, y_t, r_t = self._split_obs(observations)

        # Apply random keypoint masking during training (fresh each call)
        if self.compute_latent_loss and self.rollout_mask_num_keypoints > 0:
            y_t = self._random_kp_mask(y_t)

        h_t = self._encode_history(hp_t)
        mu_prior, logvar_prior = self.prior(h_t, y_t)
        logvar_prior = logvar_prior.clamp(*self._LOGVAR_CLAMP)

        if self.compute_latent_loss and r_t.shape[-1] > 0:
            # training: use posterior correction
            z_prior = self._sample_gaussian(mu_prior, logvar_prior)

            # posterior correction in low-rank space
            mu_raw, logvar_raw = self.posterior(r_t)
            logvar_raw = logvar_raw.clamp(*self._LOGVAR_CLAMP)
            c_t, c_raw = self.posterior.sample_and_lift(mu_raw, logvar_raw)
            z_t = z_prior + c_t

            # KL loss 1: correction regularization KL(q(c_raw|r_t) || N(0, I))
            corr_kl = self._kl_divergence(mu_raw, logvar_raw)

            # KL loss 2: KL(q(z_t) || p(z_prior))
            W = self.posterior.lift.weight  # [latent_dim, corr_rank]
            mu_zt = mu_prior + torch.nn.functional.linear(mu_raw, W)  # [batch, latent_dim]
            var_raw = logvar_raw.exp()  # [batch, corr_rank]
            W_var = torch.nn.functional.linear(var_raw, W.pow(2))  # [batch, latent_dim]
            logvar_zt = torch.log(logvar_prior.exp() + W_var + 1e-8)
            logvar_zt = logvar_zt.clamp(*self._LOGVAR_CLAMP)

            latent_kl = self._kl_divergence(mu_zt, logvar_zt, mu_prior, logvar_prior)

            self._save_dict["cvae_corr_kl"] = corr_kl * self.corr_kl_coef
            self._save_dict["cvae_latent_kl"] = latent_kl * self.latent_kl_coef
            self._save_log_dict["cvae_corr_kl"] = corr_kl.item()
            self._save_log_dict["cvae_latent_kl"] = latent_kl.item()

            # -- Prior output regularization (keeps mu/logvar from exploding) --
            if self.prior_mu_reg_coef > 0:
                mu_reg = mu_prior.pow(2).mean()
                self._save_dict["prior_mu_reg"] = mu_reg * self.prior_mu_reg_coef
                self._save_log_dict["prior_mu_reg"] = mu_reg.item()
            if self.prior_logvar_reg_coef > 0:
                logvar_reg = logvar_prior.pow(2).mean()
                self._save_dict["prior_logvar_reg"] = logvar_reg * self.prior_logvar_reg_coef
                self._save_log_dict["prior_logvar_reg"] = logvar_reg.item()

            # -- Temporal smoothness losses --
            # h_t smoothness
            if self.h_smooth_1st_coef > 0 or self.h_smooth_2nd_coef > 0:
                h1, h2 = self._smoothness_loss(h_t, self._prev_h, self._prev_prev_h)
                if h1 is not None and self.h_smooth_1st_coef > 0:
                    self._save_dict["smooth_h_1st"] = h1 * self.h_smooth_1st_coef
                    self._save_log_dict["smooth_h_1st"] = h1.item()
                if h2 is not None and self.h_smooth_2nd_coef > 0:
                    self._save_dict["smooth_h_2nd"] = h2 * self.h_smooth_2nd_coef
                    self._save_log_dict["smooth_h_2nd"] = h2.item()

            # z_prior smoothness
            if self.z_smooth_1st_coef > 0 or self.z_smooth_2nd_coef > 0:
                z1, z2 = self._smoothness_loss(z_prior, self._prev_z, self._prev_prev_z)
                if z1 is not None and self.z_smooth_1st_coef > 0:
                    self._save_dict["smooth_z_1st"] = z1 * self.z_smooth_1st_coef
                    self._save_log_dict["smooth_z_1st"] = z1.item()
                if z2 is not None and self.z_smooth_2nd_coef > 0:
                    self._save_dict["smooth_z_2nd"] = z2 * self.z_smooth_2nd_coef
                    self._save_log_dict["smooth_z_2nd"] = z2.item()

            # c_t smoothness
            if self.c_smooth_1st_coef > 0 or self.c_smooth_2nd_coef > 0:
                c1, c2 = self._smoothness_loss(c_t, self._prev_c, self._prev_prev_c)
                if c1 is not None and self.c_smooth_1st_coef > 0:
                    self._save_dict["smooth_c_1st"] = c1 * self.c_smooth_1st_coef
                    self._save_log_dict["smooth_c_1st"] = c1.item()
                if c2 is not None and self.c_smooth_2nd_coef > 0:
                    self._save_dict["smooth_c_2nd"] = c2 * self.c_smooth_2nd_coef
                    self._save_log_dict["smooth_c_2nd"] = c2.item()

            # Shift temporal buffers
            self._prev_prev_h = self._prev_h
            self._prev_h = h_t.detach()
            self._prev_prev_z = self._prev_z
            self._prev_z = z_prior.detach()
            self._prev_prev_c = self._prev_c
            self._prev_c = c_t.detach()
        else:
            # inference: use prior mean, no sampling
            z_t = mu_prior

        action_mean = self.action_decoder(o_t, y_t, z_t)
        return action_mean

    def extra_loss(self, **kwargs):
        loss_dict = dict(self._save_dict)
        log_dict = dict(self._save_log_dict)
        self._save_dict.clear()
        self._save_log_dict.clear()
        return loss_dict, log_dict

    def pre_train(self):
        self.compute_latent_loss = True
        # Reset temporal buffers at start of each training epoch
        self._prev_h = self._prev_prev_h = None
        self._prev_z = self._prev_prev_z = None
        self._prev_c = self._prev_prev_c = None

    def after_train(self):
        self.compute_latent_loss = False

    def reset(self, dones=None, hidden_states=None):
        """Reset temporal smoothness buffers and resample keypoint masks for done environments."""
        if dones is not None and dones.any():
            mask = dones.bool()
            for buf_name in ("_prev_h", "_prev_prev_h", "_prev_z", "_prev_prev_z", "_prev_c", "_prev_prev_c"):
                buf = getattr(self, buf_name)
                if buf is not None:
                    buf[mask] = 0.0
            # Resample episodic keypoint masks
            if self.rollout_mask_num_keypoints > 0 and self._ep_kp_mask is not None:
                env_ids = mask.flatten().nonzero(as_tuple=False).squeeze(-1)
                if env_ids.numel() > 0:
                    with torch.inference_mode(False):
                        self._sample_episode_kp_mask(env_ids, self._ep_kp_mask.device)

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
