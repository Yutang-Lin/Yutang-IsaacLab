# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CVAE-BFM: Foundation model CVAE with variable-interval sparse frame commands.

Extends StudentCVAETracker with:
  - 10-frame variable-interval future horizon (per-env episodic interval)
  - Per-frame delta_t conditioning (time offset to current)
  - Frame-level pad masking in transformer decoder (masked frames excluded from attention)
  - Binary keypoint masking (inherited from CVAE)
  - Standard CVAE training (prior, posterior correction, KL)
  - Teacher uses 5-frame fixed 0.02s interval
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

from isaaclab_rl.rsl_rl.networks.cvae_tracker_networks import _build_mlp, CVAEPrior, CVAEPosterior
from isaaclab_rl.rsl_rl.networks.cvae_bfm_networks import CVAEBFMDecoder


class StudentCVAEBFMTracker(nn.Module):
    """CVAE-BFM student policy for distillation.

    Foundation model CVAE with:
    - Variable-interval 10-frame future conditioning with delta_t
    - Per-frame pad masking in decoder attention
    - Episodic keypoint masking (binary)
    - Standard CVAE training (prior + posterior correction + KL)
    """

    is_recurrent = False

    _LOGVAR_CLAMP = (-10.0, 2.0)

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
            print("StudentCVAEBFMTracker.__init__ got unexpected args: " + str(list(kwargs.keys())))
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
        history_proprio_ids, current_proprio_ids, condition_ids, keybody_ids = (
            self._resolve_obs_meta(num_student_obs, obs_meta)
        )
        self.register_buffer("history_proprio_ids", history_proprio_ids)
        self.register_buffer("current_proprio_ids", current_proprio_ids)
        self.register_buffer("condition_ids", condition_ids)
        self.register_buffer("keybody_ids", keybody_ids)

        history_proprio_dim = history_proprio_ids.shape[0]
        current_proprio_dim = current_proprio_ids.shape[0]
        cond_dim = condition_ids.shape[0]
        keybody_dim = keybody_ids.shape[0]

        # -- Extract hyperparams --
        cfg = dict(student_policy_cfg)
        cfg.pop("class_name", None)
        activation_name = cfg.pop("activation", "elu")
        activation = resolve_nn_activation(activation_name)

        latent_dim = cfg.pop("latent_dim", 32)
        corr_rank = cfg.pop("corr_rank", 8)
        history_hidden_dims = cfg.pop("history_hidden_dims", [512, 256])
        prior_hidden_dims = cfg.pop("prior_hidden_dims", [256, 128])
        posterior_hidden_dims = cfg.pop("posterior_hidden_dims", [256, 128])

        # BFM frame parameters
        self.num_keypoints = cfg.pop("num_keypoints", 6)
        self.dims_per_keypoint = cfg.pop("dims_per_keypoint", 9)
        self.num_frames = cfg.pop("num_frames", 10)
        self.step_dt = cfg.pop("step_dt", 0.02)
        # Frame dim in obs: K*D + 1 (delta_t) = 55 per frame
        self.frame_dim = self.num_keypoints * self.dims_per_keypoint + 1
        expected_cond_dim = self.num_frames * self.frame_dim
        assert cond_dim == expected_cond_dim, (
            f"Condition dim mismatch: got {cond_dim}, expected F*frame_dim = "
            f"{self.num_frames}*{self.frame_dim} = {expected_cond_dim}"
        )

        # Episodic frame interval range (in steps)
        self.min_frame_interval = cfg.pop("min_frame_interval", 1)
        self.max_frame_interval = cfg.pop("max_frame_interval", 5)
        # Episodic frame mask probability
        self.frame_p_active_range = tuple(cfg.pop("frame_p_active_range", [0.5, 1.0]))
        # Episodic keypoint mask
        self.rollout_mask_num_keypoints = cfg.pop("rollout_mask_num_keypoints", 6)
        self.rollout_mask_p_clean_range = tuple(cfg.pop("rollout_mask_p_clean_range", [0.2, 1.0]))

        # Transformer decoder config
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

        # KL and smoothness coefficients
        self.corr_kl_coef = cfg.pop("corr_kl_coef", 1e-3)
        self.latent_kl_coef = cfg.pop("latent_kl_coef", 1e-3)
        self.h_smooth_1st_coef = cfg.pop("h_smooth_1st_coef", 2e-4)
        self.h_smooth_2nd_coef = cfg.pop("h_smooth_2nd_coef", 2e-4)
        self.z_smooth_1st_coef = cfg.pop("z_smooth_1st_coef", 2e-4)
        self.z_smooth_2nd_coef = cfg.pop("z_smooth_2nd_coef", 2e-4)
        self.c_smooth_1st_coef = cfg.pop("c_smooth_1st_coef", 2e-4)
        self.c_smooth_2nd_coef = cfg.pop("c_smooth_2nd_coef", 2e-4)
        self.prior_mu_reg_coef = cfg.pop("prior_mu_reg_coef", 1e-5)
        self.prior_logvar_reg_coef = cfg.pop("prior_logvar_reg_coef", 1e-5)

        self.latent_dim = latent_dim
        self.corr_rank = corr_rank
        self.num_actions = num_actions

        if cfg:
            print(f"StudentCVAEBFMTracker: unused config keys: {list(cfg.keys())}")

        # -- Build sub-networks --
        history_output_dim = history_hidden_dims[-1] if history_hidden_dims else history_proprio_dim
        self.history_encoder = _build_mlp(history_proprio_dim, history_hidden_dims, history_output_dim, activation)

        # Prior: input is h_t + y_flat (masked frames zeroed, includes delta_t)
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

        # BFM decoder: per-frame tokens with pad masking
        self.action_decoder = CVAEBFMDecoder(
            proprio_dim=current_proprio_dim,
            latent_dim=latent_dim,
            num_keypoints=self.num_keypoints,
            dims_per_keypoint=self.dims_per_keypoint,
            max_frames=self.num_frames,
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

        print(f"StudentCVAEBFMTracker: history={history_proprio_dim}, proprio={current_proprio_dim}, "
              f"cond={cond_dim} ({self.num_frames}×{self.frame_dim}), keybody={keybody_dim}")

        # -- Action noise --
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

        # -- State --
        self.compute_latent_loss = False
        self._prev_h = None
        self._prev_prev_h = None
        self._prev_z = None
        self._prev_prev_z = None
        self._prev_c = None
        self._prev_prev_c = None
        # Episodic state (lazily initialized)
        self._ep_frame_mask: torch.Tensor | None = None  # [N, F] bool
        self._ep_kp_mask: torch.Tensor | None = None  # [N, K] bool
        self._ep_frame_offsets: torch.Tensor | None = None  # [N, F] seconds
        self._env_ref = None  # reference to env for setting offsets

    @property
    def student(self):
        return self

    def set_env(self, env):
        """Called by runner to provide env reference for setting BFM offsets."""
        self._env_ref = env

    def _resolve_obs_meta(self, num_obs, obs_meta):
        all_obs = torch.arange(num_obs)
        history_mask = torch.ones(num_obs, dtype=torch.bool)
        condition_obs, keybody_obs, current_proprio_obs = [], [], []

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

        history_ids = all_obs[history_mask].clone().contiguous()
        current_ids = torch.cat(current_proprio_obs).contiguous() if current_proprio_obs else torch.tensor([], dtype=torch.long)
        cond_ids = torch.cat(condition_obs).contiguous() if condition_obs else torch.tensor([], dtype=torch.long)
        kb_ids = torch.cat(keybody_obs).contiguous() if keybody_obs else torch.tensor([], dtype=torch.long)
        return history_ids, current_ids, cond_ids, kb_ids

    def _split_obs(self, obs):
        hp_t = obs[..., self.history_proprio_ids].contiguous()
        o_t = obs[..., self.current_proprio_ids].contiguous()
        y_t = obs[..., self.condition_ids].contiguous()
        r_t = obs[..., self.keybody_ids].contiguous()
        return hp_t, o_t, y_t, r_t

    def _parse_condition(self, y_flat):
        """Parse flat condition into frames + delta_t.

        y_flat: [B, F * (K*D + 1)]
        Returns:
            frames: [B, F, K*D] keypoint data per frame
            delta_t: [B, F] time offset per frame
        """
        B = y_flat.shape[0]
        y = y_flat.view(B, self.num_frames, self.frame_dim)
        frames = y[:, :, :-1]  # [B, F, K*D]
        delta_t = y[:, :, -1]  # [B, F]
        return frames, delta_t

    def _encode_history(self, hp_t):
        h_t = self.history_encoder(hp_t)
        return F.normalize(h_t, dim=-1)

    def _sample_gaussian(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def _kl_divergence(self, mu, logvar, mu_t=None, logvar_t=None):
        if mu_t is None:
            return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean()
        else:
            kl = 0.5 * ((logvar_t - logvar) + (logvar.exp() + (mu - mu_t).pow(2)) / logvar_t.exp() - 1)
            return kl.mean()

    def _smoothness_loss(self, x, prev, prev_prev):
        first, second = None, None
        if prev is not None:
            first = (x - prev).pow(2).mean()
            if prev_prev is not None:
                second = (x - 2 * prev + prev_prev).pow(2).mean()
        return first, second

    # -- Episodic sampling + sliding window --

    def _sample_initial_offsets(self, env_ids, device):
        """Sample initial sorted future time offsets for given envs.

        Also samples episodic keypoint mask (held for entire episode).
        """
        n = env_ids.shape[0]
        F = self.num_frames
        K = self.rollout_mask_num_keypoints

        # Sample per-env interval, build sorted offsets
        intervals = torch.empty(n, device=device).uniform_(
            self.min_frame_interval * self.step_dt,
            self.max_frame_interval * self.step_dt)
        step_indices = torch.arange(1, F + 1, device=device, dtype=torch.float32)
        offsets = step_indices[None, :] * intervals[:, None]  # [n, F], starts at interval, not 0
        self._ep_frame_offsets[env_ids] = offsets

        # Frame mask: episodic per-frame active mask
        p_active = torch.empty(n, device=device).uniform_(*self.frame_p_active_range)
        frame_mask = torch.rand(n, F, device=device) < p_active[:, None]
        frame_mask[:, 0] = True
        self._ep_frame_mask[env_ids] = frame_mask

        # Keypoint mask
        if K > 0:
            p_clean = torch.empty(n, device=device).uniform_(*self.rollout_mask_p_clean_range)
            kp_mask = torch.rand(n, K, device=device) < p_clean[:, None]
            self._ep_kp_mask[env_ids] = kp_mask

        self._push_offsets_to_env(env_ids)

    def _step_offsets(self):
        """Advance sliding window: decrement offsets, recycle consumed frames.

        Operates on full [N, F] tensor in parallel, no per-env loops.
        """
        offsets = self._ep_frame_offsets  # [N, F]
        mask = self._ep_frame_mask  # [N, F]

        # Decrement all offsets by step_dt
        offsets -= self.step_dt

        # Count consumed frames per env (offset <= 0)
        consumed = (offsets[:, 0] <= 0)  # [N] bool — only check leftmost (sorted)
        if not consumed.any():
            self._push_offsets_to_env()
            return

        # For envs with consumed frame(s): shift left, append new at end
        # Since offsets are sorted and step_dt is small, usually only 1 consumed per step
        # Handle multiple by checking how many are <= 0
        n_consumed = (offsets <= 0).long().sum(dim=1)  # [N]
        max_consumed = n_consumed.max().item()

        for shift in range(1, max_consumed + 1):
            shift_mask = n_consumed >= shift  # [N] envs that need at least this many shifts
            if not shift_mask.any():
                break
            # Shift offsets and masks left by 1 for these envs
            offsets[shift_mask, :-1] = offsets[shift_mask, 1:].clone()
            mask[shift_mask, :-1] = mask[shift_mask, 1:].clone()

            # Append new offset at last position: last_offset + random interval
            last_valid = offsets[shift_mask, -2]  # second-to-last after shift
            new_interval = torch.empty(shift_mask.sum(), device=offsets.device).uniform_(
                self.min_frame_interval * self.step_dt,
                self.max_frame_interval * self.step_dt)
            offsets[shift_mask, -1] = last_valid + new_interval

            # New frame mask for appended slot
            p_active = torch.empty(shift_mask.sum(), device=offsets.device).uniform_(
                *self.frame_p_active_range)
            mask[shift_mask, -1] = torch.rand(shift_mask.sum(), device=offsets.device) < p_active

        self._push_offsets_to_env()

    def _push_offsets_to_env(self, env_ids=None):
        """Push current offsets to env for motion sampling."""
        if self._env_ref is None:
            return
        env = self._env_ref
        F = self.num_frames
        if not hasattr(env, '_bfm_future_offsets') or env._bfm_future_offsets is None:
            env._bfm_future_offsets = torch.zeros(env.num_envs, F, device=self._ep_frame_offsets.device)
            env._bfm_delta_t = torch.zeros(env.num_envs, F, device=self._ep_frame_offsets.device)
        if env_ids is None:
            env._bfm_future_offsets[:] = self._ep_frame_offsets
            env._bfm_delta_t[:] = self._ep_frame_offsets
        else:
            env._bfm_future_offsets[env_ids] = self._ep_frame_offsets[env_ids]
            env._bfm_delta_t[env_ids] = self._ep_frame_offsets[env_ids]

    def _apply_masks(self, frames, frame_mask=None, kp_mask=None):
        """Apply frame and keypoint masks to condition frames.

        frames: [B, F, K*D]
        frame_mask: [B, F] bool (True=active)
        kp_mask: [B, K] bool (True=visible)
        Returns masked frames [B, F, K*D]
        """
        B, F, KD = frames.shape
        K = self.rollout_mask_num_keypoints
        D = self.dims_per_keypoint

        if kp_mask is not None and K > 0:
            f = frames.view(B, F, K, D)
            f = f * kp_mask[:, None, :, None].float()
            frames = f.flatten(start_dim=2)

        if frame_mask is not None:
            frames = frames * frame_mask.unsqueeze(-1).float()

        return frames

    # -- Forward paths --

    def forward(self):
        raise NotImplementedError

    def act(self, observations, *args, **kwargs):
        """Rollout: prior only, with sliding window offsets + frame/keypoint masking."""
        hp_t, o_t, y_flat, r_t = self._split_obs(observations)
        frames, delta_t = self._parse_condition(y_flat)
        B = observations.shape[0]

        # Lazy init episodic state
        F = self.num_frames
        K = self.rollout_mask_num_keypoints
        if self._ep_frame_mask is None or self._ep_frame_mask.shape[0] != B:
            with torch.inference_mode(False):
                self._ep_frame_mask = torch.ones(B, F, dtype=torch.bool, device=observations.device)
                self._ep_kp_mask = torch.ones(B, K, dtype=torch.bool, device=observations.device) if K > 0 else None
                self._ep_frame_offsets = torch.zeros(B, F, device=observations.device)
                self._sample_initial_offsets(torch.arange(B, device=observations.device), observations.device)
        else:
            # Advance sliding window: decrement offsets, recycle consumed frames
            with torch.inference_mode(False):
                self._step_offsets()

        # Apply episodic masks
        masked_frames = self._apply_masks(frames, self._ep_frame_mask, self._ep_kp_mask)
        y_masked_flat = torch.cat([masked_frames.flatten(1), delta_t], dim=-1)

        h_t = self._encode_history(hp_t)
        mu_prior, logvar_prior = self.prior(h_t, y_masked_flat)
        logvar_prior = logvar_prior.clamp(*self._LOGVAR_CLAMP)
        z_t = self._sample_gaussian(mu_prior, logvar_prior)

        action_mean = self.action_decoder(o_t, z_t, masked_frames, delta_t, self._ep_frame_mask[:B])

        std = self.std.expand_as(action_mean)
        self.distribution = Normal(action_mean, std)
        return self.distribution.sample()

    def evaluate(self, teacher_observations, *args, **kwargs):
        with torch.no_grad():
            return self.teacher.act_inference(teacher_observations, *args, **kwargs)

    def act_inference(self, observations, *args, **kwargs):
        """Training: posterior correction with random masks. Inference: prior mean."""
        hp_t, o_t, y_flat, r_t = self._split_obs(observations)
        frames, delta_t = self._parse_condition(y_flat)
        B = frames.shape[0]
        F = self.num_frames
        K = self.rollout_mask_num_keypoints

        # During training: apply fresh random masks each call
        if self.compute_latent_loss:
            # Random frame mask
            p_active = torch.empty(B, device=frames.device).uniform_(*self.frame_p_active_range)
            frame_mask = torch.rand(B, F, device=frames.device) < p_active[:, None]
            frame_mask[:, 0] = True
            # Random keypoint mask
            kp_mask = None
            if K > 0:
                p_clean = torch.empty(B, device=frames.device).uniform_(*self.rollout_mask_p_clean_range)
                kp_mask = torch.rand(B, K, device=frames.device) < p_clean[:, None]
        else:
            # Inference: all frames and keypoints active
            frame_mask = torch.ones(B, F, dtype=torch.bool, device=frames.device)
            kp_mask = None

        masked_frames = self._apply_masks(frames, frame_mask, kp_mask)
        y_masked_flat = torch.cat([masked_frames.flatten(1), delta_t], dim=-1)

        h_t = self._encode_history(hp_t)
        mu_prior, logvar_prior = self.prior(h_t, y_masked_flat)
        logvar_prior = logvar_prior.clamp(*self._LOGVAR_CLAMP)

        if self.compute_latent_loss and r_t.shape[-1] > 0:
            z_prior = self._sample_gaussian(mu_prior, logvar_prior)
            mu_raw, logvar_raw = self.posterior(r_t)
            logvar_raw = logvar_raw.clamp(*self._LOGVAR_CLAMP)
            c_t, c_raw = self.posterior.sample_and_lift(mu_raw, logvar_raw)
            z_t = z_prior + c_t

            corr_kl = self._kl_divergence(mu_raw, logvar_raw)
            latent_kl = self._kl_divergence(mu_prior, logvar_prior,
                                            mu_prior.detach(), logvar_prior.detach())

            # Smoothness
            h_s1, h_s2 = self._smoothness_loss(h_t, self._prev_h, self._prev_prev_h)
            z_s1, z_s2 = self._smoothness_loss(z_prior, self._prev_z, self._prev_prev_z)
            c_s1, c_s2 = self._smoothness_loss(c_t, self._prev_c, self._prev_prev_c)

            self._prev_prev_h = self._prev_h
            self._prev_h = h_t.detach()
            self._prev_prev_z = self._prev_z
            self._prev_z = z_prior.detach()
            self._prev_prev_c = self._prev_c
            self._prev_c = c_t.detach()

            self._save_dict = {
                "corr_kl": corr_kl, "latent_kl": latent_kl,
                "h_s1": h_s1, "h_s2": h_s2,
                "z_s1": z_s1, "z_s2": z_s2,
                "c_s1": c_s1, "c_s2": c_s2,
                "mu_prior": mu_prior, "logvar_prior": logvar_prior,
            }
        else:
            z_t = mu_prior

        action_mean = self.action_decoder(o_t, z_t, masked_frames, delta_t, frame_mask)
        return action_mean

    def extra_loss(self, **kwargs):
        if not self._save_dict:
            return {}, {}

        d = self._save_dict
        loss_dict = {}
        log_dict = {}

        loss_dict["corr_kl"] = d["corr_kl"] * self.corr_kl_coef
        log_dict["corr_kl"] = d["corr_kl"].item()
        loss_dict["latent_kl"] = d["latent_kl"] * self.latent_kl_coef
        log_dict["latent_kl"] = d["latent_kl"].item()

        for name, coef, key in [
            ("h_smooth_1st", self.h_smooth_1st_coef, "h_s1"),
            ("h_smooth_2nd", self.h_smooth_2nd_coef, "h_s2"),
            ("z_smooth_1st", self.z_smooth_1st_coef, "z_s1"),
            ("z_smooth_2nd", self.z_smooth_2nd_coef, "z_s2"),
            ("c_smooth_1st", self.c_smooth_1st_coef, "c_s1"),
            ("c_smooth_2nd", self.c_smooth_2nd_coef, "c_s2"),
        ]:
            val = d[key]
            if val is not None:
                loss_dict[name] = val * coef
                log_dict[name] = val.item()

        # Prior regularization
        mu_p = d["mu_prior"]
        lv_p = d["logvar_prior"]
        if self.prior_mu_reg_coef > 0:
            loss_dict["prior_mu_reg"] = mu_p.pow(2).mean() * self.prior_mu_reg_coef
        if self.prior_logvar_reg_coef > 0:
            loss_dict["prior_logvar_reg"] = lv_p.pow(2).mean() * self.prior_logvar_reg_coef

        self._save_dict = {}
        return dict(loss_dict), dict(log_dict)

    def pre_train(self):
        self.compute_latent_loss = True

    def after_train(self):
        self.compute_latent_loss = False
        self._save_dict = {}

    def reset(self, dones=None, hidden_states=None):
        if dones is not None and dones.any():
            mask = dones.bool().flatten()
            for buf_name in ("_prev_h", "_prev_prev_h", "_prev_z", "_prev_prev_z", "_prev_c", "_prev_prev_c"):
                buf = getattr(self, buf_name)
                if buf is not None:
                    buf[mask] = 0.0
            # Resample episodic offsets + masks for reset envs
            env_ids = mask.nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() > 0 and self._ep_frame_mask is not None:
                with torch.inference_mode(False):
                    self._sample_initial_offsets(env_ids, self._ep_frame_mask.device)

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    def load_state_dict(self, state_dict, strict=True):
        student_keys = [k for k in state_dict.keys() if not k.startswith("teacher")]
        student_params = {k: v for k, v in state_dict.items() if k in student_keys}
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
