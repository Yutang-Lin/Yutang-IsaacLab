# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CVAE-BFM v2: MaskedMimic-style CVAE with transformer prior + residual posterior.

Key differences from v1:
  - Prior is a transformer over [latent_query, h_enc, o_t_enc, frame_tokens]
  - Posterior outputs residual (delta_mu, delta_logvar) on top of prior
  - Decoder is MLP: concat(z_proj, o_t_enc, h_enc) → action
  - History on unit sphere with noise (like LFM-BFM)
  - Count-based masking for frames and keypoints
  - KL scheduling: starts low, ramps up
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

from isaaclab_rl.rsl_rl.networks.cvae_tracker_networks import _build_mlp
from isaaclab_rl.rsl_rl.networks.cvae_bfm_networks import (
    BFMFrameEncoder, CVAEBFMPriorV2, CVAEBFMPosteriorV2, CVAEBFMDecoderV2,
)


class StudentCVAEBFMTracker(nn.Module):
    """MaskedMimic-style CVAE-BFM student policy."""

    is_recurrent = False
    _LOGVAR_CLAMP = (-10.0, 2.0)

    def __init__(self, num_student_obs, num_teacher_obs, num_actions,
                 student_policy_cfg, teacher_policy_ckpt,
                 student_obs_meta, teacher_obs_meta,
                 init_noise_std=0.1, **kwargs):
        if kwargs:
            print(f"StudentCVAEBFMTracker: unexpected args: {list(kwargs.keys())}")
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

        # -- Config --
        cfg = dict(student_policy_cfg)
        cfg.pop("class_name", None)
        activation_name = cfg.pop("activation", "gelu")
        if activation_name == "gelu":
            activation = nn.GELU(approximate="tanh")
        else:
            activation = resolve_nn_activation(activation_name)

        latent_dim = cfg.pop("latent_dim", 64)
        prior_hidden_dims = cfg.pop("prior_hidden_dims", [512, 256])
        proprio_hidden_dims = cfg.pop("proprio_hidden_dims", [512, 256])
        decoder_hidden_dims = cfg.pop("decoder_hidden_dims", [512, 256, 128])
        # Discard legacy keys
        for k in ["corr_rank", "history_hidden_dims", "posterior_hidden_dims",
                   "corr_kl_coef", "latent_kl_coef", "prior_reg_coef",
                   "posterior_dropout", "recon_hidden_dims", "recon_coef"]:
            cfg.pop(k, None)

        self.num_keypoints = cfg.pop("num_keypoints", 6)
        self.dims_per_keypoint = cfg.pop("dims_per_keypoint", 9)
        self.num_frames = cfg.pop("num_frames", 10)
        self.step_dt = cfg.pop("step_dt", 0.02)
        self.frame_dim = self.num_keypoints * self.dims_per_keypoint + 1
        assert cond_dim == self.num_frames * self.frame_dim

        self.min_frame_delta = cfg.pop("min_frame_delta", 0.02)
        self.max_frame_delta = cfg.pop("max_frame_delta", 1.0)
        self.rollout_mask_num_keypoints = cfg.pop("rollout_mask_num_keypoints", 6)
        # Remove legacy probability-based masking args
        cfg.pop("frame_p_active_range", None)
        cfg.pop("rollout_mask_p_clean_range", None)

        self._mask_rng_seed = cfg.pop("mask_rng_seed", 42)
        self._mask_rng = None

        tf_d_model = cfg.pop("tf_d_model", 256)
        tf_num_heads = cfg.pop("tf_num_heads", 4)
        tf_hidden_dim = cfg.pop("tf_hidden_dim", 512)
        tf_dropout = cfg.pop("tf_dropout", 0.0)
        tf_activation_name = cfg.pop("tf_activation", "gelu")
        if tf_activation_name == "gelu":
            tf_activation = nn.GELU(approximate="tanh")
        else:
            tf_activation = resolve_nn_activation(tf_activation_name)

        prior_num_layers = cfg.pop("prior_num_layers", 2)
        cfg.pop("tf_num_layers", None)  # legacy
        cfg.pop("encoder_num_layers", None)
        cfg.pop("decoder_num_layers", None)
        cfg.pop("flow_num_layers", None)

        # KL scheduling
        self.kl_coef_start = cfg.pop("kl_coef_start", 1e-4)
        self.kl_coef_end = cfg.pop("kl_coef_end", 1e-2)
        self.kl_ramp_steps = cfg.pop("kl_ramp_steps", 5000)

        # History on sphere
        self.history_sigma = cfg.pop("history_sigma", 0.1)
        self.history_dropout_prob = cfg.pop("history_dropout_prob", 0.0)

        # Posterior dropout (zero c_t to match rollout)
        self.posterior_dropout = cfg.pop("posterior_dropout", 0.5)

        # Teacher forcing
        self.teacher_forcing_ratio = cfg.pop("teacher_forcing_ratio", 0.0)
        self.teacher_forcing_noise = cfg.pop("teacher_forcing_noise", 0.1)

        self.latent_dim = latent_dim
        self.num_actions = num_actions

        if cfg:
            print(f"StudentCVAEBFMTracker: unused keys: {list(cfg.keys())}")

        # -- Networks --
        # Encoders
        self.o_t_encoder = _build_mlp(current_proprio_dim, proprio_hidden_dims, tf_d_model, activation)
        self.history_encoder = _build_mlp(history_proprio_dim, prior_hidden_dims, tf_d_model, activation)
        self.frame_encoder = BFMFrameEncoder(
            self.num_keypoints, self.dims_per_keypoint, tf_d_model, tf_activation)

        # Transformer prior
        self.prior = CVAEBFMPriorV2(
            tf_d_model, latent_dim, tf_num_heads, tf_hidden_dim,
            prior_num_layers, tf_dropout, tf_activation)

        # Residual posterior
        self.posterior = CVAEBFMPosteriorV2(
            keybody_dim, latent_dim, tf_d_model, tf_num_heads, tf_hidden_dim, tf_activation)

        # MLP decoder
        self.action_decoder = CVAEBFMDecoderV2(
            latent_dim, tf_d_model, decoder_hidden_dims, num_actions, tf_activation)

        # -- Teacher --
        teacher_ckpt = torch.load(teacher_policy_ckpt, map_location="cpu", weights_only=False)
        self.obs_norm_state_dict = teacher_ckpt.get("obs_norm_state_dict", None)
        teacher_policy_cfg = teacher_ckpt["policy_cfg"]
        teacher_policy_class = eval(teacher_policy_cfg.pop("class_name"))
        teacher_policy_args = teacher_policy_cfg.pop("_args")
        assert num_actions == teacher_policy_args[2]
        self.teacher: ActorCritic = teacher_policy_class(*teacher_policy_args, **teacher_policy_cfg)
        self.teacher.load_state_dict(teacher_ckpt["model_state_dict"], strict=True)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.loaded_teacher = True

        # -- Noise --
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

        # -- State --
        self.compute_latent_loss = False
        self._save_dict = {}
        self._ep_frame_mask = None
        self._ep_kp_mask = None
        self._ep_frame_offsets = None
        self._env_ref = None
        self._distill_step = 0

    @property
    def student(self):
        return self

    def set_env(self, env):
        self._env_ref = env

    def _resolve_obs_meta(self, num_obs, obs_meta):
        all_obs = torch.arange(num_obs)
        history_mask = torch.ones(num_obs, dtype=torch.bool)
        condition_obs, keybody_obs, current_proprio_obs = [], [], []
        for group, lst in [("conditions", condition_obs), ("posterior_conditions", keybody_obs), ("current_proprio", current_proprio_obs)]:
            if group in obs_meta:
                for seg in obs_meta[group]:
                    lst.append(all_obs[seg["start"]:seg["end"]].clone())
                    history_mask[seg["start"]:seg["end"]] = False
        return (all_obs[history_mask].clone().contiguous(),
                torch.cat(current_proprio_obs).contiguous() if current_proprio_obs else torch.tensor([], dtype=torch.long),
                torch.cat(condition_obs).contiguous() if condition_obs else torch.tensor([], dtype=torch.long),
                torch.cat(keybody_obs).contiguous() if keybody_obs else torch.tensor([], dtype=torch.long))

    def _split_obs(self, obs):
        return (obs[..., self.history_proprio_ids].contiguous(),
                obs[..., self.current_proprio_ids].contiguous(),
                obs[..., self.condition_ids].contiguous(),
                obs[..., self.keybody_ids].contiguous())

    def _parse_condition(self, y_flat):
        B = y_flat.shape[0]
        y = y_flat.view(B, self.num_frames, self.frame_dim)
        return y[:, :, :-1], y[:, :, -1]

    def _get_rng(self, device):
        if self._mask_rng is None or self._mask_rng.device != device:
            self._mask_rng = torch.Generator(device=device)
            self._mask_rng.manual_seed(self._mask_rng_seed)
        return self._mask_rng

    def _sample_count_mask(self, n, total, gen, device, min_active=0):
        counts = torch.randint(min_active, total + 1, (n,), device=device, generator=gen)
        mask = torch.zeros(n, total, dtype=torch.bool, device=device)
        for i in range(n):
            c = counts[i].item()
            if c > 0:
                perm = torch.randperm(total, device=device, generator=gen)[:c]
                mask[i, perm] = True
        return mask

    def _normalize_history(self, h_enc, add_noise=False):
        h_enc = F.normalize(h_enc, dim=-1)
        if add_noise and self.history_sigma > 0:
            h_enc = h_enc + self.history_sigma * torch.randn_like(h_enc)
            h_enc = F.normalize(h_enc, dim=-1)
        return h_enc

    def _apply_masks(self, frames, frame_mask=None, kp_mask=None):
        B, nf, KD = frames.shape
        K, D = self.rollout_mask_num_keypoints, self.dims_per_keypoint
        if kp_mask is not None and K > 0:
            frames = (frames.view(B, nf, K, D) * kp_mask[:, None, :, None].float()).flatten(2)
        if frame_mask is not None:
            frames = frames * frame_mask.unsqueeze(-1).float()
        return frames

    def _kl_coef(self):
        """KL coefficient with linear ramp."""
        if self.kl_ramp_steps <= 0:
            return self.kl_coef_end
        t = min(self._distill_step / self.kl_ramp_steps, 1.0)
        return self.kl_coef_start + t * (self.kl_coef_end - self.kl_coef_start)

    # -- Offsets (same as LFM-BFM) --

    def _sample_initial_offsets(self, env_ids, device):
        n = env_ids.shape[0]
        nf, K = self.num_frames, self.rollout_mask_num_keypoints
        gen = self._get_rng(device)
        offsets = torch.empty(n, nf, device=device).uniform_(self.min_frame_delta, self.max_frame_delta, generator=gen)
        offsets = (offsets / self.step_dt).round() * self.step_dt
        offsets = offsets.clamp(min=self.step_dt)
        offsets = offsets.sort(dim=1).values
        for i in range(1, nf):
            offsets[:, i] = torch.max(offsets[:, i], offsets[:, i - 1] + self.step_dt)
        self._ep_frame_offsets[env_ids] = offsets
        self._ep_frame_mask[env_ids] = self._sample_count_mask(n, nf, gen, device, min_active=1)
        if K > 0:
            self._ep_kp_mask[env_ids] = self._sample_count_mask(n, K, gen, device, min_active=1)
        self._push_offsets_to_env(env_ids)

    def _step_offsets(self):
        offsets, mask = self._ep_frame_offsets, self._ep_frame_mask
        dev = offsets.device
        gen = self._get_rng(dev)
        offsets -= self.step_dt
        consumed = offsets[:, 0] <= 0
        if consumed.any():
            consumed_was_active = mask[consumed, 0].clone()
            offsets[consumed, :-1] = offsets[consumed, 1:].clone()
            mask[consumed, :-1] = mask[consumed, 1:].clone()
            n = consumed.sum()
            gap = torch.empty(n, device=dev).uniform_(self.min_frame_delta, self.max_frame_delta, generator=gen)
            gap = (gap / self.step_dt).round() * self.step_dt
            gap = gap.clamp(min=self.step_dt)
            offsets[consumed, -1] = offsets[consumed, -2] + gap
            mask[consumed, -1] = consumed_was_active
        self._push_offsets_to_env()

    def _push_offsets_to_env(self, env_ids=None):
        if self._env_ref is None:
            return
        env = self._env_ref
        nf = self.num_frames
        if not hasattr(env, '_bfm_future_offsets') or env._bfm_future_offsets is None:
            env._bfm_future_offsets = torch.zeros(env.num_envs, nf, device=self._ep_frame_offsets.device)
            env._bfm_delta_t = torch.zeros(env.num_envs, nf, device=self._ep_frame_offsets.device)
        if env_ids is None:
            env._bfm_future_offsets[:] = self._ep_frame_offsets
            env._bfm_delta_t[:] = self._ep_frame_offsets
        else:
            env._bfm_future_offsets[env_ids] = self._ep_frame_offsets[env_ids]
            env._bfm_delta_t[env_ids] = self._ep_frame_offsets[env_ids]

    # -- Forward --

    def forward(self):
        raise NotImplementedError

    def act(self, observations, *args, **kwargs):
        """Rollout: prior only → sample z → MLP decoder."""
        hp_t, o_t, y_flat, r_t = self._split_obs(observations)
        frames, delta_t = self._parse_condition(y_flat)
        B = observations.shape[0]
        nf, K = self.num_frames, self.rollout_mask_num_keypoints

        if self._ep_frame_mask is None or self._ep_frame_mask.shape[0] != B:
            with torch.inference_mode(False):
                self._ep_frame_mask = torch.ones(B, nf, dtype=torch.bool, device=observations.device)
                self._ep_kp_mask = torch.ones(B, K, dtype=torch.bool, device=observations.device) if K > 0 else None
                self._ep_frame_offsets = torch.zeros(B, nf, device=observations.device)
                self._sample_initial_offsets(torch.arange(B, device=observations.device), observations.device)
        else:
            with torch.inference_mode(False):
                self._step_offsets()

        masked_frames = self._apply_masks(frames, self._ep_frame_mask, self._ep_kp_mask)
        cur_frame_mask = self._ep_frame_mask[:B]

        o_t_enc = self.o_t_encoder(o_t)
        h_enc = self._normalize_history(self.history_encoder(hp_t), add_noise=False)
        frame_tokens = self.frame_encoder(masked_frames, delta_t)

        # Prior → sample z (no posterior at deployment)
        mu_prior, logvar_prior = self.prior(h_enc, o_t_enc, frame_tokens, cur_frame_mask)
        z = mu_prior  # use mean at inference (no sampling noise)

        action_mean = self.action_decoder(z, o_t_enc)

        std = self.std.expand_as(action_mean)
        self.distribution = Normal(action_mean, std)
        return self.distribution.sample()

    def evaluate(self, teacher_observations, *args, **kwargs):
        with torch.no_grad():
            return self.teacher.act_inference(teacher_observations, *args, **kwargs)

    def act_inference(self, observations, *args, **kwargs):
        """Training: prior + residual posterior → sample z → decoder."""
        hp_t, o_t, y_flat, r_t = self._split_obs(observations)
        frames, delta_t = self._parse_condition(y_flat)
        B = frames.shape[0]
        nf, K = self.num_frames, self.rollout_mask_num_keypoints

        o_t_enc = self.o_t_encoder(o_t)
        h_enc = self._normalize_history(self.history_encoder(hp_t), add_noise=self.compute_latent_loss)

        if self.compute_latent_loss:
            dev = frames.device
            gen = self._get_rng(dev)
            frame_mask = self._sample_count_mask(B, nf, gen, dev, min_active=0)
            kp_mask = self._sample_count_mask(B, K, gen, dev, min_active=0) if K > 0 else None
        else:
            frame_mask = torch.ones(B, nf, dtype=torch.bool, device=frames.device)
            kp_mask = None

        masked_frames = self._apply_masks(frames, frame_mask, kp_mask)
        frame_tokens = self.frame_encoder(masked_frames, delta_t)

        # Prior
        mu_prior, logvar_prior = self.prior(h_enc, o_t_enc, frame_tokens, frame_mask)
        logvar_prior = logvar_prior.clamp(*self._LOGVAR_CLAMP)

        if self.compute_latent_loss and r_t.shape[-1] > 0:
            # History dropout for prior context
            if self.history_dropout_prob > 0:
                # Re-run prior with h_enc masked for some samples
                h_drop = torch.rand(B, device=dev, generator=gen) < self.history_dropout_prob
                if h_drop.any():
                    h_enc_dropped = h_enc.clone()
                    h_enc_dropped[h_drop] = 0.0
                    mu_prior_d, logvar_prior_d = self.prior(h_enc_dropped, o_t_enc, frame_tokens, frame_mask)
                    mu_prior[h_drop] = mu_prior_d[h_drop]
                    logvar_prior[h_drop] = logvar_prior_d[h_drop].clamp(*self._LOGVAR_CLAMP)

            # Residual posterior
            delta_mu, delta_logvar = self.posterior(r_t, frame_tokens, frame_mask)
            mu_post = mu_prior + delta_mu
            logvar_post = delta_logvar.clamp(*self._LOGVAR_CLAMP)

            # Sample z from posterior
            std_post = (0.5 * logvar_post).exp()
            z = mu_post + std_post * torch.randn_like(std_post)

            # KL(posterior || prior)
            std_prior = (0.5 * logvar_prior).exp()
            kl = (logvar_prior - logvar_post + (std_post.pow(2) + (mu_post - mu_prior).pow(2)) / (2 * std_prior.pow(2) + 1e-8) - 0.5).mean()

            # Posterior dropout: zero z for some samples (matches rollout prior-only)
            if self.posterior_dropout > 0:
                drop_mask = torch.rand(B, device=z.device) < self.posterior_dropout
                if drop_mask.any():
                    # Use prior sample instead of posterior for dropped envs
                    z[drop_mask] = mu_prior[drop_mask]

            self._save_dict = {"kl": kl}
        else:
            z = mu_prior

        action_mean = self.action_decoder(z, o_t_enc)
        return action_mean

    def extra_loss(self, **kwargs):
        if not self._save_dict:
            return {}, {}
        d = self._save_dict
        kl_coef = self._kl_coef()
        loss_dict = {"kl": d["kl"] * kl_coef}
        log_dict = {"kl": d["kl"].item(), "kl_coef": kl_coef}
        self._save_dict = {}
        return dict(loss_dict), dict(log_dict)

    def pre_train(self):
        self.compute_latent_loss = True
        self._distill_step += 1

    def after_train(self):
        self.compute_latent_loss = False
        self._save_dict = {}

    def reset(self, dones=None, hidden_states=None):
        if dones is not None and dones.any():
            env_ids = dones.bool().flatten().nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() > 0 and self._ep_frame_mask is not None:
                with torch.inference_mode(False):
                    self._sample_initial_offsets(env_ids, self._ep_frame_mask.device)

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    def load_state_dict(self, state_dict, strict=True):
        student_keys = [k for k in state_dict.keys() if not k.startswith("teacher")]
        super().load_state_dict({k: v for k, v in state_dict.items() if k in student_keys}, strict=False)
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
