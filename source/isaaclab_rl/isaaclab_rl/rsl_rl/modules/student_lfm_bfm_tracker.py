# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LFM-BFM: Latent Flow Matching foundation model.

Flow matching on L2-normalized latent space (hypersphere):
  - Posterior: keybody + frames → z_t on unit sphere (training only)
  - Latent flow: ODE in latent space to generate z_t from noise
  - Decoder: deterministic action from z_t + context (no action-space ODE)
  - Training: flow loss on latent velocity + behavior loss on action
  - Rollout: latent ODE with KV cache → z_t → action
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
from isaaclab_rl.rsl_rl.networks.cvae_bfm_networks import BFMFrameEncoder
from isaaclab_rl.rsl_rl.networks.flow_bfm_networks import FlowBFMEncoder
from isaaclab_rl.rsl_rl.networks.lfm_bfm_networks import LFMPosterior, LatentFlowDecoder, LFMActionDecoder


class StudentLFMBFMTracker(nn.Module):
    """Latent Flow Matching BFM student policy."""

    is_recurrent = False

    def __init__(self, num_student_obs, num_teacher_obs, num_actions,
                 student_policy_cfg, teacher_policy_ckpt,
                 student_obs_meta, teacher_obs_meta,
                 init_noise_std=0.1, **kwargs):
        if kwargs:
            print("StudentLFMBFMTracker: unexpected args: " + str(list(kwargs.keys())))
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
        activation_name = cfg.pop("activation", "elu")
        activation = resolve_nn_activation(activation_name)

        latent_dim = cfg.pop("latent_dim", 64)
        for k in ["corr_rank", "history_hidden_dims", "posterior_hidden_dims",
                   "corr_kl_coef", "latent_kl_coef", "prior_reg_coef"]:
            cfg.pop(k, None)
        prior_hidden_dims = cfg.pop("prior_hidden_dims", [512, 256])

        self.num_keypoints = cfg.pop("num_keypoints", 6)
        self.dims_per_keypoint = cfg.pop("dims_per_keypoint", 9)
        self.num_frames = cfg.pop("num_frames", 10)
        self.step_dt = cfg.pop("step_dt", 0.02)
        self.frame_dim = self.num_keypoints * self.dims_per_keypoint + 1
        assert cond_dim == self.num_frames * self.frame_dim

        self.min_frame_delta = cfg.pop("min_frame_delta", 0.02)
        self.max_frame_delta = cfg.pop("max_frame_delta", 1.0)
        self.frame_p_active_range = tuple(cfg.pop("frame_p_active_range", [0.5, 1.0]))
        self.rollout_mask_num_keypoints = cfg.pop("rollout_mask_num_keypoints", 6)
        self.rollout_mask_p_clean_range = tuple(cfg.pop("rollout_mask_p_clean_range", [0.2, 1.0]))

        tf_d_model = cfg.pop("tf_d_model", 256)
        tf_num_heads = cfg.pop("tf_num_heads", 4)
        tf_hidden_dim = cfg.pop("tf_hidden_dim", 512)
        tf_dropout = cfg.pop("tf_dropout", 0.0)
        tf_activation_name = cfg.pop("tf_activation", "gelu")
        if tf_activation_name == "gelu":
            tf_activation = nn.GELU(approximate="tanh")
        else:
            tf_activation = resolve_nn_activation(tf_activation_name)

        encoder_num_layers = cfg.pop("encoder_num_layers", 2)
        decoder_num_layers = cfg.pop("decoder_num_layers", 2)
        flow_num_layers = cfg.pop("flow_num_layers", 2)
        self.ode_steps = cfg.pop("ode_steps", 10)
        self.posterior_sigma = cfg.pop("posterior_sigma", 0.1)
        self.boundary_coef = cfg.pop("boundary_coef", 1.0)
        cfg.pop("posterior_dropout", None)  # unused

        self.latent_dim = latent_dim
        self.num_actions = num_actions

        # Teacher forcing: fraction of rollout envs that use teacher action + noise
        self.teacher_forcing_ratio = cfg.pop("teacher_forcing_ratio", 0.0)
        self.teacher_forcing_noise = cfg.pop("teacher_forcing_noise", 0.1)

        if cfg:
            print(f"StudentLFMBFMTracker: unused keys: {list(cfg.keys())}")

        # -- Networks --
        self.history_prior = _build_mlp(history_proprio_dim, prior_hidden_dims, latent_dim, activation)

        self.frame_encoder = BFMFrameEncoder(
            self.num_keypoints, self.dims_per_keypoint, tf_d_model, tf_activation)

        # Posterior: keybody + frames → z_t on sphere
        self.posterior = LFMPosterior(
            keybody_dim, latent_dim, tf_d_model, self.frame_encoder,
            tf_num_heads, tf_hidden_dim, num_layers=2, activation=tf_activation)

        # Context encoder: [h_prior, proprio, frames] → context tokens
        self.encoder = FlowBFMEncoder(
            latent_dim, current_proprio_dim, tf_d_model, self.frame_encoder,
            tf_num_heads, tf_hidden_dim, encoder_num_layers, tf_dropout, tf_activation)

        # Latent flow decoder: noised z cross-attends to context → velocity
        self.latent_flow = LatentFlowDecoder(
            latent_dim, tf_d_model, tf_num_heads, tf_hidden_dim,
            flow_num_layers, tf_dropout, tf_activation)

        # Action decoder: takes encoded context + z_t → action (1-layer)
        self.action_decoder = LFMActionDecoder(
            latent_dim, tf_d_model, tf_num_heads, tf_hidden_dim,
            decoder_num_layers, num_actions, tf_dropout, tf_activation)

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

    def _apply_masks(self, frames, frame_mask=None, kp_mask=None):
        B, nf, KD = frames.shape
        K, D = self.rollout_mask_num_keypoints, self.dims_per_keypoint
        if kp_mask is not None and K > 0:
            frames = (frames.view(B, nf, K, D) * kp_mask[:, None, :, None].float()).flatten(2)
        if frame_mask is not None:
            frames = frames * frame_mask.unsqueeze(-1).float()
        return frames

    # -- Offsets (same as other BFM variants) --

    def _sample_initial_offsets(self, env_ids, device):
        n = env_ids.shape[0]
        nf, K = self.num_frames, self.rollout_mask_num_keypoints
        offsets = torch.empty(n, nf, device=device).uniform_(self.min_frame_delta, self.max_frame_delta)
        offsets = (offsets / self.step_dt).round() * self.step_dt
        offsets = offsets.clamp(min=self.step_dt)
        offsets = offsets.sort(dim=1).values
        for i in range(1, nf):
            offsets[:, i] = torch.max(offsets[:, i], offsets[:, i - 1] + self.step_dt)
        self._ep_frame_offsets[env_ids] = offsets
        p_active = torch.empty(n, device=device).uniform_(*self.frame_p_active_range)
        fm = torch.rand(n, nf, device=device) < p_active[:, None]
        all_off = ~fm.any(dim=1)
        if all_off.any():
            fm[all_off, torch.randint(0, nf, (all_off.sum(),), device=device)] = True
        self._ep_frame_mask[env_ids] = fm
        if K > 0:
            p_clean = torch.empty(n, device=device).uniform_(*self.rollout_mask_p_clean_range)
            self._ep_kp_mask[env_ids] = torch.rand(n, K, device=device) < p_clean[:, None]
        self._push_offsets_to_env(env_ids)

    def _step_offsets(self):
        offsets, mask = self._ep_frame_offsets, self._ep_frame_mask
        offsets -= self.step_dt
        consumed = offsets[:, 0] <= 0
        if consumed.any():
            offsets[consumed, :-1] = offsets[consumed, 1:].clone()
            mask[consumed, :-1] = mask[consumed, 1:].clone()
            n = consumed.sum()
            gap = torch.empty(n, device=offsets.device).uniform_(self.min_frame_delta, self.max_frame_delta)
            gap = (gap / self.step_dt).round() * self.step_dt
            gap = gap.clamp(min=self.step_dt)
            offsets[consumed, -1] = offsets[consumed, -2] + gap
            p = torch.empty(n, device=offsets.device).uniform_(*self.frame_p_active_range)
            mask[consumed, -1] = torch.rand(n, device=offsets.device) < p
        # Ensure at least 1 active
        all_off = ~mask.any(dim=1)
        if all_off.any():
            nf = mask.shape[1]
            mask[all_off, torch.randint(0, nf, (all_off.sum(),), device=mask.device)] = True
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

    def _ode_sample_latent(self, context, ctx_mask, B, device):
        """K-step Euler ODE in latent space with KV cache."""
        kv_cache, ctx_mask = self.latent_flow.build_kv_cache(context, ctx_mask)
        z = torch.randn(B, self.latent_dim, device=device)
        dt = 1.0 / self.ode_steps
        for k in range(self.ode_steps):
            t = 1.0 - k * dt
            t_tensor = torch.full((B,), t, device=device)
            v = self.latent_flow.forward_cached(z, t_tensor, kv_cache, ctx_mask)
            z = z - dt * v
        return z  # bounded by posterior reg loss, not hard-normalized

    # -- Forward --

    def forward(self):
        raise NotImplementedError

    def act(self, observations, *args, **kwargs):
        """Rollout: latent ODE → z_t → decoder → action."""
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

        h_prior = self.history_prior(hp_t)
        context, ctx_mask = self.encoder(h_prior, o_t, masked_frames, delta_t, cur_frame_mask)

        # Latent ODE → z_t
        z_t = self._ode_sample_latent(context, ctx_mask, B, observations.device)

        action_mean = self.action_decoder(z_t, context, ctx_mask)

        std = self.std.expand_as(action_mean)
        self.distribution = Normal(action_mean, std)
        return self.distribution.sample()

    def evaluate(self, teacher_observations, *args, **kwargs):
        with torch.no_grad():
            return self.teacher.act_inference(teacher_observations, *args, **kwargs)

    def act_inference(self, observations, *args, **kwargs):
        """Training: posterior z_t + flow loss. Inference: latent ODE."""
        hp_t, o_t, y_flat, r_t = self._split_obs(observations)
        frames, delta_t = self._parse_condition(y_flat)
        B = frames.shape[0]
        nf, K = self.num_frames, self.rollout_mask_num_keypoints

        if self.compute_latent_loss:
            p_active = torch.empty(B, device=frames.device).uniform_(*self.frame_p_active_range)
            frame_mask = torch.rand(B, nf, device=frames.device) < p_active[:, None]
            all_off = ~frame_mask.any(dim=1)
            if all_off.any():
                frame_mask[all_off, torch.randint(0, nf, (all_off.sum(),), device=frames.device)] = True
            kp_mask = None
            if K > 0:
                p_clean = torch.empty(B, device=frames.device).uniform_(*self.rollout_mask_p_clean_range)
                kp_mask = torch.rand(B, K, device=frames.device) < p_clean[:, None]
        else:
            frame_mask = torch.ones(B, nf, dtype=torch.bool, device=frames.device)
            kp_mask = None

        masked_frames = self._apply_masks(frames, frame_mask, kp_mask)
        h_prior = self.history_prior(hp_t)
        context, ctx_mask = self.encoder(h_prior, o_t, masked_frames, delta_t, frame_mask)

        if self.compute_latent_loss and r_t.shape[-1] > 0:
            # Posterior: z_t in [-1, 1] (soft bounded via boundary loss)
            z_t = self.posterior(r_t, masked_frames, delta_t, frame_mask)

            # Flow loss: noise z_t, predict velocity in latent space
            t = torch.rand(B, device=frames.device)
            eps = torch.randn(B, self.latent_dim, device=frames.device)
            z_noised = (1 - t.unsqueeze(-1)) * z_t + t.unsqueeze(-1) * eps
            v_target = eps - z_t

            v_pred = self.latent_flow(z_noised, t, context, ctx_mask)
            flow_loss = F.mse_loss(v_pred, v_target)

            # Add fixed noise for tolerance area
            z_posterior = z_t + self.posterior_sigma * torch.randn_like(z_t)
            # no sphere normalization — posterior reg keeps values bounded

            # Boundary loss: penalize |z_t| > 1 per dimension
            boundary_loss = F.relu(z_t.abs() - 1.0).pow(2).mean()

            self._save_dict = {
                "flow_loss": flow_loss,
                "boundary_loss": boundary_loss,
                "context": context,
                "ctx_mask": ctx_mask,
            }

            # Action from z_posterior
            action_mean = self.action_decoder(z_posterior, context, ctx_mask)
            return action_mean
        else:
            # Inference: latent ODE
            z_t = self._ode_sample_latent(context, ctx_mask, B, frames.device)
            action_mean = self.action_decoder(z_t, context, ctx_mask)
            return action_mean

    def extra_loss(self, **kwargs):
        if not self._save_dict:
            return {}, {}

        d = self._save_dict
        loss_dict = {}
        log_dict = {}

        loss_dict["flow"] = d["flow_loss"]
        log_dict["flow"] = d["flow_loss"].item()

        loss_dict["boundary"] = d["boundary_loss"] * self.boundary_coef
        log_dict["boundary"] = d["boundary_loss"].item()

        self._save_dict = {}
        return dict(loss_dict), dict(log_dict)

    def pre_train(self):
        self.compute_latent_loss = True

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
