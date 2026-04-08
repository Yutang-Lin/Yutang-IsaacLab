# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flow-BFM: Flow matching on action space with cross-attention decoder.

No posterior. Encoder provides context, decoder denoises action via ODE.
  - Encoder: [h_prior, proprio, frame_0..F-1] → context tokens
  - Decoder: noised action cross-attends to context → velocity
  - Training: flow matching loss on action velocity
  - Rollout: K-step Euler ODE integration with KV cache
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
from isaaclab_rl.rsl_rl.networks.flow_bfm_networks import FlowBFMEncoder, FlowBFMDecoder


class StudentFlowBFMTracker(nn.Module):
    """Flow-BFM student policy for distillation."""

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
            print("StudentFlowBFMTracker.__init__ got unexpected args: " + str(list(kwargs.keys())))
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

        # -- Extract hyperparams --
        cfg = dict(student_policy_cfg)
        cfg.pop("class_name", None)
        activation_name = cfg.pop("activation", "elu")
        activation = resolve_nn_activation(activation_name)

        latent_dim = cfg.pop("latent_dim", 64)
        # Pop unused CVAE params for compatibility
        for k in ["corr_rank", "history_hidden_dims", "posterior_hidden_dims",
                   "corr_kl_coef", "latent_kl_coef", "posterior_dropout"]:
            cfg.pop(k, None)
        prior_hidden_dims = cfg.pop("prior_hidden_dims", [512, 256])

        # BFM frame parameters
        self.num_keypoints = cfg.pop("num_keypoints", 6)
        self.dims_per_keypoint = cfg.pop("dims_per_keypoint", 9)
        self.num_frames = cfg.pop("num_frames", 10)
        self.step_dt = cfg.pop("step_dt", 0.02)
        self.frame_dim = self.num_keypoints * self.dims_per_keypoint + 1
        expected_cond_dim = self.num_frames * self.frame_dim
        assert cond_dim == expected_cond_dim

        self.min_frame_delta = cfg.pop("min_frame_delta", 0.02)
        self.max_frame_delta = cfg.pop("max_frame_delta", 1.0)
        self.frame_p_active_range = tuple(cfg.pop("frame_p_active_range", [0.5, 1.0]))
        self.rollout_mask_num_keypoints = cfg.pop("rollout_mask_num_keypoints", 6)
        self.rollout_mask_p_clean_range = tuple(cfg.pop("rollout_mask_p_clean_range", [0.2, 1.0]))

        # Transformer config
        tf_d_model = cfg.pop("tf_d_model", 256)
        tf_num_heads = cfg.pop("tf_num_heads", 4)
        tf_hidden_dim = cfg.pop("tf_hidden_dim", 512)
        tf_dropout = cfg.pop("tf_dropout", 0.0)
        tf_activation_name = cfg.pop("tf_activation", "gelu")
        if tf_activation_name == "gelu":
            tf_activation = nn.GELU(approximate="tanh")
        else:
            tf_activation = resolve_nn_activation(tf_activation_name)

        # Encoder/decoder layer counts
        encoder_num_layers = cfg.pop("encoder_num_layers", 2)
        decoder_num_layers = cfg.pop("decoder_num_layers", 1)

        # Flow config
        self.ode_steps = cfg.pop("ode_steps", 10)
        self.prior_reg_coef = cfg.pop("prior_reg_coef", 1e-5)

        self.latent_dim = latent_dim
        self.num_actions = num_actions

        if cfg:
            print(f"StudentFlowBFMTracker: unused config keys: {list(cfg.keys())}")

        # -- Build sub-networks --
        self.history_prior = _build_mlp(history_proprio_dim, prior_hidden_dims, latent_dim, activation)

        self.frame_encoder = BFMFrameEncoder(
            num_keypoints=self.num_keypoints,
            dims_per_keypoint=self.dims_per_keypoint,
            d_model=tf_d_model,
            activation=tf_activation,
        )

        self.encoder = FlowBFMEncoder(
            latent_dim=latent_dim,
            proprio_dim=current_proprio_dim,
            d_model=tf_d_model,
            frame_encoder=self.frame_encoder,
            num_heads=tf_num_heads,
            hidden_dim=tf_hidden_dim,
            num_layers=encoder_num_layers,
            dropout=tf_dropout,
            activation=tf_activation,
        )

        self.decoder = FlowBFMDecoder(
            num_actions=num_actions,
            d_model=tf_d_model,
            num_heads=tf_num_heads,
            hidden_dim=tf_hidden_dim,
            num_layers=decoder_num_layers,
            dropout=tf_dropout,
            activation=tf_activation,
        )

        # -- Load frozen teacher --
        teacher_ckpt = torch.load(teacher_policy_ckpt, map_location="cpu", weights_only=False)
        self.obs_norm_state_dict = teacher_ckpt.get("obs_norm_state_dict", None)
        teacher_policy_cfg = teacher_ckpt["policy_cfg"]
        teacher_policy_class = eval(teacher_policy_cfg.pop("class_name"))
        teacher_policy_args = teacher_policy_cfg.pop("_args")
        assert num_actions == teacher_policy_args[2]
        self.teacher: ActorCritic = teacher_policy_class(*teacher_policy_args, **teacher_policy_cfg)
        self.teacher.load_state_dict(teacher_ckpt["model_state_dict"], strict=True)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.loaded_teacher = True

        print(f"StudentFlowBFMTracker: latent={latent_dim}, ode_steps={self.ode_steps}, "
              f"enc_layers={encoder_num_layers}, dec_layers={decoder_num_layers}")

        # -- Action EMA min/max normalizer --
        # Track EMA min/max of teacher actions for first warmup_iters, then freeze
        self.register_buffer("_act_ema_min", torch.zeros(num_actions))
        self.register_buffer("_act_ema_max", torch.ones(num_actions))
        self._act_ema_decay = 0.99
        self._act_warmup_iters = cfg.pop("act_warmup_iters", 100)
        self._act_warmup_count = 0

        # -- Action noise --
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
        B, F, KD = frames.shape
        K, D = self.rollout_mask_num_keypoints, self.dims_per_keypoint
        if kp_mask is not None and K > 0:
            frames = (frames.view(B, F, K, D) * kp_mask[:, None, :, None].float()).flatten(2)
        if frame_mask is not None:
            frames = frames * frame_mask.unsqueeze(-1).float()
        return frames

    # -- Episodic offsets (same as CVAE-BFM) --

    def _sample_initial_offsets(self, env_ids, device):
        n = env_ids.shape[0]
        F, K = self.num_frames, self.rollout_mask_num_keypoints
        offsets = torch.empty(n, F, device=device).uniform_(self.min_frame_delta, self.max_frame_delta)
        self._ep_frame_offsets[env_ids] = offsets.sort(dim=1).values
        p_active = torch.empty(n, device=device).uniform_(*self.frame_p_active_range)
        fm = torch.rand(n, F, device=device) < p_active[:, None]
        all_off = ~fm.any(dim=1)
        if all_off.any():
            fm[all_off, torch.randint(0, F, (all_off.sum(),), device=device)] = True
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
            offsets[consumed, -1] = offsets[consumed, -2] + torch.empty(n, device=offsets.device).uniform_(self.min_frame_delta, self.max_frame_delta)
            p = torch.empty(n, device=offsets.device).uniform_(*self.frame_p_active_range)
            mask[consumed, -1] = torch.rand(n, device=offsets.device) < p
        self._push_offsets_to_env()

    def _push_offsets_to_env(self, env_ids=None):
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

    def _encode_context(self, hp_t, o_t, masked_frames, delta_t, frame_mask):
        """Encode context from history + proprio + frames."""
        h_prior = self.history_prior(hp_t)
        context, ctx_mask = self.encoder(h_prior, o_t, masked_frames, delta_t, frame_mask)
        return context, ctx_mask, h_prior

    def _update_act_norm(self, actions):
        """Update EMA min/max from teacher actions during warmup."""
        if self._act_warmup_count >= self._act_warmup_iters:
            return
        with torch.no_grad():
            batch_min = actions.min(dim=0).values
            batch_max = actions.max(dim=0).values
            d = self._act_ema_decay
            if self._act_warmup_count == 0:
                self._act_ema_min.copy_(batch_min)
                self._act_ema_max.copy_(batch_max)
            else:
                self._act_ema_min.copy_(self._act_ema_min * d + batch_min * (1 - d))
                self._act_ema_max.copy_(self._act_ema_max * d + batch_max * (1 - d))
            self._act_warmup_count += 1

    def _normalize_action(self, a):
        """Normalize action to [-1, 1] using EMA min/max."""
        r = (self._act_ema_max - self._act_ema_min).clamp(min=1e-6)
        return 2.0 * (a - self._act_ema_min) / r - 1.0

    def _denormalize_action(self, a_norm):
        """Denormalize action from [-1, 1] to original scale."""
        r = (self._act_ema_max - self._act_ema_min).clamp(min=1e-6)
        return (a_norm + 1.0) * 0.5 * r + self._act_ema_min

    def _ode_sample(self, context, ctx_mask, B, device):
        """K-step Euler ODE integration from noise to normalized action, then denormalize."""
        # Build KV cache once
        kv_cache, ctx_mask = self.decoder.build_kv_cache(context, ctx_mask)

        # Start from pure noise (t=1)
        a_t = torch.randn(B, self.num_actions, device=device)
        dt = 1.0 / self.ode_steps

        for k in range(self.ode_steps):
            t = 1.0 - k * dt  # t goes from 1 → dt
            t_tensor = torch.full((B,), t, device=device)
            v = self.decoder.forward_cached(a_t, t_tensor, kv_cache, ctx_mask)
            a_t = a_t - dt * v  # Euler step

        # a_0 is in normalized [-1, 1] space, denormalize to action scale
        return self._denormalize_action(a_t)

    # -- Forward paths --

    def forward(self):
        raise NotImplementedError

    def act(self, observations, *args, **kwargs):
        """Rollout: ODE integration with KV cache."""
        hp_t, o_t, y_flat, r_t = self._split_obs(observations)
        frames, delta_t = self._parse_condition(y_flat)
        B = observations.shape[0]
        F, K = self.num_frames, self.rollout_mask_num_keypoints

        if self._ep_frame_mask is None or self._ep_frame_mask.shape[0] != B:
            with torch.inference_mode(False):
                self._ep_frame_mask = torch.ones(B, F, dtype=torch.bool, device=observations.device)
                self._ep_kp_mask = torch.ones(B, K, dtype=torch.bool, device=observations.device) if K > 0 else None
                self._ep_frame_offsets = torch.zeros(B, F, device=observations.device)
                self._sample_initial_offsets(torch.arange(B, device=observations.device), observations.device)
        else:
            with torch.inference_mode(False):
                self._step_offsets()

        masked_frames = self._apply_masks(frames, self._ep_frame_mask, self._ep_kp_mask)
        cur_frame_mask = self._ep_frame_mask[:B]

        context, ctx_mask, _ = self._encode_context(hp_t, o_t, masked_frames, delta_t, cur_frame_mask)
        action_mean = self._ode_sample(context, ctx_mask, B, observations.device)

        std = self.std.expand_as(action_mean)
        self.distribution = Normal(action_mean, std)
        return self.distribution.sample()

    def evaluate(self, teacher_observations, *args, **kwargs):
        with torch.no_grad():
            return self.teacher.act_inference(teacher_observations, *args, **kwargs)

    def act_inference(self, observations, *args, **kwargs):
        """Training: flow matching loss. Inference: ODE integration."""
        hp_t, o_t, y_flat, r_t = self._split_obs(observations)
        frames, delta_t = self._parse_condition(y_flat)
        B, F, K = frames.shape[0], self.num_frames, self.rollout_mask_num_keypoints

        if self.compute_latent_loss:
            # Random masks
            p_active = torch.empty(B, device=frames.device).uniform_(*self.frame_p_active_range)
            frame_mask = torch.rand(B, F, device=frames.device) < p_active[:, None]
            all_off = ~frame_mask.any(dim=1)
            if all_off.any():
                frame_mask[all_off, torch.randint(0, F, (all_off.sum(),), device=frames.device)] = True
            kp_mask = None
            if K > 0:
                p_clean = torch.empty(B, device=frames.device).uniform_(*self.rollout_mask_p_clean_range)
                kp_mask = torch.rand(B, K, device=frames.device) < p_clean[:, None]
        else:
            frame_mask = torch.ones(B, F, dtype=torch.bool, device=frames.device)
            kp_mask = None

        masked_frames = self._apply_masks(frames, frame_mask, kp_mask)
        context, ctx_mask, h_prior = self._encode_context(hp_t, o_t, masked_frames, delta_t, frame_mask)

        if self.compute_latent_loss:
            # Cache context for flow_loss in extra_loss (which receives teacher actions)
            # Return zeros detached — behavior_loss from distillation loop is disabled
            # (gradient blocked by detach, all training comes from flow_loss)
            self._save_dict = {
                "h_prior": h_prior,
                "context": context,
                "ctx_mask": ctx_mask,
            }
            return torch.zeros(B, self.num_actions, device=frames.device).detach()
        else:
            return self._ode_sample(context, ctx_mask, B, frames.device)

    def extra_loss(self, **kwargs):
        if not self._save_dict:
            return {}, {}

        d = self._save_dict
        loss_dict = {}
        log_dict = {}

        # Flow matching loss on normalized actions
        privileged_actions = kwargs.get("privileged_actions_batch", None)
        if privileged_actions is not None:
            context = d["context"]
            ctx_mask = d["ctx_mask"]
            B = privileged_actions.shape[0]
            device = privileged_actions.device

            # Update EMA min/max during warmup
            self._update_act_norm(privileged_actions)

            # Normalize teacher actions to [-1, 1]
            a_clean = self._normalize_action(privileged_actions)

            # Sample t ~ U(0, 1), construct a_t, compute velocity
            t = torch.rand(B, device=device)
            eps = torch.randn_like(a_clean)
            a_t = (1 - t.unsqueeze(-1)) * a_clean + t.unsqueeze(-1) * eps
            v_target = eps - a_clean  # constant velocity for linear path

            v_pred = self.decoder(a_t, t, context, ctx_mask)
            flow_loss = F.mse_loss(v_pred, v_target)
            loss_dict["flow"] = flow_loss
            log_dict["flow"] = flow_loss.item()

        if self.prior_reg_coef > 0:
            loss_dict["prior_reg"] = d["h_prior"].pow(2).mean() * self.prior_reg_coef

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
