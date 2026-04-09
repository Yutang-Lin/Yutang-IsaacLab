# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BC-BFM: Naive Transformer Behavior Cloning foundation model.

Simplest BFM variant — no latent, no posterior, no flow.
Just a transformer that takes [h_prior, o_t, frame_0..F-1] and outputs action.
History encoded via MLP → h_prior token.
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
from isaaclab_rl.rsl_rl.networks.cvae_bfm_networks import BFMFrameEncoder, _build_frame_attn_mask
from isaaclab_rl.rsl_rl.networks.transformer import TransformerEncoder


class StudentBCBFMTracker(nn.Module):
    """Naive BC-BFM: transformer behavior cloning with sparse frame commands.

    Tokens: [h_prior(0), o_t(1), frame_0(2), ..., frame_{F-1}(F+1)]
    Action from o_t token (index 1).
    No latent, no posterior, no flow — pure supervised BC.
    """

    is_recurrent = False

    def __init__(self, num_student_obs, num_teacher_obs, num_actions,
                 student_policy_cfg, teacher_policy_ckpt,
                 student_obs_meta, teacher_obs_meta,
                 init_noise_std=0.1, **kwargs):
        if kwargs:
            print("StudentBCBFMTracker: unexpected args: " + str(list(kwargs.keys())))
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

        # -- Config --
        cfg = dict(student_policy_cfg)
        cfg.pop("class_name", None)
        activation_name = cfg.pop("activation", "elu")
        activation = resolve_nn_activation(activation_name)

        latent_dim = cfg.pop("latent_dim", 128)
        for k in ["corr_rank", "history_hidden_dims", "posterior_hidden_dims",
                   "corr_kl_coef", "latent_kl_coef", "prior_reg_coef",
                   "posterior_dropout", "posterior_sigma"]:
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

        # Dedicated RNG for mask/offset sampling (deterministic across variants for fair comparison)
        self._mask_rng = torch.Generator()
        self._mask_rng.manual_seed(cfg.pop("mask_rng_seed", 42))

        tf_d_model = cfg.pop("tf_d_model", 256)
        tf_num_heads = cfg.pop("tf_num_heads", 4)
        tf_num_layers = cfg.pop("tf_num_layers", 2)
        tf_hidden_dim = cfg.pop("tf_hidden_dim", 1024)
        tf_dropout = cfg.pop("tf_dropout", 0.0)
        tf_activation_name = cfg.pop("tf_activation", "gelu")
        if tf_activation_name == "gelu":
            tf_activation = nn.GELU(approximate="tanh")
        else:
            tf_activation = resolve_nn_activation(tf_activation_name)

        # Teacher forcing
        self.teacher_forcing_ratio = cfg.pop("teacher_forcing_ratio", 0.0)
        self.teacher_forcing_noise = cfg.pop("teacher_forcing_noise", 0.1)

        self.latent_dim = latent_dim
        self.num_actions = num_actions

        if cfg:
            print(f"StudentBCBFMTracker: unused keys: {list(cfg.keys())}")

        # -- Networks --
        # History prior: MLP(hp_t) → h_prior
        self.history_prior = _build_mlp(history_proprio_dim, prior_hidden_dims, latent_dim, activation)

        # Shared frame encoder
        self.frame_encoder = BFMFrameEncoder(
            self.num_keypoints, self.dims_per_keypoint, tf_d_model, tf_activation)

        # Token projections for h_prior and o_t
        self.prior_proj = nn.Linear(latent_dim, tf_d_model)
        self.proprio_proj = nn.Linear(current_proprio_dim, tf_d_model)
        self.prior_embed = nn.Parameter(torch.randn(tf_d_model) * 0.02)
        self.proprio_embed = nn.Parameter(torch.randn(tf_d_model) * 0.02)

        # Main transformer
        self.transformer = TransformerEncoder(
            d_model=tf_d_model, num_heads=tf_num_heads, hidden_dim=tf_hidden_dim,
            num_layers=tf_num_layers, dropout=tf_dropout, is_causal=False,
            activation=tf_activation, enable_sdpa=False,
        )

        # Action head from o_t token
        self.action_head = nn.Linear(tf_d_model, num_actions)

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

    def _forward(self, o_t, h_prior, masked_frames, delta_t, frame_mask):
        """Transformer forward: [h_prior, o_t, frames] → action from o_t."""
        B = o_t.shape[0]
        nf = masked_frames.shape[1]

        tok_prior = self.prior_proj(h_prior) + self.prior_embed
        tok_proprio = self.proprio_proj(o_t) + self.proprio_embed
        tok_frames = self.frame_encoder(masked_frames, delta_t)

        tokens = torch.cat([
            tok_prior.unsqueeze(1),
            tok_proprio.unsqueeze(1),
            tok_frames,
        ], dim=1)  # [B, 2+F, d]

        attn_mask = _build_frame_attn_mask(B, nf, frame_mask, n_prefix=2, device=o_t.device)
        out = self.transformer(tokens, attn_mask=attn_mask)

        return self.action_head(out[:, 1])  # o_t position

    # -- Offsets (same as other BFM variants) --

    def _sample_initial_offsets(self, env_ids, device):
        n = env_ids.shape[0]
        nf, K = self.num_frames, self.rollout_mask_num_keypoints
        gen = self._mask_rng
        offsets = torch.empty(n, nf).uniform_(self.min_frame_delta, self.max_frame_delta, generator=gen)
        offsets = (offsets / self.step_dt).round() * self.step_dt
        offsets = offsets.clamp(min=self.step_dt)
        offsets = offsets.sort(dim=1).values
        for i in range(1, nf):
            offsets[:, i] = torch.max(offsets[:, i], offsets[:, i - 1] + self.step_dt)
        self._ep_frame_offsets[env_ids] = offsets.to(device)
        p_active = torch.empty(n).uniform_(*self.frame_p_active_range, generator=gen)
        fm = torch.rand(n, nf, generator=gen) < p_active[:, None]
        all_off = ~fm.any(dim=1)
        if all_off.any():
            fm[all_off, torch.randint(0, nf, (all_off.sum(),), generator=gen)] = True
        self._ep_frame_mask[env_ids] = fm.to(device)
        if K > 0:
            p_clean = torch.empty(n).uniform_(*self.rollout_mask_p_clean_range, generator=gen)
            self._ep_kp_mask[env_ids] = (torch.rand(n, K, generator=gen) < p_clean[:, None]).to(device)
        self._push_offsets_to_env(env_ids)

    def _step_offsets(self):
        offsets, mask = self._ep_frame_offsets, self._ep_frame_mask
        gen = self._mask_rng
        dev = offsets.device
        offsets -= self.step_dt
        consumed = offsets[:, 0] <= 0
        if consumed.any():
            offsets[consumed, :-1] = offsets[consumed, 1:].clone()
            mask[consumed, :-1] = mask[consumed, 1:].clone()
            n = consumed.sum()
            gap = torch.empty(n).uniform_(self.min_frame_delta, self.max_frame_delta, generator=gen)
            gap = (gap / self.step_dt).round() * self.step_dt
            gap = gap.clamp(min=self.step_dt)
            offsets[consumed, -1] = offsets[consumed, -2] + gap.to(dev)
            p = torch.empty(n).uniform_(*self.frame_p_active_range, generator=gen)
            mask[consumed, -1] = (torch.rand(n, generator=gen) < p).to(dev)
        all_off = ~mask.any(dim=1)
        if all_off.any():
            nf = mask.shape[1]
            mask[all_off, torch.randint(0, nf, (all_off.sum(),), generator=gen)] = True
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

    # -- Forward paths --

    def forward(self):
        raise NotImplementedError

    def act(self, observations, *args, **kwargs):
        """Rollout."""
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
        action_mean = self._forward(o_t, h_prior, masked_frames, delta_t, cur_frame_mask)

        std = self.std.expand_as(action_mean)
        self.distribution = Normal(action_mean, std)
        return self.distribution.sample()

    def evaluate(self, teacher_observations, *args, **kwargs):
        with torch.no_grad():
            return self.teacher.act_inference(teacher_observations, *args, **kwargs)

    def act_inference(self, observations, *args, **kwargs):
        """Training: BC forward with random masks. Inference: all active."""
        hp_t, o_t, y_flat, r_t = self._split_obs(observations)
        frames, delta_t = self._parse_condition(y_flat)
        B = frames.shape[0]
        nf, K = self.num_frames, self.rollout_mask_num_keypoints

        if self.training:
            gen = self._mask_rng
            dev = frames.device
            p_active = torch.empty(B).uniform_(*self.frame_p_active_range, generator=gen)
            frame_mask = (torch.rand(B, nf, generator=gen) < p_active[:, None]).to(dev)
            all_off = ~frame_mask.any(dim=1)
            if all_off.any():
                frame_mask[all_off, torch.randint(0, nf, (all_off.sum(),), generator=gen)] = True
            kp_mask = None
            if K > 0:
                p_clean = torch.empty(B).uniform_(*self.rollout_mask_p_clean_range, generator=gen)
                kp_mask = (torch.rand(B, K, generator=gen) < p_clean[:, None]).to(dev)
        else:
            frame_mask = torch.ones(B, nf, dtype=torch.bool, device=frames.device)
            kp_mask = None

        masked_frames = self._apply_masks(frames, frame_mask, kp_mask)
        h_prior = self.history_prior(hp_t)
        action_mean = self._forward(o_t, h_prior, masked_frames, delta_t, frame_mask)
        return action_mean

    def extra_loss(self, **kwargs):
        return {}, {}

    def pre_train(self):
        pass

    def after_train(self):
        pass

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
