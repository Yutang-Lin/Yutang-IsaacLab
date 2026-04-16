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
from isaaclab_rl.rsl_rl.networks.lfm_bfm_networks import LatentFlowDecoder, LFMActionDecoder, LFMReconDecoder


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
        activation_name = cfg.pop("activation", "gelu")
        if activation_name == "gelu":
            activation = nn.GELU(approximate="tanh")
        else:
            activation = resolve_nn_activation(activation_name)

        latent_dim = cfg.pop("latent_dim", 64)
        prior_hidden_dims = cfg.pop("prior_hidden_dims", [512, 256])
        proprio_hidden_dims = cfg.pop("proprio_hidden_dims", [512, 256])
        posterior_hidden_dims = cfg.pop("posterior_hidden_dims", [512, 256])
        recon_hidden_dims = cfg.pop("recon_hidden_dims", [256, 256])
        # Discard legacy CVAE keys
        for k in ["corr_rank", "history_hidden_dims",
                   "corr_kl_coef", "latent_kl_coef", "prior_reg_coef"]:
            cfg.pop(k, None)

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
        # Lazily moved to GPU on first use to avoid CPU→GPU transfers every step.
        self._mask_rng_seed = cfg.pop("mask_rng_seed", 42)
        self._mask_rng = None  # initialized lazily in _get_rng(device)

        tf_d_model = cfg.pop("tf_d_model", 256)
        tf_num_heads = cfg.pop("tf_num_heads", 4)
        tf_hidden_dim = cfg.pop("tf_hidden_dim", 512)
        tf_dropout = cfg.pop("tf_dropout", 0.0)
        tf_activation_name = cfg.pop("tf_activation", "gelu")
        if tf_activation_name == "gelu":
            tf_activation = nn.GELU(approximate="tanh")
        else:
            tf_activation = resolve_nn_activation(tf_activation_name)

        cfg.pop("encoder_num_layers", None)  # unused (no shared encoder)
        decoder_num_layers = cfg.pop("decoder_num_layers", 2)
        decoder_hidden_dims = cfg.pop("decoder_hidden_dims", None)  # MLP hidden dims; None = use tf_hidden_dim
        flow_num_layers = cfg.pop("flow_num_layers", 2)
        self.ode_steps = cfg.pop("ode_steps", 10)
        self.posterior_sigma = cfg.pop("posterior_sigma", 0.1)
        self.boundary_coef = cfg.pop("boundary_coef", 1.0)
        self.spread_coef = cfg.pop("spread_coef", 1e-2)
        self.grad_penalty_coef = cfg.pop("grad_penalty_coef", 0.0)
        self.use_mean_flow = cfg.pop("use_mean_flow", False)
        self.mf_propagation_ratio = cfg.pop("mf_propagation_ratio", 0.25)
        use_proj_norm = cfg.pop("use_proj_norm", False)
        cfg.pop("full_context", None)  # deprecated, ignored
        cfg.pop("posterior_dropout", None)  # unused

        self.latent_dim = latent_dim
        self.num_actions = num_actions

        # Teacher forcing: fraction of rollout envs that use teacher action + noise
        self.teacher_forcing_ratio = cfg.pop("teacher_forcing_ratio", 0.0)
        self.teacher_forcing_noise = cfg.pop("teacher_forcing_noise", 0.1)

        # z_prev: pass previous step's z_t to the flow decoder
        self.use_prev_z = cfg.pop("use_prev_z", False)

        # Legacy args (kept for checkpoint compat, no longer used)
        cfg.pop("history_dropout_prob", None)
        cfg.pop("history_sigma", None)

        # Reconstruction decoder: z_t → posterior condition
        self.recon_coef = cfg.pop("recon_coef", 0.0)

        # Two-stage training: stage1 trains posterior/decoder, stage2 trains flow
        self.flow_start_step = cfg.pop("flow_start_step", 0)  # 0 = no staging, train all together
        self._distill_step = 0
        self._stage = 1 if self.flow_start_step > 0 else 0  # 0=joint, 1=posterior, 2=flow

        if cfg:
            print(f"StudentLFMBFMTracker: unused keys: {list(cfg.keys())}")

        # -- Networks (separate encoders, no shared transformer) --

        # Stage 1 networks (trained in stage 1, frozen in stage 2):
        self.o_t_encoder = _build_mlp(current_proprio_dim, proprio_hidden_dims, tf_d_model, activation)
        self.posterior = _build_mlp(keybody_dim, posterior_hidden_dims, latent_dim, activation)
        self.history_encoder = _build_mlp(history_proprio_dim, prior_hidden_dims, tf_d_model, activation)

        # Stage 2 networks (frozen in stage 1, trained in stage 2):
        self.frame_encoder = BFMFrameEncoder(
            self.num_keypoints, self.dims_per_keypoint, tf_d_model, tf_activation)

        # Latent flow decoder: noised z cross-attends to [o_t_enc, h_enc, frames] → velocity
        self.latent_flow = LatentFlowDecoder(
            latent_dim, tf_d_model, tf_num_heads, tf_hidden_dim,
            flow_num_layers, tf_dropout, tf_activation,
            use_proj_norm=use_proj_norm,
            use_prev_z=self.use_prev_z)

        # Action decoder: concat(z_proj, o_t_enc, h_enc) → MLP → action
        if decoder_hidden_dims is not None:
            # Use explicit hidden dims list for MLP layers
            self.action_decoder = LFMActionDecoder(
                latent_dim, tf_d_model, hidden_dims=decoder_hidden_dims,
                num_actions=num_actions, activation=tf_activation)
        else:
            self.action_decoder = LFMActionDecoder(
                latent_dim, tf_d_model, num_layers=decoder_num_layers,
                hidden_dim=tf_hidden_dim, num_actions=num_actions,
                activation=tf_activation)

        # Reconstruction decoder: z_t → posterior condition (optional)
        self.recon_decoder = None
        if self.recon_coef > 0:
            self.recon_decoder = LFMReconDecoder(
                latent_dim, keybody_dim, recon_hidden_dims, tf_activation)

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
        self._ep_ode_noise = None  # per-env fixed ODE starting noise, resampled on reset
        self._ep_prev_z = None  # per-env previous step's latent z_t (for use_prev_z)
        self._rollout_z_list = []  # collects z_t per rollout step
        self._rollout_z_buffer = None  # [T, B, D] stacked rollout z for training
        self._train_step = 0  # current step index during training

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
        """Return a torch.Generator on the given device, lazily created."""
        if self._mask_rng is None or self._mask_rng.device != device:
            self._mask_rng = torch.Generator(device=device)
            self._mask_rng.manual_seed(self._mask_rng_seed)
        return self._mask_rng

    def _build_flow_context(self, o_t_enc, h_enc, frame_tokens, frame_mask):
        """Build flow decoder context: [o_t_enc, h_enc, frame_0, ..., frame_F-1] with masking."""
        B, nf, _ = frame_tokens.shape
        tokens = torch.cat([
            o_t_enc.unsqueeze(1),   # [B, 1, d]
            h_enc.unsqueeze(1),     # [B, 1, d]
            frame_tokens,           # [B, F, d]
        ], dim=1)  # [B, 2+F, d]
        mask = torch.ones(B, 2 + nf, dtype=torch.bool, device=o_t_enc.device)
        mask[:, 2:] = frame_mask
        return tokens, mask

    def _apply_masks(self, frames, frame_mask=None, kp_mask=None):
        B, nf, KD = frames.shape
        K, D = self.rollout_mask_num_keypoints, self.dims_per_keypoint
        if kp_mask is not None and K > 0:
            frames = (frames.view(B, nf, K, D) * kp_mask[:, None, :, None].float()).flatten(2)
        if frame_mask is not None:
            frames = frames * frame_mask.unsqueeze(-1).float()
        return frames

    # -- Offsets (same as other BFM variants) --

    def _sample_count_mask(self, n, total, gen, device, min_active=0):
        """Sample masks by first choosing a count from U(min_active, total), then randomly selecting which slots are active.

        Args:
            n: batch size
            total: number of slots (frames or keypoints)
            gen: torch.Generator
            device: torch device
            min_active: minimum number of active slots (0 allows all-masked)

        Returns:
            mask: [n, total] bool tensor
        """
        counts = torch.randint(min_active, total + 1, (n,), device=device, generator=gen)
        mask = torch.zeros(n, total, dtype=torch.bool, device=device)
        for i in range(n):
            c = counts[i].item()
            if c > 0:
                perm = torch.randperm(total, device=device, generator=gen)[:c]
                mask[i, perm] = True
        return mask

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
        # Frame mask: U(1, nf) count, random selection — rollout guarantees >= 1
        self._ep_frame_mask[env_ids] = self._sample_count_mask(
            n, nf, gen, device, min_active=1)
        # Keypoint mask: U(1, K) count, random selection — rollout guarantees >= 1
        if K > 0:
            self._ep_kp_mask[env_ids] = self._sample_count_mask(
                n, K, gen, device, min_active=1)
        self._push_offsets_to_env(env_ids)

    def _step_offsets(self):
        offsets, mask = self._ep_frame_offsets, self._ep_frame_mask
        dev = offsets.device
        gen = self._get_rng(dev)
        offsets -= self.step_dt
        consumed = offsets[:, 0] <= 0
        if consumed.any():
            # Remember consumed frame's active status before shifting
            consumed_was_active = mask[consumed, 0].clone()
            offsets[consumed, :-1] = offsets[consumed, 1:].clone()
            mask[consumed, :-1] = mask[consumed, 1:].clone()
            n = consumed.sum()
            gap = torch.empty(n, device=dev).uniform_(self.min_frame_delta, self.max_frame_delta, generator=gen)
            gap = (gap / self.step_dt).round() * self.step_dt
            gap = gap.clamp(min=self.step_dt)
            offsets[consumed, -1] = offsets[consumed, -2] + gap
            # New frame inherits consumed frame's active status (preserves count)
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

    def _ode_sample_latent(self, context, ctx_mask, B, device, z_prev=None):
        """K-step Euler ODE in latent space with KV cache.

        Each env uses a fixed per-episode noise as ODE starting point.
        This keeps each env in a consistent mode throughout its episode.
        """
        if self._ep_ode_noise is None or self._ep_ode_noise.shape[0] != B:
            self._ep_ode_noise = torch.randn(B, self.latent_dim, device=device)
        z = self._ep_ode_noise[:B].clone()
        # z_prev is baked into the KV cache as an extra context token
        kv_cache, ctx_mask = self.latent_flow.build_kv_cache(context, ctx_mask, z_prev=z_prev, z_noised_like=z)
        dt = 1.0 / self.ode_steps
        for k in range(self.ode_steps):
            t = 1.0 - k * dt
            t_tensor = torch.full((B,), t, device=device)
            if self.use_mean_flow:
                r_tensor = torch.full((B,), max(t - dt, 0.0), device=device)
                v = self.latent_flow.forward_cached(z, t_tensor, kv_cache, ctx_mask, r=r_tensor)
            else:
                v = self.latent_flow.forward_cached(z, t_tensor, kv_cache, ctx_mask)
            z = z - dt * v
        return z

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
                self._ep_ode_noise = torch.randn(B, self.latent_dim, device=observations.device)
                self._ep_prev_z = torch.zeros(B, self.latent_dim, device=observations.device)
                self._sample_initial_offsets(torch.arange(B, device=observations.device), observations.device)
        else:
            with torch.inference_mode(False):
                self._step_offsets()

        masked_frames = self._apply_masks(frames, self._ep_frame_mask, self._ep_kp_mask)
        cur_frame_mask = self._ep_frame_mask[:B]

        # Encode inputs separately
        o_t_enc = self.o_t_encoder(o_t)  # [B, d_model]
        h_enc = self.history_encoder(hp_t)  # [B, d_model]

        if self._stage == 1:
            # Stage 1: rollout with posterior z (privileged, no ODE)
            z_t = self.posterior(r_t)
        else:
            # Stage 0 or 2: rollout with flow ODE
            frame_tokens = self.frame_encoder(masked_frames, delta_t) # [B, F, d_model]
            flow_context, flow_mask = self._build_flow_context(
                o_t_enc, h_enc, frame_tokens, cur_frame_mask)
            z_prev = self._ep_prev_z[:B] if self.use_prev_z else None
            z_t = self._ode_sample_latent(flow_context, flow_mask, B, observations.device, z_prev=z_prev)

        # Store z_t as prev_z for next step
        if self.use_prev_z and self._ep_prev_z is not None:
            self._ep_prev_z[:B] = z_t.detach()
            self._rollout_z_list.append(z_t.detach().clone())

        action_mean = self.action_decoder(z_t, o_t_enc, h_enc)

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

        # Always encode o_t and history
        o_t_enc = self.o_t_encoder(o_t)    # [B, d_model]
        h_enc = self.history_encoder(hp_t)  # [B, d_model]

        if self.compute_latent_loss and r_t.shape[-1] > 0:
            # Enable grad on r_t if gradient penalty is active
            if self.grad_penalty_coef > 0:
                r_t = r_t.detach().requires_grad_(True)

            # Posterior: MLP r_t → z_t
            z_t = self.posterior(r_t)

            # Gradient penalty
            grad_penalty = None
            if self.grad_penalty_coef > 0:
                grads = torch.autograd.grad(
                    z_t.sum(), r_t, create_graph=True, retain_graph=True
                )[0]
                grad_penalty = grads.pow(2).mean()

            # --- Flow loss: only in stage 2 (or stage 0 = joint) ---
            flow_loss = None
            if self._stage != 1:
                dev = frames.device
                gen = self._get_rng(dev)
                # Training: U(0, nf) and U(0, K) — can be all-zero
                frame_mask = self._sample_count_mask(B, nf, gen, dev, min_active=0)
                kp_mask = None
                if K > 0:
                    kp_mask = self._sample_count_mask(B, K, gen, dev, min_active=0)

                masked_frames = self._apply_masks(frames, frame_mask, kp_mask)
                frame_tokens = self.frame_encoder(masked_frames, delta_t)
                flow_context, flow_mask = self._build_flow_context(o_t_enc, h_enc, frame_tokens, frame_mask)

                z_prev_train = None
                if self.use_prev_z and self._rollout_z_buffer is not None:
                    if self._train_step > 0:
                        z_prev_train = self._rollout_z_buffer[self._train_step - 1]

                t = torch.rand(B, device=dev)
                eps = torch.randn(B, self.latent_dim, device=dev)
                z_noised = (1 - t.unsqueeze(-1)) * z_t + t.unsqueeze(-1) * eps
                v_t = eps - z_t

                if self.use_mean_flow:
                    from torch.func import jvp as func_jvp
                    from torch.nn.attention import sdpa_kernel, SDPBackend

                    prop_mask = torch.rand(B, device=dev) < self.mf_propagation_ratio
                    r = t.clone()
                    prop_idx = prop_mask.nonzero(as_tuple=True)[0]
                    if prop_idx.numel() > 0:
                        r[prop_idx] = torch.rand(prop_idx.numel(), device=dev) * t[prop_idx]

                    v_pred = self.latent_flow(z_noised, t, flow_context, flow_mask, r=r, z_prev=z_prev_train)

                    du_dt = torch.zeros_like(v_pred)
                    if prop_idx.numel() > 0:
                        zp_sub = z_prev_train[prop_idx] if z_prev_train is not None else None
                        def _flow_fn(z, t_, r_):
                            return self.latent_flow(z, t_, flow_context[prop_idx], flow_mask[prop_idx], r=r_, z_prev=zp_sub)
                        with sdpa_kernel(SDPBackend.MATH):
                            _, du_sub = func_jvp(
                                _flow_fn,
                                (z_noised[prop_idx], t[prop_idx], r[prop_idx]),
                                (v_t[prop_idx], torch.ones(prop_idx.numel(), device=dev),
                                 torch.zeros(prop_idx.numel(), device=dev)),
                            )
                        du_dt[prop_idx] = du_sub

                    t_minus_r = (t - r).unsqueeze(-1)
                    u_target = v_t - t_minus_r * du_dt
                    flow_loss = F.mse_loss(v_pred, u_target.detach())
                else:
                    v_pred = self.latent_flow(z_noised, t, flow_context, flow_mask, z_prev=z_prev_train)
                    flow_loss = F.mse_loss(v_pred, v_t)

                if self.use_prev_z and self._rollout_z_buffer is not None:
                    self._train_step = (self._train_step + 1) % self._rollout_z_buffer.shape[0]

            # --- Recon loss: only in stage 1 (or stage 0 = joint) ---
            recon_loss = None
            if self._stage != 2 and self.recon_decoder is not None:
                recon_pred = self.recon_decoder(z_t)
                recon_loss = F.mse_loss(recon_pred, r_t.detach())

            # Boundary loss
            boundary_loss = F.relu(z_t.abs() - 1.0).pow(2).mean()

            # Add fixed noise for tolerance area
            z_posterior = z_t + self.posterior_sigma * torch.randn_like(z_t)

            # Spread loss
            spread_loss = None
            if self.spread_coef > 0:
                target_var = 0.1
                per_dim_var = z_t.var(dim=0)
                spread_loss = F.relu(target_var - per_dim_var).mean()

            self._save_dict = {
                "flow_loss": flow_loss,
                "boundary_loss": boundary_loss,
                "spread_loss": spread_loss,
                "grad_penalty": grad_penalty,
                "recon_loss": recon_loss,
            }

            action_mean = self.action_decoder(z_posterior, o_t_enc, h_enc)
            return action_mean
        else:
            # Inference: latent ODE (needs frame_encoder + flow)
            frame_mask = torch.ones(B, nf, dtype=torch.bool, device=frames.device)
            masked_frames = self._apply_masks(frames, frame_mask, None)
            frame_tokens = self.frame_encoder(masked_frames, delta_t)
            flow_context, flow_mask = self._build_flow_context(o_t_enc, h_enc, frame_tokens, frame_mask)
            z_t = self._ode_sample_latent(flow_context, flow_mask, B, frames.device)
            action_mean = self.action_decoder(z_t, o_t_enc, h_enc)
            return action_mean

    def extra_loss(self, **kwargs):
        if not self._save_dict:
            return {}, {}

        d = self._save_dict
        loss_dict = {}
        log_dict = {}

        # Stage 1: log flow loss but don't backprop through it
        # Stage 2: only flow loss (posterior/decoder frozen)
        # Flow loss: only computed in stage 2 (or stage 0 = joint)
        if d["flow_loss"] is not None:
            if self._stage == 2 or self._stage == 0:
                loss_dict["flow"] = d["flow_loss"]
            log_dict["flow"] = d["flow_loss"].item()

        # Stage 1 losses: boundary, spread, grad_penalty, recon
        if self._stage != 2:
            loss_dict["boundary"] = d["boundary_loss"] * self.boundary_coef
            log_dict["boundary"] = d["boundary_loss"].item()

            if d["spread_loss"] is not None:
                loss_dict["spread"] = d["spread_loss"] * self.spread_coef
                log_dict["spread"] = d["spread_loss"].item()

            if d["grad_penalty"] is not None:
                loss_dict["grad_penalty"] = d["grad_penalty"] * self.grad_penalty_coef
                log_dict["grad_penalty"] = d["grad_penalty"].item()

            if d["recon_loss"] is not None:
                loss_dict["recon"] = d["recon_loss"] * self.recon_coef
                log_dict["recon"] = d["recon_loss"].item()
        else:
            # Stage 2: log auxiliary losses for monitoring (no backprop)
            log_dict["boundary"] = d["boundary_loss"].item()
            if d["spread_loss"] is not None:
                log_dict["spread"] = d["spread_loss"].item()

        self._save_dict = {}
        return dict(loss_dict), dict(log_dict)

    def _enter_stage(self, stage):
        """Freeze/unfreeze parameters for the given training stage."""
        self._stage = stage
        if stage == 1:
            # Stage 1: train o_t_encoder, posterior, history_encoder, action_decoder, recon
            # Freeze: latent_flow, frame_encoder
            for m in [self.o_t_encoder, self.posterior, self.history_encoder, self.action_decoder]:
                for p in m.parameters():
                    p.requires_grad = True
            if self.recon_decoder is not None:
                for p in self.recon_decoder.parameters():
                    p.requires_grad = True
            self.std.requires_grad = True
            for m in [self.latent_flow, self.frame_encoder]:
                for p in m.parameters():
                    p.requires_grad = False
            print(f"[INFO] LFM-BFM: entering stage 1 (o_t+posterior+history+action, flow/frames frozen)")
        elif stage == 2:
            # Stage 2: train latent_flow, frame_encoder
            # Freeze: o_t_encoder, posterior, history_encoder, action_decoder, recon
            for m in [self.latent_flow, self.frame_encoder]:
                for p in m.parameters():
                    p.requires_grad = True
            for m in [self.o_t_encoder, self.posterior, self.history_encoder, self.action_decoder]:
                for p in m.parameters():
                    p.requires_grad = False
            if self.recon_decoder is not None:
                for p in self.recon_decoder.parameters():
                    p.requires_grad = False
            self.std.requires_grad = False
            print(f"[INFO] LFM-BFM: entering stage 2 (flow+frames, o_t+posterior+history+action frozen)")

    def pre_train(self):
        self.compute_latent_loss = True
        # Stack rollout z into buffer for training z_prev lookup
        if self._rollout_z_list:
            self._rollout_z_buffer = torch.stack(self._rollout_z_list)  # [T, B, D]
            self._rollout_z_list = []
        self._train_step = 0

        # Stage transition
        self._distill_step += 1
        if self.flow_start_step > 0:
            if self._distill_step == 1:
                self._enter_stage(1)
            elif self._distill_step == self.flow_start_step:
                self._enter_stage(2)

    def after_train(self):
        self.compute_latent_loss = False
        self._save_dict = {}
        self._rollout_z_buffer = None

    def reset(self, dones=None, hidden_states=None):
        if dones is not None and dones.any():
            env_ids = dones.bool().flatten().nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() > 0:
                if self._ep_frame_mask is not None:
                    with torch.inference_mode(False):
                        self._sample_initial_offsets(env_ids, self._ep_frame_mask.device)
                if self._ep_ode_noise is not None:
                    self._ep_ode_noise[env_ids] = torch.randn(env_ids.shape[0], self.latent_dim, device=self._ep_ode_noise.device)
                if self._ep_prev_z is not None:
                    self._ep_prev_z[env_ids] = 0.0

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
