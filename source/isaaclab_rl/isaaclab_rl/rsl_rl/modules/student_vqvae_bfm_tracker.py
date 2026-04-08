# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VQ-VAE BFM: Foundation model with vector-quantized latent space.

Replaces CVAE-BFM's Gaussian posterior with VQ codebook:
  - Posterior: cross-attention → continuous embedding → VQ quantize → e_q
  - Prior: 1-layer transformer predicting codebook index from history + frames
  - Decoder: [proprio, h_prior, e_q, frames] (same structure as CVAE-BFM)
  - Rollout: prior predicts codebook index (argmax), look up e_q
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
from isaaclab_rl.rsl_rl.networks.cvae_bfm_networks import BFMFrameEncoder, CVAEBFMDecoder
from isaaclab_rl.rsl_rl.networks.vqvae_bfm_networks import VQCodebook, VQBFMPosterior, VQBFMPrior


class StudentVQVAEBFMTracker(nn.Module):
    """VQ-VAE BFM student policy for distillation."""

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
            print("StudentVQVAEBFMTracker.__init__ got unexpected args: " + str(list(kwargs.keys())))
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

        latent_dim = cfg.pop("latent_dim", 64)
        num_codes = cfg.pop("num_codes", 512)
        ema_decay = cfg.pop("ema_decay", 0.99)
        dead_code_threshold = cfg.pop("dead_code_threshold", 100)
        cfg.pop("corr_rank", None)
        cfg.pop("history_hidden_dims", None)
        cfg.pop("posterior_hidden_dims", None)
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
        tf_num_layers = cfg.pop("tf_num_layers", 2)
        tf_hidden_dim = cfg.pop("tf_hidden_dim", 512)
        tf_dropout = cfg.pop("tf_dropout", 0.0)
        tf_activation_name = cfg.pop("tf_activation", "gelu")
        if tf_activation_name == "gelu":
            tf_activation = nn.GELU(approximate="tanh")
        else:
            tf_activation = resolve_nn_activation(tf_activation_name)

        # Loss coefficients
        self.commit_coef = cfg.pop("commit_coef", 0.25)
        self.prior_ce_coef = cfg.pop("prior_ce_coef", 1.0)
        cfg.pop("prior_reg_coef", None)  # unused in VQ-VAE
        self.posterior_dropout = cfg.pop("posterior_dropout", 0.5)

        self.latent_dim = latent_dim
        self.num_codes = num_codes
        self.num_actions = num_actions

        if cfg:
            print(f"StudentVQVAEBFMTracker: unused config keys: {list(cfg.keys())}")

        # -- Build sub-networks --
        # History prior MLP: hp_t → h_prior (deterministic)
        self.history_prior = _build_mlp(history_proprio_dim, prior_hidden_dims, latent_dim, activation)

        # Shared frame encoder
        self.frame_encoder = BFMFrameEncoder(
            num_keypoints=self.num_keypoints,
            dims_per_keypoint=self.dims_per_keypoint,
            d_model=tf_d_model,
            activation=tf_activation,
        )

        # VQ codebook
        self.codebook = VQCodebook(num_codes, latent_dim, ema_decay, dead_code_threshold)

        # VQ Posterior: 2-layer cross-attention (keybody queries, frames KV)
        self.posterior = VQBFMPosterior(
            keybody_dim=keybody_dim,
            latent_dim=latent_dim,
            d_model=tf_d_model,
            frame_encoder=self.frame_encoder,
            num_heads=tf_num_heads,
            hidden_dim=tf_hidden_dim,
            num_layers=2,
            activation=tf_activation,
        )

        # VQ Prior: 2-layer cross-attention (o_t queries, [h_prior, prev_e_q, frames] KV)
        self.prior_predictor = VQBFMPrior(
            proprio_dim=current_proprio_dim,
            h_dim=latent_dim,
            latent_dim=latent_dim,
            num_codes=num_codes,
            d_model=tf_d_model,
            frame_encoder=self.frame_encoder,
            num_heads=tf_num_heads,
            hidden_dim=tf_hidden_dim,
            num_layers=2,
            activation=tf_activation,
        )

        # Decoder: same as CVAE-BFM with shared frame encoder
        self.action_decoder = CVAEBFMDecoder(
            proprio_dim=current_proprio_dim,
            latent_dim=latent_dim,
            max_frames=self.num_frames,
            d_model=tf_d_model,
            frame_encoder=self.frame_encoder,
            num_heads=tf_num_heads,
            hidden_dim=tf_hidden_dim,
            num_layers=tf_num_layers,
            num_actions=num_actions,
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
        if num_teacher_obs != teacher_policy_args[0]:
            print(f"[WARN] Teacher obs mismatch: env={num_teacher_obs}, ckpt={teacher_policy_args[0]}")
        self.teacher: ActorCritic = teacher_policy_class(*teacher_policy_args, **teacher_policy_cfg)
        self.teacher.load_state_dict(teacher_ckpt["model_state_dict"], strict=True)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.loaded_teacher = True

        print(f"StudentVQVAEBFMTracker: latent={latent_dim}, codes={num_codes}, "
              f"frames={self.num_frames}, keybody={keybody_dim}")

        # -- Action noise --
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

        # -- State --
        self.compute_latent_loss = False
        self._ep_frame_mask = None
        self._ep_kp_mask = None
        self._ep_frame_offsets = None
        self._env_ref = None
        # Auto-regressive: previous step's e_q per env (zero at episode start)
        self._prev_e_q: torch.Tensor | None = None  # [N, latent_dim]

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
        return y[:, :, :-1], y[:, :, -1]  # frames [B,F,K*D], delta_t [B,F]

    def _apply_masks(self, frames, frame_mask=None, kp_mask=None):
        B, nf, KD = frames.shape
        K, D = self.rollout_mask_num_keypoints, self.dims_per_keypoint
        if kp_mask is not None and K > 0:
            frames = (frames.view(B, nf, K, D) * kp_mask[:, None, :, None].float()).flatten(2)
        if frame_mask is not None:
            frames = frames * frame_mask.unsqueeze(-1).float()
        return frames

    # -- Episodic offset management (same as CVAE-BFM) --

    def _sample_initial_offsets(self, env_ids, device):
        n = env_ids.shape[0]
        n_frames, K = self.num_frames, self.rollout_mask_num_keypoints
        offsets = torch.empty(n, n_frames, device=device).uniform_(self.min_frame_delta, self.max_frame_delta)
        self._ep_frame_offsets[env_ids] = offsets.sort(dim=1).values
        p_active = torch.empty(n, device=device).uniform_(*self.frame_p_active_range)
        fm = torch.rand(n, n_frames, device=device) < p_active[:, None]
        all_off = ~fm.any(dim=1)
        if all_off.any():
            fm[all_off, torch.randint(0, n_frames, (all_off.sum(),), device=device)] = True
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
            last = offsets[consumed, -2]
            offsets[consumed, -1] = last + torch.empty(n, device=offsets.device).uniform_(self.min_frame_delta, self.max_frame_delta)
            p = torch.empty(n, device=offsets.device).uniform_(*self.frame_p_active_range)
            mask[consumed, -1] = torch.rand(n, device=offsets.device) < p

        # Ensure at least 1 frame active after shift (prevents all-masked → NaN in cross-attn)
        all_off = ~mask.any(dim=1)
        if all_off.any():
            n_frames = mask.shape[1]
            rand_slot = torch.randint(0, n_frames, (all_off.sum(),), device=mask.device)
            mask[all_off, rand_slot] = True

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
        """Rollout: use prior predictor to select codebook entry."""
        hp_t, o_t, y_flat, r_t = self._split_obs(observations)
        frames, delta_t = self._parse_condition(y_flat)
        B = observations.shape[0]
        n_frames, K = self.num_frames, self.rollout_mask_num_keypoints

        if self._ep_frame_mask is None or self._ep_frame_mask.shape[0] != B:
            with torch.inference_mode(False):
                self._ep_frame_mask = torch.ones(B, n_frames, dtype=torch.bool, device=observations.device)
                self._ep_kp_mask = torch.ones(B, K, dtype=torch.bool, device=observations.device) if K > 0 else None
                self._ep_frame_offsets = torch.zeros(B, n_frames, device=observations.device)
                self._sample_initial_offsets(torch.arange(B, device=observations.device), observations.device)
        else:
            with torch.inference_mode(False):
                self._step_offsets()

        masked_frames = self._apply_masks(frames, self._ep_frame_mask, self._ep_kp_mask)
        cur_frame_mask = self._ep_frame_mask[:B]

        # Init prev_e_q buffer if needed
        if self._prev_e_q is None or self._prev_e_q.shape[0] != B:
            with torch.inference_mode(False):
                self._prev_e_q = torch.zeros(B, self.latent_dim, device=observations.device)

        # Prior: predict codebook index + enriched tokens for decoder
        h_prior = self.history_prior(hp_t)
        logits, o_t_enc, h_prior_enc = self.prior_predictor(o_t, h_prior, self._prev_e_q, masked_frames, delta_t, cur_frame_mask)
        indices = logits.argmax(dim=-1)  # [B]
        e_q = self.codebook.embedding(indices)  # [B, latent_dim]

        # Update prev_e_q for next step
        with torch.inference_mode(False):
            self._prev_e_q = e_q.detach().clone()

        # Debug NaN tracing
        if torch.isnan(o_t_enc).any() or torch.isnan(h_prior_enc).any():
            print(f"[NaN DEBUG] prior output: o_t_enc_nan={torch.isnan(o_t_enc).sum().item()}, "
                  f"h_prior_enc_nan={torch.isnan(h_prior_enc).sum().item()}, "
                  f"e_q_nan={torch.isnan(e_q).sum().item()}, "
                  f"hp_t_nan={torch.isnan(hp_t).sum().item()}, "
                  f"o_t_nan={torch.isnan(o_t).sum().item()}, "
                  f"h_prior_nan={torch.isnan(h_prior).sum().item()}, "
                  f"prev_e_q_nan={torch.isnan(self._prev_e_q).sum().item()}, "
                  f"masked_frames_nan={torch.isnan(masked_frames).sum().item()}, "
                  f"delta_t_nan={torch.isnan(delta_t).sum().item()}")

        action_mean = self.action_decoder(o_t_enc, h_prior_enc, e_q, masked_frames, delta_t, cur_frame_mask, pre_encoded=True)

        if torch.isnan(action_mean).any():
            print(f"[NaN DEBUG] decoder output NaN: {torch.isnan(action_mean).sum().item()}/{action_mean.numel()}")

        std = self.std.expand_as(action_mean)
        self.distribution = Normal(action_mean, std)
        return self.distribution.sample()

    def evaluate(self, teacher_observations, *args, **kwargs):
        with torch.no_grad():
            return self.teacher.act_inference(teacher_observations, *args, **kwargs)

    def act_inference(self, observations, *args, **kwargs):
        """Training: posterior quantize + prior CE. Inference: prior argmax."""
        hp_t, o_t, y_flat, r_t = self._split_obs(observations)
        frames, delta_t = self._parse_condition(y_flat)
        B = frames.shape[0]
        n_frames = self.num_frames
        K = self.rollout_mask_num_keypoints

        if self.compute_latent_loss:
            # Random masks
            p_active = torch.empty(B, device=frames.device).uniform_(*self.frame_p_active_range)
            frame_mask = torch.rand(B, n_frames, device=frames.device) < p_active[:, None]
            all_off = ~frame_mask.any(dim=1)
            if all_off.any():
                frame_mask[all_off, torch.randint(0, n_frames, (all_off.sum(),), device=frames.device)] = True
            kp_mask = None
            if K > 0:
                p_clean = torch.empty(B, device=frames.device).uniform_(*self.rollout_mask_p_clean_range)
                kp_mask = torch.rand(B, K, device=frames.device) < p_clean[:, None]
        else:
            frame_mask = torch.ones(B, n_frames, dtype=torch.bool, device=frames.device)
            kp_mask = None

        masked_frames = self._apply_masks(frames, frame_mask, kp_mask)

        h_prior = self.history_prior(hp_t)

        if self.compute_latent_loss and r_t.shape[-1] > 0:
            # Posterior: continuous → quantize
            z_e = self.posterior(r_t, masked_frames, delta_t, frame_mask)
            e_q, vq_indices, commit_loss = self.codebook.quantize(z_e)

            # Prior: predict codebook index + enriched tokens
            prev_e_q_zero = torch.zeros(B, self.latent_dim, device=frames.device)
            logits, o_t_enc, h_prior_enc = self.prior_predictor(o_t, h_prior, prev_e_q_zero, masked_frames, delta_t, frame_mask)
            prior_ce = F.cross_entropy(logits, vq_indices.detach())

            # Posterior dropout
            if self.posterior_dropout > 0:
                drop_mask = torch.rand(B, device=e_q.device) < self.posterior_dropout
                if drop_mask.any():
                    prior_indices = logits.detach().argmax(dim=-1)
                    e_q_prior = self.codebook.embedding(prior_indices)
                    e_q = torch.where(drop_mask.unsqueeze(-1), e_q_prior, e_q)

            self._save_dict = {
                "commit_loss": commit_loss,
                "prior_ce": prior_ce,
                "logits": logits,
                "vq_indices": vq_indices,
            }
        else:
            # Inference: prior argmax + enriched tokens
            prev_e_q_zero = torch.zeros(B, self.latent_dim, device=frames.device)
            logits, o_t_enc, h_prior_enc = self.prior_predictor(o_t, h_prior, prev_e_q_zero, masked_frames, delta_t, frame_mask)
            indices = logits.argmax(dim=-1)
            e_q = self.codebook.embedding(indices)

        # Decoder uses enriched o_t and h_prior from prior (LayerNorm'd, d_model-dim)
        action_mean = self.action_decoder(o_t_enc, h_prior_enc, e_q, masked_frames, delta_t, frame_mask, pre_encoded=True)
        return action_mean

    def extra_loss(self, **kwargs):
        if not self._save_dict:
            return {}, {}

        d = self._save_dict
        loss_dict = {}
        log_dict = {}

        loss_dict["commit"] = d["commit_loss"] * self.commit_coef
        log_dict["commit"] = d["commit_loss"].item()

        loss_dict["prior_ce"] = d["prior_ce"] * self.prior_ce_coef
        log_dict["prior_ce"] = d["prior_ce"].item()

        # Logging: add zero-weight entries to loss_dict so the runner iterates them,
        # then value_dict provides the actual scalar values
        with torch.no_grad():
            unique_codes = d["vq_indices"].unique().numel()
            log_dict["codebook_usage"] = unique_codes / self.num_codes
            loss_dict["codebook_usage"] = torch.tensor(0.0, device=d["vq_indices"].device)

            prior_pred = d["logits"].argmax(dim=-1)
            log_dict["prior_accuracy"] = (prior_pred == d["vq_indices"]).float().mean().item()
            loss_dict["prior_accuracy"] = torch.tensor(0.0, device=d["vq_indices"].device)

            dead = (self.codebook._steps_since_used >= self.codebook.dead_code_threshold).sum().item()
            log_dict["dead_codes"] = dead
            loss_dict["dead_codes"] = torch.tensor(0.0, device=d["vq_indices"].device)

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
            if env_ids.numel() > 0:
                if self._ep_frame_mask is not None:
                    with torch.inference_mode(False):
                        self._sample_initial_offsets(env_ids, self._ep_frame_mask.device)
                # Zero prev_e_q for reset envs (episode start = no previous code)
                if self._prev_e_q is not None:
                    with torch.inference_mode(False):
                        self._prev_e_q[env_ids] = 0.0

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
