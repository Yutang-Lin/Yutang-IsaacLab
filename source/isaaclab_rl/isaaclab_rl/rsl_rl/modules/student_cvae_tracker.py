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

    def _compute_prior(self, hp_t: torch.Tensor, y_t: torch.Tensor):
        """Encode history-stacked proprio, compute prior distribution."""
        h_t = self.history_encoder(hp_t)
        mu_prior, logvar_prior = self.prior(h_t, y_t)
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

    def act(self, observations, *args, **kwargs):
        """Rollout: use prior only, sample with noise."""
        hp_t, o_t, y_t, r_t = self._split_obs(observations)

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

    def act_inference(self, observations, *args, **kwargs):
        """Training update or deployment inference.

        During training (compute_latent_loss=True): uses posterior correction and stores KL losses.
        During inference: uses prior mean only.
        """
        hp_t, o_t, y_t, r_t = self._split_obs(observations)

        mu_prior, logvar_prior = self._compute_prior(hp_t, y_t)

        if self.compute_latent_loss and r_t.shape[-1] > 0:
            # training: use posterior correction
            z_prior = self._sample_gaussian(mu_prior, logvar_prior)

            # posterior correction in low-rank space
            mu_raw, logvar_raw = self.posterior(r_t)
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

            latent_kl = self._kl_divergence(mu_zt, logvar_zt, mu_prior, logvar_prior)

            self._save_dict["cvae_corr_kl"] = corr_kl * self.corr_kl_coef
            self._save_dict["cvae_latent_kl"] = latent_kl * self.latent_kl_coef
            # Store raw (unscaled) KLs for logging
            self._save_log_dict["cvae_corr_kl"] = corr_kl.item()
            self._save_log_dict["cvae_latent_kl"] = latent_kl.item()
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

    def after_train(self):
        self.compute_latent_loss = False

    def reset(self, dones=None, hidden_states=None):
        pass

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
