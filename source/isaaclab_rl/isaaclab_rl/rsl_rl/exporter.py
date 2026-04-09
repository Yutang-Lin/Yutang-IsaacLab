# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch
from isaaclab_rl.rsl_rl.modules import EmpiricalNormalization


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    from isaaclab_rl.rsl_rl.modules.student_cvae_tracker import StudentCVAETracker
    target = policy.student if hasattr(policy, "student") else policy

    # BFM family: CVAE-BFM, LFM-BFM, Flow-BFM, VQ-VAE BFM
    bfm_classes = []
    try:
        from isaaclab_rl.rsl_rl.modules.student_cvae_bfm_tracker import StudentCVAEBFMTracker
        bfm_classes.append(StudentCVAEBFMTracker)
    except ImportError:
        pass
    try:
        from isaaclab_rl.rsl_rl.modules.student_lfm_bfm_tracker import StudentLFMBFMTracker
        bfm_classes.append(StudentLFMBFMTracker)
    except ImportError:
        pass
    try:
        from isaaclab_rl.rsl_rl.modules.student_flow_bfm_tracker import StudentFlowBFMTracker
        bfm_classes.append(StudentFlowBFMTracker)
    except ImportError:
        pass
    try:
        from isaaclab_rl.rsl_rl.modules.student_vqvae_bfm_tracker import StudentVQVAEBFMTracker
        bfm_classes.append(StudentVQVAEBFMTracker)
    except ImportError:
        pass

    if bfm_classes and isinstance(target, tuple(bfm_classes)):
        policy_exporter = _BFMTrackerExporter(target, normalizer)
    elif isinstance(target, StudentCVAETracker):
        policy_exporter = _CVAETrackerExporter(target, normalizer)
    else:
        policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    from isaaclab_rl.rsl_rl.modules.student_cvae_tracker import StudentCVAETracker
    target = policy.student if hasattr(policy, "student") else policy

    # BFM family
    bfm_classes = []
    for cls_name in ['StudentCVAEBFMTracker', 'StudentLFMBFMTracker', 'StudentFlowBFMTracker', 'StudentVQVAEBFMTracker']:
        try:
            cls = getattr(__import__('isaaclab_rl.rsl_rl.modules', fromlist=[cls_name]), cls_name)
            bfm_classes.append(cls)
        except (ImportError, AttributeError):
            pass

    if bfm_classes and isinstance(target, tuple(bfm_classes)):
        policy_exporter = _BFMTrackerExporter(target, normalizer)
        policy_exporter.export_onnx(path, filename)
        return
    if isinstance(target, StudentCVAETracker):
        policy_exporter = _CVAETrackerExporter(target, normalizer)
        policy_exporter.export_onnx(path, filename)
        return
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _BFMTrackerExporter(torch.nn.Module):
    """Exporter for BFM family (CVAE-BFM, LFM-BFM, Flow-BFM, VQ-VAE BFM).

    Takes pre-split inputs: (history_proprio, current_proprio, condition).
    Internally runs the inference path (prior, encoder, decoder; ODE for flow variants).
    Excludes frozen teacher.

    Exported model signature::

        actions = model(history_proprio, current_proprio, condition)
    """

    def __init__(self, policy, normalizer=None):
        super().__init__()

        # Store dims
        self.hp_dim = policy.history_proprio_ids.shape[0]
        self.o_dim = policy.current_proprio_ids.shape[0]
        self.y_dim = policy.condition_ids.shape[0]

        # Copy inference components (no teacher)
        self.history_prior = copy.deepcopy(policy.history_prior) if hasattr(policy, 'history_prior') else None
        self.history_encoder = copy.deepcopy(policy.history_encoder) if hasattr(policy, 'history_encoder') else None
        self.prior = copy.deepcopy(policy.prior) if hasattr(policy, 'prior') else None

        # Encoder (Flow-BFM, LFM-BFM)
        self.encoder = copy.deepcopy(policy.encoder) if hasattr(policy, 'encoder') else None

        # Action decoder (Flow-BFM has no action_decoder — uses self.decoder directly)
        self.action_decoder = copy.deepcopy(policy.action_decoder) if hasattr(policy, 'action_decoder') else None

        # Flow/LFM specific
        self.latent_flow = copy.deepcopy(policy.latent_flow) if hasattr(policy, 'latent_flow') else None
        self.decoder = copy.deepcopy(policy.decoder) if hasattr(policy, 'decoder') else None
        self.ode_steps = getattr(policy, 'ode_steps', 0)
        self.latent_dim = getattr(policy, 'latent_dim', 64)
        self.use_mean_flow = getattr(policy, 'use_mean_flow', False)
        self.step_dt = getattr(policy, 'step_dt', 0.02)
        self.num_frames = getattr(policy, 'num_frames', 10)
        self.frame_dim = getattr(policy, 'frame_dim', 55)
        self.num_keypoints = getattr(policy, 'num_keypoints', 6)
        self.dims_per_keypoint = getattr(policy, 'dims_per_keypoint', 9)
        self.num_actions = getattr(policy, 'num_actions', 29)

        # VQ-VAE specific
        self.codebook = copy.deepcopy(policy.codebook) if hasattr(policy, 'codebook') else None
        self.prior_predictor = copy.deepcopy(policy.prior_predictor) if hasattr(policy, 'prior_predictor') else None

        # Detect variant
        self.variant = "cvae_bfm"
        if self.latent_flow is not None:
            self.variant = "lfm_bfm"
        elif self.decoder is not None:
            self.variant = "flow_bfm"
        elif self.codebook is not None:
            self.variant = "vqvae_bfm"

        # Split normalizer
        if isinstance(normalizer, EmpiricalNormalization):
            self.hp_normalizer = normalizer.split(policy.history_proprio_ids)
            self.o_normalizer = normalizer.split(policy.current_proprio_ids)
            self.y_normalizer = normalizer.split(policy.condition_ids)
        else:
            self.hp_normalizer = torch.nn.Identity()
            self.o_normalizer = torch.nn.Identity()
            self.y_normalizer = torch.nn.Identity()

    def _parse_condition(self, y_flat):
        B = y_flat.shape[0]
        y = y_flat.view(B, self.num_frames, self.frame_dim)
        return y[:, :, :-1], y[:, :, -1]

    def forward(self, history_proprio: torch.Tensor,
                current_proprio: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        hp_t = self.hp_normalizer(history_proprio)
        o_t = self.o_normalizer(current_proprio)
        y_flat = self.y_normalizer(condition)

        frames, delta_t = self._parse_condition(y_flat)
        B = frames.shape[0]
        nf = self.num_frames
        frame_mask = torch.ones(B, nf, dtype=torch.bool, device=o_t.device)

        if self.variant == "cvae_bfm":
            h_prior = self.history_prior(hp_t) if self.history_prior is not None else self.prior(hp_t)
            if self.history_prior is not None:
                prior_out = self.prior(h_prior) if self.prior is not None else h_prior
                mu_prior, _ = prior_out.chunk(2, dim=-1)
            else:
                mu_prior = h_prior
            c_t = torch.zeros(B, self.latent_dim, device=o_t.device)
            return self.action_decoder(o_t, mu_prior, c_t, frames, delta_t, frame_mask)

        elif self.variant == "lfm_bfm":
            h_prior = self.history_prior(hp_t)
            context, ctx_mask = self.encoder(h_prior, o_t, frames, delta_t, frame_mask)
            # Latent ODE
            kv_cache, ctx_mask = self.latent_flow.build_kv_cache(context, ctx_mask)
            z = torch.randn(B, self.latent_dim, device=o_t.device)
            dt = 1.0 / self.ode_steps
            for k in range(self.ode_steps):
                t_val = 1.0 - k * dt
                t_tensor = torch.full((B,), t_val, device=o_t.device)
                v = self.latent_flow.forward_cached(z, t_tensor, kv_cache, ctx_mask)
                z = z - dt * v
            return self.action_decoder(z, context, ctx_mask)

        elif self.variant == "flow_bfm":
            h_prior = self.history_prior(hp_t)
            context, ctx_mask = self.encoder(h_prior, o_t, frames, delta_t, frame_mask)
            # Action ODE
            kv_cache, ctx_mask = self.decoder.build_kv_cache(context, ctx_mask)
            a = torch.randn(B, self.num_actions, device=o_t.device)
            dt = 1.0 / self.ode_steps
            for k in range(self.ode_steps):
                t_val = 1.0 - k * dt
                t_tensor = torch.full((B,), t_val, device=o_t.device)
                v = self.decoder.forward_cached(a, t_tensor, kv_cache, ctx_mask)
                a = a - dt * v
            return a

        elif self.variant == "vqvae_bfm":
            h_prior = self.history_prior(hp_t)
            prev_e_q = torch.zeros(B, self.latent_dim, device=o_t.device)
            logits, o_t_enc, h_prior_enc = self.prior_predictor(o_t, h_prior, prev_e_q, frames, delta_t, frame_mask)
            indices = logits.argmax(dim=-1)
            e_q = self.codebook.embedding(indices)
            return self.action_decoder(o_t_enc, h_prior_enc, e_q, frames, delta_t, frame_mask, pre_encoded=True)

        return torch.zeros(B, 29, device=o_t.device)

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        self.to("cpu")
        self.eval()
        dummy_hp = torch.zeros(1, self.hp_dim)
        dummy_o = torch.zeros(1, self.o_dim)
        dummy_y = torch.zeros(1, self.y_dim)
        try:
            traced = torch.jit.trace(self, (dummy_hp, dummy_o, dummy_y))
            traced.save(filepath)
            print(f"[INFO] Exported BFM ({self.variant}) via JIT trace to {filepath}")
        except Exception as e:
            print(f"[WARNING] JIT trace failed for {self.variant}: {e}")
            # Save as plain state dict fallback
            torch.save({
                "model_state_dict": self.state_dict(),
                "variant": self.variant,
                "hp_dim": self.hp_dim, "o_dim": self.o_dim, "y_dim": self.y_dim,
            }, filepath)
            print(f"[INFO] Saved state dict fallback to {filepath}")

    def export_onnx(self, path, filename):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        self.to("cpu")
        self.eval()
        dummy_hp = torch.zeros(1, self.hp_dim)
        dummy_o = torch.zeros(1, self.o_dim)
        dummy_y = torch.zeros(1, self.y_dim)
        try:
            torch.onnx.export(
                self, (dummy_hp, dummy_o, dummy_y), filepath,
                export_params=True, opset_version=14,
                input_names=["history_proprio", "current_proprio", "condition"],
                output_names=["actions"],
                dynamic_axes={},
            )
            print(f"[INFO] Exported BFM ({self.variant}) via ONNX to {filepath}")
        except Exception as e:
            print(f"[WARNING] ONNX export failed for {self.variant}: {e}")


class _CVAETrackerExporter(torch.nn.Module):
    """Exporter for StudentCVAETracker into JIT/ONNX file.

    Takes three separate inputs (history_proprio, current_proprio, condition),
    each with its own split normalizer. Excludes the frozen teacher to avoid
    JIT errors from ActorCritic.distribution being None.

    Exported model signature::

        actions = model(history_proprio, current_proprio, condition)
    """

    def __init__(self, policy, normalizer: EmpiricalNormalization | None = None):
        super().__init__()
        # Copy only inference components (no teacher)
        self.history_encoder = copy.deepcopy(policy.history_encoder)
        self.prior = copy.deepcopy(policy.prior)
        self.action_decoder = copy.deepcopy(policy.action_decoder)

        # Store dims for dummy input generation
        self.hp_dim = policy.history_proprio_ids.shape[0]
        self.o_dim = policy.current_proprio_ids.shape[0]
        self.y_dim = policy.condition_ids.shape[0]

        # Split normalizer per group
        if isinstance(normalizer, EmpiricalNormalization):
            self.hp_normalizer = normalizer.split(policy.history_proprio_ids)
            self.o_normalizer = normalizer.split(policy.current_proprio_ids)
            self.y_normalizer = normalizer.split(policy.condition_ids)
        else:
            self.hp_normalizer = torch.nn.Identity()
            self.o_normalizer = torch.nn.Identity()
            self.y_normalizer = torch.nn.Identity()

    def forward(self, history_proprio: torch.Tensor,
                current_proprio: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        hp_t = self.hp_normalizer(history_proprio)
        o_t = self.o_normalizer(current_proprio)
        y_t = self.y_normalizer(condition)
        h_t = torch.nn.functional.normalize(self.history_encoder(hp_t), dim=-1)
        mu_prior, _ = self.prior(h_t, y_t)
        return self.action_decoder(o_t, y_t, mu_prior)

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        self.to("cpu")
        self.eval()
        dummy_hp = torch.zeros(1, self.hp_dim)
        dummy_o = torch.zeros(1, self.o_dim)
        dummy_y = torch.zeros(1, self.y_dim)
        try:
            traced = torch.jit.trace(self, (dummy_hp, dummy_o, dummy_y))
            traced.save(filepath)
        except Exception as e:
            print(f"[WARNING]: Error exporting CVAE tracker policy: {e}", flush=True)

    def export_onnx(self, path, filename):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        self.to("cpu")
        self.eval()
        dummy_hp = torch.zeros(1, self.hp_dim)
        dummy_o = torch.zeros(1, self.o_dim)
        dummy_y = torch.zeros(1, self.y_dim)
        torch.onnx.export(
            self, (dummy_hp, dummy_o, dummy_y), filepath,
            export_params=True, opset_version=14,
            input_names=["history_proprio", "current_proprio", "condition"],
            output_names=["actions"],
            dynamic_axes={},
        )


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer: EmpiricalNormalization | None = None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.is_recurrent = policy.student.is_recurrent
            if hasattr(policy.student, "actor"):
                self.actor = copy.deepcopy(policy.student.actor)
            else:
                raise ValueError("Policy student does not have an actor module. Use _CVAETrackerExporter instead.")
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.student.memory_a.rnn)
            policy = policy.student
        else:
            raise ValueError("Policy does not have an actor/student module.")
        
        self.split_ids = dict()
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            if self.rnn_type == "lstm":
                self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self.forward_lstm
                self.reset = self.reset_memory
            elif self.rnn_type == "gru":
                self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self.forward_gru
                self.reset = self.reset_memory
            elif self.rnn_type in ["lnnstyletransformer", "lnnstyletransformerml", 'lnnstyletransformerll', 'lnnstyletransformerlatent']:
                self.register_buffer("hidden_state", self.rnn.initial_history_tokens.clone().unsqueeze(1))
                self.split_ids['proprio'] = policy.actor_proprio_ids
                self.split_ids['condition'] = policy.actor_condition_ids
                self.forward = self.forward_transformer
                self.reset = self.reset_memory_transformer
                self.rnn.forward = self.rnn.forward_inference
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
            self.hidden_state: torch.Tensor

        # get policy
        if hasattr(self.actor, "forward_inference"):
            self.split_ids['proprio'] = policy.actor_proprio_ids
            self.split_ids['condition'] = policy.actor_condition_ids
            self.actor.forward = self.actor.forward_inference
            self.forward = self.forward_latent

        if self.actor.__class__.__name__ == "TransformerPolicyInteractionField":
            self.split_ids['proprio'] = policy.actor_proprio_ids
            self.split_ids['interaction_field'] = policy.actor_interaction_field_ids
            self.split_ids['movement_goal'] = policy.actor_movement_goal_ids
            self.split_ids['task_condition'] = policy.actor_task_condition_ids
            self.forward = self.forward_interaction_field

        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        self.split_normalizer = torch.nn.ModuleDict()
        if len(self.split_ids) > 0:
            if isinstance(self.normalizer, EmpiricalNormalization):
                for key in self.split_ids:
                    self.split_normalizer[key] = self.normalizer.split(self.split_ids[key])
            else:
                for key in self.split_ids:
                    self.split_normalizer[key] = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward_gru(self, x):
        x = self.normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        x = x.squeeze(0)
        return self.actor(x)
    
    def forward_transformer(self, proprio: torch.Tensor, 
                            condition: torch.Tensor | None = None, 
                            latent: torch.Tensor | None = None,
                            apply_vae_noise: bool = False):
        proprio = self.split_normalizer['proprio'](proprio)
        if condition is not None:
            condition = self.split_normalizer['condition'](condition).unsqueeze(0)
        if latent is not None:
            latent = latent.unsqueeze(0)
        x, h = self.rnn.forward_inference(proprio.unsqueeze(0), 
                                          condition, 
                                          latent, 
                                          self.hidden_state,
                                          apply_vae_noise)
        self.hidden_state[:] = h
        x = x.squeeze(0)
        return self.actor(x)
    
    def forward_interaction_field(self, proprio: torch.Tensor, 
                                  interaction_field: torch.Tensor, 
                                  movement_goal: torch.Tensor, 
                                  task_condition: torch.Tensor):
        proprio = self.split_normalizer['proprio'](proprio)
        interaction_field = self.split_normalizer['interaction_field'](interaction_field)
        movement_goal = self.split_normalizer['movement_goal'](movement_goal)
        task_condition = self.split_normalizer['task_condition'](task_condition)
        x = self.actor(proprio=proprio, interaction_field=interaction_field, movement_goal=movement_goal, task_condition=task_condition)
        return x

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def forward_latent(self, proprio: torch.Tensor, 
                       condition: torch.Tensor | None = None, 
                       latent: torch.Tensor | None = None,
                       apply_vae_noise: bool = False):
        proprio = self.split_normalizer['proprio'](proprio)
        if condition is not None:
            condition = self.split_normalizer['condition'](condition)
        return self.actor.forward_inference(proprio, condition, latent, apply_vae_noise)

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        if hasattr(self, "cell_state"):
            self.cell_state[:] = 0.0

    def reset_memory_transformer(self):
        self.hidden_state[:] = self.rnn.initial_history_tokens.clone().unsqueeze(1)

    def export(self, path, filename):
        try:
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, filename)
            self.to("cpu")
            traced_script_module = torch.jit.script(self)
            traced_script_module.save(path)
        except Exception as e:
            print(f"[WARNING]: Error exporting policy: {e}", flush=True)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.is_recurrent = policy.student.is_recurrent
            if hasattr(policy.student, "actor"):
                self.actor = copy.deepcopy(policy.student.actor)
            else:
                raise ValueError("Policy student does not have an actor module.")
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.student.memory_a.rnn)
            policy = policy.student
        else:
            raise ValueError("Policy does not have an actor/student module.")
        
        self.split_ids = dict()
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            elif self.rnn_type in ["rnnstyletransformer", "lnnstyletransformer", "lnnstyletransformerml", "lnnstyletransformerll", "lnnstyletransformerlatent"]:
                self.forward = self.forward_transformer
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        
        # get policy
        if hasattr(self.actor, "forward_inference"):
            self.split_ids['proprio'] = policy.actor_proprio_ids
            self.split_ids['condition'] = policy.actor_condition_ids
            self.actor.forward = self.actor.forward_inference
            self.forward = self.forward_latent

        if self.actor.__class__.__name__ == "TransformerPolicyInteractionField":
            self.split_ids['proprio'] = policy.actor_proprio_ids
            self.split_ids['interaction_field'] = policy.actor_interaction_field_ids
            self.split_ids['movement_goal'] = policy.actor_movement_goal_ids
            self.split_ids['task_condition'] = policy.actor_task_condition_ids
            self.forward = self.forward_interaction_field
        
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        self.split_normalizer = torch.nn.ModuleDict()
        if len(self.split_ids) > 0:
            if isinstance(self.normalizer, EmpiricalNormalization):
                for key in self.split_ids:
                    self.split_normalizer[key] = self.normalizer.split(self.split_ids[key])
            else:
                for key in self.split_ids:
                    self.split_normalizer[key] = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_gru(self, x_in, h_in):
        x_in = self.normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h
    
    def forward_transformer(self, x_in, h_in):
        x_in = self.normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h

    def forward_interaction_field(self, proprio: torch.Tensor, 
                                  interaction_field: torch.Tensor, 
                                  movement_goal: torch.Tensor, 
                                  task_condition: torch.Tensor):
        proprio = self.split_normalizer['proprio'](proprio)
        interaction_field = self.split_normalizer['interaction_field'](interaction_field)
        movement_goal = self.split_normalizer['movement_goal'](movement_goal)
        task_condition = self.split_normalizer['task_condition'](task_condition)
        x = self.actor(proprio=proprio, interaction_field=interaction_field, movement_goal=movement_goal, task_condition=task_condition)
        return x

    def forward_latent(self, proprio: torch.Tensor, 
                       condition: torch.Tensor | None = None, 
                       latent: torch.Tensor | None = None,
                       apply_vae_noise: bool = False):
        proprio = self.split_normalizer['proprio'](proprio)
        if condition is not None:
            condition = self.split_normalizer['condition'](condition)
        return self.actor.forward_inference(proprio, condition, latent, apply_vae_noise)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        try:
            self.to("cpu")
            self.eval()
            # Check if this is an interaction field policy
            if 'interaction_field' in self.split_ids:
                # Get input sizes from split normalizers
                proprio_size = len(self.split_ids['proprio'])
                interaction_field_size = len(self.split_ids['interaction_field'])
                movement_goal_size = len(self.split_ids['movement_goal'])
                task_condition_size = len(self.split_ids['task_condition'])
                
                proprio = torch.zeros(1, proprio_size)
                interaction_field = torch.zeros(1, interaction_field_size)
                movement_goal = torch.zeros(1, movement_goal_size)
                task_condition = torch.zeros(1, task_condition_size)
                
                torch.onnx.export(
                    self,
                    (proprio, interaction_field, movement_goal, task_condition),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=14,
                    verbose=self.verbose,
                    input_names=["proprio", "interaction_field", "movement_goal", "task_condition"],
                    output_names=["actions"],
                    dynamic_axes={},
                )
            elif self.is_recurrent:
                obs = torch.zeros(1, self.rnn.input_size)
                
                if self.rnn_type == "lstm":
                    h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                    c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                    torch.onnx.export(
                        self,
                        (obs, h_in, c_in),
                        os.path.join(path, filename),
                        export_params=True,
                        opset_version=11,
                        verbose=self.verbose,
                        input_names=["obs", "h_in", "c_in"],
                        output_names=["actions", "h_out", "c_out"],
                        dynamic_axes={},
                    )
                elif self.rnn_type == "gru":
                    h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                    torch.onnx.export(
                        self,
                        (obs, h_in),
                        os.path.join(path, filename),
                        export_params=True,
                        opset_version=11,
                        verbose=self.verbose,
                        input_names=["obs", "h_in"],
                        output_names=["actions", "h_out"],
                        dynamic_axes={},
                    )
                elif self.rnn_type in ["lnnstyletransformer", "lnnstyletransformerml", 'lnnstyletransformerll', 'lnnstyletransformerlatent']:
                    h_in = torch.zeros(self.rnn.num_history_tokens, 1, self.rnn.d_model)
                    torch.onnx.export(
                        self,
                        (obs, h_in),
                        os.path.join(path, filename),
                        export_params=True,
                        opset_version=11,
                        verbose=self.verbose,
                        input_names=["obs", "h_in"],
                        output_names=["actions", "h_out"],
                        dynamic_axes={},
                    )
                else:
                    raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
            else:
                try:
                    obs = torch.zeros(1, self.actor[0].in_features)
                except:
                    obs = torch.zeros(1, self.actor.in_features)
                torch.onnx.export(
                    self,
                    obs,
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=14,
                    verbose=self.verbose,
                    input_names=["obs"],
                    output_names=["actions"],
                    dynamic_axes={},
                )
        except Exception as e:
            print(f"[WARNING]: Error exporting policy: {e}", flush=True)
            import traceback
            traceback.print_exc()
