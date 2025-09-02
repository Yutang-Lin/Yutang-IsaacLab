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
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


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
            self.actor = copy.deepcopy(policy.student.actor)
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
            self.actor = copy.deepcopy(policy.student.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.student.memory_a.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
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
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

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

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        try:
            self.to("cpu")
            self.eval()
            if self.is_recurrent:
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
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs"],
                    output_names=["actions"],
                    dynamic_axes={},
                )
        except Exception as e:
            print(f"[WARNING]: Error exporting policy: {e}", flush=True)
            # import traceback
            # traceback.print_exc()
