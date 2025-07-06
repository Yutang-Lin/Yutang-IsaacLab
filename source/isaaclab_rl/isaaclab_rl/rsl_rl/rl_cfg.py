# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .distillation_cfg import RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherCfg
from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg
from .amp_cfg import RslRlAmpCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    layer_norm: bool = False
    """Whether to use layer normalization."""

    dropout_rate: float = 0.0
    """The dropout rate for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticOUCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCriticOU"
    """The policy class name. Default is ActorCriticOU."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    step_dt: float = 0.02
    """The time step for the OU process."""

    init_theta: float = 0.25
    """The initial theta for the OU process."""

    init_sigma: float = 0.10
    """The initial sigma for the OU process."""

    theta_range: list[float, float] = [0.1, 0.9]
    """The range of theta for the OU process."""

    sigma_range: list[float, float] = [0.1, 5.0]
    """The range of sigma for the OU process."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    layer_norm: bool = False
    """Whether to use layer normalization."""

    dropout_rate: float = 0.0
    """The dropout rate for the actor and critic networks."""


@configclass
class RslRlPpoActorDoubleCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorDoubleCritic"
    """The policy class name. Default is ActorDoubleCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    step_dt: float = 0.02
    """The time step for the OU process."""

    init_theta: float = 0.25
    """The initial theta for the OU process."""

    init_sigma: float = 0.10
    """The initial sigma for the OU process."""

    theta_range: list[float, float] = [0.1, 0.9]
    """The range of theta for the OU process."""

    sigma_range: list[float, float] = [0.1, 5.0]
    """The range of sigma for the OU process."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    layer_norm: bool = False
    """Whether to use layer normalization."""

    dropout_rate: float = 0.0
    """The dropout rate for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers."""

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCriticRecurrent."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""


@configclass
class RslRlTd3ActorCriticCfg:
    """Configuration for the TD3 actor-critic networks."""

    class_name: str = "TwinDelayed"
    """The policy class name. Default is ActorCriticTd3."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    gamma_f: float = MISSING
    """The discount factor for the forward return."""

    gamma_r: float = MISSING
    """The discount factor for the backward return."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    alpha: float = MISSING
    """The alpha parameter for the hybrid return."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    kl_coef: float = MISSING
    """The coefficient for the KL divergence."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""

    rnd_cfg: RslRlRndCfg | None = None
    """The configuration for the Random Network Distillation (RND) module. Default is None,
    in which case RND is not used.
    """

    importance_sample_value: bool = False
    """Whether to use importance sampling for the value function. Default is False."""

    centralize_log_prob: bool = False
    """Whether to centralize the log probability. Default is False."""

    init_beta: float = 0.01
    """The initial beta for the PPOKL. Default is 0.01."""

    beta_range: list[float, float] = [0.01, 1.0]
    """The range of beta for the PPOKL. Default is [0.01, 1.0]."""

    

@configclass
class RslRlTd3AlgorithmCfg:
    """Configuration for the TD3 algorithm."""

    class_name: str = "TD3"
    """The algorithm class name. Default is TD3."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    tau: float = MISSING
    """The target smoothing coefficient."""

    epsilon: float = MISSING
    """The epsilon parameter for the TD3 algorithm."""

    max_epsilon: float = MISSING
    """The maximum epsilon parameter for the TD3 algorithm."""

    num_critic_updates: int = MISSING
    """The number of critic updates per update."""

    exploration_type: Literal["ou", "normal"] = "ou"
    """The type of exploration to use. Default is ou."""

    exploration_params: dict = MISSING
    """The parameters for the exploration. Default is None."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""

    rnd_cfg: RslRlRndCfg | None = None
    """The configuration for the Random Network Distillation (RND) module. Default is None,
    in which case RND is not used.
    """

#########################
# Runner configurations #
#########################


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""
    
    class_name: str = "BaseRunner"
    """The runner class name. Default is BaseRunner."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg | RslRlDistillationStudentTeacherCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg | RslRlDistillationAlgorithmCfg = MISSING
    """The algorithm configuration."""

    clip_actions: float | None = None
    """The clipping value for actions. If ``None``, then no clipping is done.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    save_interval: int = MISSING
    """The number of iterations between saves."""

    upload_checkpoint: bool = True
    """Whether to upload the checkpoint to the cloud. Default is True."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "wandb"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    amp_cfg: RslRlAmpCfg | None = None
    """The configuration for the Adversarial Model Priors (AMP) module. Default is None,
    in which case AMP is not used.
    """