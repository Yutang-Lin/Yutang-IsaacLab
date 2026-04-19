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
from .smp_cfg import RslRlSmpCfg

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

    load_noise_std: bool = True
    """Whether to load the noise std from the checkpoint. Default is True."""

    learnable_noise_std: bool = True
    """Whether to make the noise std learnable. Default is True."""

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

    residual: bool = False
    """Whether to use residual connections."""

    init_zero: bool = False
    """Whether to initialize the actor and critic networks to zero."""

@configclass
class RslRlPpoActorCriticHyperMLPCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with hyper-MLP."""

    class_name: str = "ActorCriticHyperMLP"
    """The policy class name. Default is ActorCriticHyperMLP."""

    hyper_layer_idx: int = 0
    """The index of the hyper-layer."""

    proprio_horizon: int = 5
    """The horizon of the proprioceptive observations."""


@configclass
class RslRlPpoActorCriticDPCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with diffusion process."""

    class_name: str = "ActorCriticDP"
    """The policy class name. Default is ActorCriticDP."""
    
    condition_hidden_dim: int = 256
    """The hidden dimension of the condition."""

    timestep_hidden_dim: int = 256
    """The hidden dimension of the timestep."""

    max_timesteps: int = 1000
    """The maximum number of timesteps."""

    action_timestep: int = 50
    """The action timestep."""

    action_step_num: int = 10
    """The number of action steps."""

    diffusion_loss_step_num: int = 2
    """The number of diffusion loss iterations."""

    reference_loss_step_num: int = 1
    """The number of reference loss iterations."""

    reference_gradient: bool = False
    """Whether to use the reference gradient."""

    alphas: any = None
    """The alphas for the diffusion process."""

    sigmas: any = None
    """The sigmas for the diffusion process."""

    ddim_lambda: float = 1.0
    """The lambda for the classifier-free diffusion process."""

    ddim_eta: float = 0.0
    """The eta for the diffusion process."""

    lernable_sigmas: bool = False
    """Whether to make the sigmas learnable."""

    learn_residual: bool = False
    """Whether to make the residual learnable."""


@configclass
class RslRlPpoActorCriticDPTransformerCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with diffusion process."""

    class_name: str = "ActorCriticDPTransformer"
    """The policy class name. Default is ActorCriticDPTransformer."""

    tf_d_model: int = MISSING
    """The dimension of the transformer model."""

    tf_num_heads: int = MISSING
    """The number of transformer heads."""

    tf_hidden_dim: int = MISSING
    """The dimension of the transformer hidden layers."""

    tf_num_layers: int = MISSING
    """The number of transformer layers."""

    tf_condition_tokens: int = MISSING
    """The number of condition tokens."""

    timestep_hidden_dim: int = 256
    """The hidden dimension of the timestep."""

    max_timesteps: int = 1000
    """The maximum number of timesteps."""

    action_timestep: int = 50
    """The action timestep."""

    action_step_num: int = 10
    """The number of action steps."""

    diffusion_loss_step_num: int = 2
    """The number of diffusion loss iterations."""

    reference_loss_step_num: int = 1
    """The number of reference loss iterations."""

    reference_gradient: bool = False
    """Whether to use the reference gradient."""

    alphas: any = None
    """The alphas for the diffusion process."""

    sigmas: any = None
    """The sigmas for the diffusion process."""

    ddim_lambda: float = 1.0
    """The lambda for the classifier-free diffusion process."""

    ddim_eta: float = 0.0
    """The eta for the diffusion process."""

    lernable_sigmas: bool = False
    """Whether to make the sigmas learnable."""

    learn_residual: bool = False
    """Whether to make the residual learnable."""

    # Deprecated
    actor_hidden_dims: list[int] = []
    """The hidden dimensions of the actor network."""


@configclass
class RslRlPpoActorCriticMoECfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic mixture of experts networks."""

    class_name: str = "ActorCriticMoE"
    """The policy class name. Default is ActorCriticMoE."""

    num_experts: int = MISSING
    """The number of experts."""

    top_k: int = MISSING
    """The top k for the MoE."""

    balance_tolerance: float = MISSING
    """The balance tolerance for the MoE."""

    balance_loss_weight: float = MISSING
    """The balance loss weight for the MoE."""

    moe_critic: bool = False
    """Whether to use the MoE critic."""


@configclass
class RslRlPpoActorCriticMoPCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic mixture of policies networks."""

    class_name: str = "ActorCriticMoP"
    """The policy class name. Default is ActorCriticMoP."""

    num_policies: int = MISSING
    """The number of policies."""

    router_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the router network."""

    balance_tolerance: float = MISSING
    """The balance tolerance for the MoE."""

    balance_loss_weight: float = MISSING
    """The balance loss weight for the MoE."""

    grad_penalty_weight: float = MISSING
    """The gradient penalty weight for the MoP."""

    mop_critic: bool = False
    """Whether to use the MoP critic."""

@configclass
class RslRlPpoActorCriticPNNCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic progressive neural network."""

    class_name: str = "ActorCriticPNN"
    """The policy class name. Default is ActorCriticPNN."""

    num_policies: int = MISSING
    """The number of policies."""

    pnn_critic: bool = False
    """Whether to use the PNN critic."""

    weight_sharing: bool = True
    """Whether to share the weights of the PNN."""

    start_by_id: int = 0
    """The policy id to start the PNN."""

    router_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the router network."""

    grad_penalty_weight: float = MISSING
    """The gradient penalty weight for the PNN."""


@configclass
class RslRlPpoActorCriticOUCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCriticOU"
    """The policy class name. Default is ActorCriticOU."""

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


@configclass
class RslRlPpoActorDoubleCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorDoubleCritic"
    """The policy class name. Default is ActorDoubleCritic."""

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
class RslRlPpoActorCriticTransformerCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with transformer layers."""

    class_name: str = "ActorCriticTransformer"
    """The policy class name. Default is ActorCriticTransformer."""

    tf_d_model: int = MISSING
    """The dimension of the transformer model."""

    tf_critic_d_model: int | None = None
    """The dimension of the critic transformer model."""

    tf_num_input_tokens: int = MISSING
    """The number of input tokens."""

    tf_critic_num_input_tokens: int | None = None
    """The number of input tokens for the critic transformer model."""

    tf_num_heads: int = MISSING
    """The number of transformer heads."""

    tf_critic_num_heads: int | None = None
    """The number of transformer heads for the critic."""

    tf_num_layers: int = MISSING
    """The number of transformer layers."""

    tf_critic_num_layers: int | None = None
    """The number of transformer layers for the critic."""

    tf_hidden_dim: int = MISSING
    """The dimension of the transformer hidden layers."""

    tf_critic_hidden_dim: int | None = None
    """The dimension of the transformer hidden layers for the critic."""

    tf_dropout: float = 0.0
    """The dropout rate for the transformer."""

    tf_critic_dropout: float | None = None
    """The dropout rate for the critic transformer."""

    tf_activation: str = "gelu"
    """The activation function for the transformer."""

    distributed_critic: bool = False
    """Whether to use distributed critic (no gradient sync across ranks).

    When enabled, each rank's critic does not sync gradients with other ranks.
    Parameters are only synchronized at the beginning of training via broadcast.
    This leads to a naturally multi-rank mixture of critics architecture.
    Checkpoints will only include rank 0's critic parameters.
    """

    distributed_actor: bool = False
    """Fully distributed training — no gradient sync for actor or critic.

    Each rank trains completely independently. Each rank saves its own
    checkpoints and logs to a rank-specific subdirectory. Implies distributed_critic.
    """

    distributed_s3_prefix: str = ""
    """S3 prefix for uploading per-rank checkpoints.

    Example: 's3://far-research-internal/yutangl/checkpoints/{run_name}'.
    '{run_name}' is replaced with the experiment run name at runtime.
    Each rank uploads to {prefix}/rank_{rank}/.
    """

@configclass
class RslRlPpoActorCriticTransformerFlowCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with transformer layers."""

    class_name: str = "ActorCriticTransformerFlow"
    """The policy class name. Default is ActorCriticTransformerFlow."""

    tf_d_model: int = MISSING
    """The dimension of the transformer model."""

    tf_num_proprio_tokens: int = MISSING
    """The number of input tokens."""

    tf_num_action_tokens: int = MISSING
    """The number of action tokens."""

    tf_control_obs_horizon: int = MISSING
    """The number of control observation tokens."""

    tf_num_heads: int = MISSING
    """The number of transformer heads."""

    tf_num_layers: int = MISSING
    """The number of transformer layers."""

    tf_hidden_dim: int = MISSING
    """The dimension of the transformer hidden layers."""

    tf_dropout: float = 0.0
    """The dropout rate for the transformer."""

    tf_activation: str = "gelu"
    """The activation function for the transformer."""

    denoise_loss_coef: float = 1.0
    """The coefficient for the denoise loss."""

    sim_learning_epochs: int = 1
    """The number of epochs to simulate the learning."""

    sim_action_loss_coef: float = 1.0
    """The coefficient for the action loss."""

    sim_state_loss_coef: float = 1.0
    """The coefficient for the state loss."""

@configclass
class RslRlPpoActorCriticTransformerDDIMCfg(RslRlPpoActorCriticTransformerFlowCfg):
    """Configuration for the PPO actor-critic networks with transformer layers."""

    class_name: str = "ActorCriticTransformerDDIM"
    """The policy class name. Default is ActorCriticTransformerDDIM."""

@configclass
class RslRlPpoActorCriticTransformerMeanFlowCfg(RslRlPpoActorCriticTransformerFlowCfg):
    """Configuration for the PPO actor-critic networks with transformer layers."""

    class_name: str = "ActorCriticTransformerMeanFlow"
    """The policy class name. Default is ActorCriticTransformerMeanFlow."""

    tf_proprio_horizon: int = MISSING
    """The number of proprioceptive tokens."""

    flow_r_neq_t_prob: float = 0.25,
    """The probability of r not equal to t. Default is 0.25."""

    flow_loss_coef_p: float = 1.0,
    """The power for the flow loss. Default is 1.0."""
    
    flow_loss_coef_c: float = 1e-3,
    """The constant for the flow loss. Default is 1e-3."""

@configclass
class RslRlPpoActorCriticTransformerCoMeanFlowCfg(RslRlPpoActorCriticTransformerMeanFlowCfg):
    """Configuration for the PPO actor-critic networks with transformer layers."""

    class_name: str = "ActorCriticTransformerCoMeanFlow"
    """The policy class name. Default is ActorCriticTransformerCoMeanFlow."""

    tf_action_horizon: int = MISSING
    """The number of action tokens."""

@configclass
class RslRlPpoActorCriticTransformerResidualCfg(RslRlPpoActorCriticTransformerCfg):
    """Configuration for the PPO actor-critic networks with transformer layers."""

    class_name: str = "ActorCriticTransformerResidual"
    """The policy class name. Default is ActorCriticTransformerResidual."""

    base_policy_path: str | None = None
    """The path to the base policy."""

@configclass
class RslRlPpoActorCriticTransformerLatentCfg(RslRlPpoActorCriticTransformerCfg):
    """Configuration for the PPO actor-critic networks with transformer layers."""

    class_name: str = "ActorCriticTransformerLatent"
    """The policy class name. Default is ActorCriticTransformerLatent."""

    tf_num_latent_tokens: int = MISSING
    """The number of latent tokens."""

    latent_kl_coef: float = 1e-5
    """The KL coefficient for the latent tokens."""

    latent_recons_coef: float = 1.0
    """The reconstruction coefficient for the latent tokens."""

    latent_stable_coef: float = 1e-3
    """The stable coefficient for the latent tokens."""

@configclass
class RslRlPpoActorCriticTransformerInteractionFieldCfg(RslRlPpoActorCriticTransformerCfg):
    """Configuration for the PPO actor-critic networks with transformer layers."""

    class_name: str = "ActorCriticTransformerInteractionField"
    """The policy class name. Default is ActorCriticTransformerLatent."""

    tf_num_fusion_heads: int = MISSING
    """The number of fusion heads for the interaction field."""
    

@configclass
class RslRlPpoActorCriticTFRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with transformer layers."""

    class_name: str = "ActorCriticTFRecurrent"
    """The policy class name. Default is ActorCriticTFRecurrent."""

    tf_d_model: int = MISSING
    """The dimension of the transformer model."""

    tf_num_input_tokens: int = MISSING
    """The number of input tokens."""

    tf_num_history_tokens: int = MISSING
    """The number of history tokens."""

    tf_lnn_dt: float = MISSING
    """The time step for the LNN."""

    tf_lnn_tau: float = MISSING
    """The tau for the LNN."""

    tf_num_heads: int = MISSING
    """The number of transformer heads."""

    tf_num_layers: int = MISSING
    """The number of transformer layers."""

    tf_hidden_dim: int = MISSING
    """The dimension of the transformer hidden layers."""

    tf_dropout: float = 0.0
    """The dropout rate for the transformer."""

    tf_activation: str = "gelu"
    """The activation function for the transformer."""

@configclass
class RslRlPpoActorCriticTFRecurrentMLCfg(RslRlPpoActorCriticTFRecurrentCfg):
    """Configuration for the PPO actor-critic networks with transformer layers."""

    tf_num_task_tokens: int = MISSING
    """The number of task tokens."""

    class_name: str = "ActorCriticTFRecurrentML"
    """The policy class name. Default is ActorCriticTFRecurrentML."""

@configclass
class RslRlPpoActorCriticTFRecurrentLLCfg(RslRlPpoActorCriticTFRecurrentCfg):
    """Configuration for the PPO actor-critic networks with transformer layers."""

    tf_num_task_tokens: int = MISSING
    """The number of task tokens."""

    tf_num_latent_tokens: int = MISSING
    """The number of latent tokens."""

    latent_kl_coef: float = 1e-5
    """The KL coefficient for the latent tokens."""

    latent_recons_coef: float = 1.0
    """The reconstruction coefficient for the latent tokens."""

    class_name: str = "ActorCriticTFRecurrentLL"
    """The policy class name. Default is ActorCriticTFRecurrentLL."""

@configclass
class RslRlPpoActorCriticTFRecurrentLatentCfg(RslRlPpoActorCriticTFRecurrentCfg):
    """Configuration for the PPO actor-critic networks with transformer layers."""

    tf_num_latent_tokens: int = MISSING
    """The number of latent tokens."""

    latent_kl_coef: float = 1e-5
    """The KL coefficient for the latent tokens."""

    latent_recons_coef: float = 1.0
    """The reconstruction coefficient for the latent tokens."""

    latent_stable_coef: float = 1e-3
    """The stable coefficient for the latent tokens."""

    class_name: str = "ActorCriticTFRecurrentLatent"
    """The policy class name. Default is ActorCriticTFRecurrentLatent."""

@configclass
class RslRlSparseSuccessorPolicyCfg:
    """Configuration for the sparse-constraint successor tracking policy."""

    class_name: str = "SparseSuccessorPolicy"
    """The policy class name."""

    num_keypoints: int = MISSING
    """Number of trackable keypoints on the robot body."""

    target_dim: int = 3
    """Dimensionality of each keypoint target (3 for position)."""

    d_model: int = 128
    """Latent dimension for query/constraint encodings."""

    max_constraints: int = 16
    """Maximum number of constraints in a padded set."""

    actor_hidden_dims: list[int] = MISSING
    """Hidden dimensions of the actor MLP."""

    critic_hidden_dims: list[int] = MISSING
    """Hidden dimensions of the successor and style critic MLPs."""

    disc_hidden_dims: list[int] = MISSING
    """Hidden dimensions of the style discriminator."""

    activation: str = "elu"
    """Activation function name."""

    actor_fixed_std: float = 0.2
    """Fixed exploration stddev of the TruncatedNormal action distribution
    (BFM-style). No learnable noise; use ``actor_stddev_clip`` to bound the
    noise magnitude."""

    actor_stddev_clip: float = 0.3
    """Hard clip on the noise term inside TruncatedNormal.sample()."""

    action_low: float = -1.0
    """Lower bound of the tanh-squashed action range."""

    action_high: float = 1.0
    """Upper bound of the tanh-squashed action range."""

    snippet_length: int = 8
    """Number of frames in a style snippet."""

    project_constraint_latent: Literal["none", "unit_sphere", "clamp_radius"] = "unit_sphere"
    """Projection applied to ``z_C`` right after ``ConstraintSetEncoder.post_mlp``.

    - ``none``: identity. ``z_C`` magnitude is unconstrained — can drift during
      training, which tends to destabilize the actor and discriminator.
    - ``unit_sphere``: scale to fixed L2 norm ``sqrt(d_model)``. Mirrors
      BFM-Zero's ``project_z`` and gives ``z_C`` bounded scale.
    - ``clamp_radius``: shrink to radius ``constraint_latent_clamp_radius`` when
      above, identity otherwise. Cheaper constraint than unit-sphere."""

    constraint_latent_clamp_radius: float = 1.0
    """Radius used when ``project_constraint_latent == 'clamp_radius'``."""

    style_feature_dim: int | None = None
    """Per-frame style feature dim. Should match the ExpertMotionBuffer. If
    ``None``, the discriminator falls back to ``num_actor_obs`` per frame
    (legacy path; only useful when running without an expert dataset)."""

    snippet_dim: int | None = None
    """Override the full snippet_dim (takes priority over style_feature_dim *
    snippet_length). Leave ``None`` unless you know why."""

    layer_norm: bool = False
    """Whether to use layer normalization in networks (plain-MLP path only)."""

    use_residual_arch: bool = True
    """Use BFM-style residual architecture (LayerNorm → Linear → Mish + skip
    connections) for the actor, successor critics, style critics, and aux
    critics. When False, the sub-networks fall back to the plain-MLP path."""

    residual_hidden_dim: int = 1024
    """Residual body hidden dim (BFM default = 1024)."""

    residual_hidden_layers: int = 1
    """Number of residual blocks in each sub-network body (BFM default = 1)."""

    residual_embedding_layers: int = 2
    """Number of residual-embedding blocks per input branch (BFM default = 2).
    The final block halves the dim to ``hidden_dim/2`` so that
    ``concat(embed_a, embed_b)`` is the full hidden dim."""


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

    num_critic_extra_epochs: int = 0
    """The number of extra critic epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    max_learning_rate: float = 1e-2
    """The maximum learning rate for the policy."""

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

    desired_clipping: float = -1.0
    """The desired clipping."""

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

    use_lipschitz_constraint: bool = False
    """Whether to use the lipschitz constraint. Default is False."""

    lipschitz_constraint_coef: float = 2e-3
    """The coefficient for the lipschitz constraint. Default is 1e-3."""
    
    adjust_critic_lr: bool = True
    """Whether to adjust the critic learning rate. Default is True."""

    use_distillation: bool = False
    """Whether to use distillation. Default is False."""

    distillation_only: bool = False
    """Whether to only use distillation. Default is False."""

    distillation_coef: float = 1.0
    """The coefficient for the distillation loss. Default is 1.0."""

    critic_only_steps: int = 0
    """The number of steps to only update the critic. Default is 0."""


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


@configclass
class RslRlSparseSuccessorAlgorithmCfg:
    """Configuration for the sparse-constraint successor tracking algorithm."""

    class_name: str = "SparseSuccessor"
    """The algorithm class name."""

    lr_actor: float = 3e-4
    """Learning rate for the actor."""

    lr_critic: float = 3e-4
    """Learning rate for successor and style critics."""

    lr_query: float = 3e-4
    """Learning rate for query and constraint encoders."""

    lr_disc: float = 1e-4
    """Learning rate for the style discriminator."""

    gamma: float = 0.99
    """Discount factor."""

    target_tau: float = 0.005
    """Soft target update rate."""

    lambda_style: float = 0.1
    """Weight of the style Q-value in the actor objective.

    Default deliberately small because ``q_track`` is bounded roughly in
    ``[0, 1/(1-gamma)]`` while ``q_style`` is logit-based and unbounded. Log
    ``Scale/q_track_*`` vs ``Scale/q_style_*`` and retune after a short warmup."""

    lambda_aux: float = 1.0
    """Weight of the auxiliary-env-reward Q-value in the actor objective.

    The aux reward is running-normalized before hitting the critic, so its Q
    should be of order 1. Set ``0.0`` to disable the aux branch entirely
    (env rewards will still be logged but not consumed by training)."""

    critic_pessimism_penalty: float = 0.5
    """Ensemble pessimism penalty used when computing the TD bootstrap for all
    three critic families. ``Q = 0.5*(Q1+Q2) - penalty * |Q1 - Q2|``. 0.5
    recovers standard min-Q double-Q; larger is more conservative."""

    actor_pessimism_penalty: float = 0.5
    """Ensemble pessimism penalty when the actor consumes Q-values for its
    own policy-improvement loss. Often set equal to or slightly smaller than
    ``critic_pessimism_penalty``."""

    sigma_time: float = 2.0
    """Gaussian kernel width for time proximity in successor critic."""

    beta: float | list[float] = 0.1
    """Gaussian kernel width for keypoint satisfaction.

    If a ``float``, the same bandwidth is used for every keypoint.
    If a ``list[float]`` of length ``num_keypoints`` (matching the agent
    config's ``SPARSE_SUCCESSOR_KEYPOINTS`` order), each keypoint gets its
    own bandwidth. Per-keypoint is recommended because end-effectors (wrists,
    ankles) cover much wider positional ranges than the pelvis/torso."""

    tau_max: int = 20
    """Maximum future lag for queries."""

    n_constraints_min: int = 1
    """Minimum number of constraints sampled per set."""

    n_constraints_max: int = 8
    """Maximum number of constraints sampled per set."""

    weight_range: tuple[float, float] = (0.5, 1.5)
    """Range for random constraint importance weights."""

    target_noise_std: float = 0.02
    """Noise added to constraint target values during sampling."""

    constraint_dropout_prob: float = 0.1
    """Probability of dropping individual constraints."""

    constraint_horizon: int = 16
    """Chunk length (in env steps) over which the rollout-time constraint set
    is held fixed. Only ``tau`` is decremented each step within a chunk; at
    chunk boundary a fresh ``C`` is sampled. Setting this to 1 reproduces
    the legacy per-step resampling behaviour."""

    expert_chunk_fraction: float = 0.15
    """Fraction of newly-sampled rollout chunks (at chunk boundaries) that
    draw their constraint set from the expert motion buffer's keypoints
    instead of the live privileged state. Small by design — the method is
    meant to be self-supervised, not imitation-heavy."""

    relabel_ratio_stored: float = 0.4
    """Per-sample relabeling share: keep the constraint set that was actually
    stored in replay with this transition."""

    relabel_ratio_hindsight: float = 0.3
    """Per-sample relabeling share: build a fresh constraint set from the
    batch's ``next_priv`` keypoint positions (hindsight analogue)."""

    relabel_ratio_expert: float = 0.3
    """Per-sample relabeling share: build a fresh constraint set from the
    expert motion buffer's keypoint positions. Folded back into ``stored``
    when no expert buffer is present."""

    snippet_length: int = 10
    """Number of frames in a style snippet."""

    num_learning_epochs: int = 1
    """Number of passes over the rollout buffer per update."""

    mini_batch_size: int = 512
    """Mini-batch size for training."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    updates_per_step: int = 1
    """Number of gradient updates per rollout collection."""

    grad_penalty_weight: float = 10.0
    """WGAN-GP coefficient for the style discriminator."""

    replay_capacity_per_env: int | None = None
    """Off-policy replay capacity per env. When ``None``, the replay buffer is
    sized to the rollout length (pure on-policy behaviour — each transition is
    used once and overwritten). Set this larger (e.g. 2048) to enable true
    replay: the circular buffer will hold ``num_envs * capacity`` transitions
    and update() will sample from the full buffer every iteration.

    Recommended: large enough that each env's replay spans ~30-60 s of control
    (BFM-Zero equivalent ≈ 100k frames per env). With ``num_envs=4096`` and
    control dt ≈ 20 ms, 2048 capacity ≈ 40 s/env ≈ 8.4M total transitions.
    Must be >= num_steps_per_env."""

    replay_device: str | None = None
    """Device that holds the replay tensors. Defaults to the training device.
    Set to ``"cpu"`` (recommended) to keep a large replay off the GPU — BFM-Zero
    calls this ``buffer_device``. Sampled batches are moved to the training
    device with non-blocking pinned-memory transfers."""

    num_seed_steps: int = 0
    """Total env transitions collected with random actions before updates
    begin. Mirrors BFM's ``num_seed_steps=50_000``. Scales naturally with
    ``num_envs`` — a seed of 50k means ~12 iters at num_envs=4096 with
    num_steps_per_env=1. Transitions during seed are still written to the
    replay buffer."""

    num_updates_per_iter: int | None = None
    """Number of gradient updates (each = one full SAC-style update block:
    discriminator, successor critics, style critics, actor, soft target) per
    training iteration. When ``None``, the legacy on-policy behaviour is used
    (one shuffled pass × num_learning_epochs). For replay training, set this
    explicitly — BFM-Zero uses 16."""

    expert_dataset_path: str | None = None
    """Path to a precomputed expert dataset (``.pt``) produced by
    ``scripts/precompute_expert_dataset.py``. When ``None`` the style branch
    (discriminator + style critics) is disabled."""

    expert_dataset_device: str | None = None
    """Device to hold the expert buffer on. Defaults to ``cpu``. Use ``cuda``
    if the dataset fits comfortably in GPU memory and you want faster sampling."""


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

    max_checkpoint_num: int = 10
    """The maximum number of checkpoints to save."""

    action_clip_range: list[float, float] = [-50.0, 50.0]
    """The action clip range. Default is [-50.0, 50.0]."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg | RslRlDistillationStudentTeacherCfg | RslRlSparseSuccessorPolicyCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg | RslRlDistillationAlgorithmCfg | RslRlSparseSuccessorAlgorithmCfg = MISSING
    """The algorithm configuration."""

    clip_actions: float | None = None
    """The clipping value for actions. If ``None``, then no clipping is done.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    save_interval: int = MISSING
    """The number of iterations between saves."""

    upload_checkpoint: bool = False
    """Whether to upload the checkpoint to the cloud. Default is False."""

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

    smp_cfg: RslRlSmpCfg | None = None
    """The configuration for the Score-Matching Motion Prior (SMP) module. Default is None,
    in which case SMP is not used.
    """