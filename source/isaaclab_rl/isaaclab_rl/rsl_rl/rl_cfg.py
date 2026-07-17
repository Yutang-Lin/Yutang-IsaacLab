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
    """Projection applied to ``z_C`` after the weighted linear aggregation
    inside ``ConstraintSetEncoder`` (which has no learnable post-pool MLP
    — the latent is a pure additive composition of per-atom embeddings).

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
    """Discount factor (applied to the successor critic and the aux critic)."""

    gamma_style: float | None = None
    """Optional separate discount factor for the style critic. When ``None``
    (default), ``gamma`` is used for style too. Setting this to a smaller
    value (e.g. 0.98) halves the geometric fixed point of the log-odds style
    reward and stops ``q_style_mean`` from drifting unboundedly negative
    when the discriminator briefly wins. BFM-Zero effectively uses 0.98
    here (matching its global discount)."""

    target_tau: float = 0.005
    """Soft target update rate."""

    lambda_style: float = 0.05
    """Weight of the style Q-value in the actor objective. BFM's
    ``reg_coeff = 0.05``. When ``scale_lambda_by_q_track`` is enabled
    (default), the effective per-step weight is
    ``lambda_style × |q_track|.abs().mean()`` so the style branch
    stays proportionate to the task Q as the satisfaction-reward scale
    drifts during training."""

    lambda_aux: float = 0.02
    """Weight of the auxiliary-env-reward Q-value in the actor objective.
    BFM's ``reg_coeff_aux = 0.02``. Same adaptive scaling as
    ``lambda_style``. Set to 0 to disable the aux branch entirely
    (env rewards will still be logged but not consumed by training)."""

    scale_lambda_by_q_track: bool = True
    """Adaptive BFM-style ``scale_reg``. When True, multiply
    ``lambda_style`` / ``lambda_aux`` by the detached running mean of
    ``|q_track|`` so the non-task branches never dominate the actor
    gradient regardless of how the task Q magnitude evolves. When
    False, ``lambda_*`` are absolute fixed coefficients (legacy)."""

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
    """**Deprecated** — retained only so older configs import cleanly.
    The rollout source mixture now lives in the two
    ``rollout_{replay,expert}_fraction`` knobs below."""

    rollout_replay_fraction: float = 0.4
    """Per-env source share for fresh rollout chunks: sample a safe
    future anchor from the replay buffer (guaranteed populated + no
    episode-boundary crossing) and build a per-atom future-grounded C
    from the env's REALIZED future priv over ``tau_max`` steps."""

    rollout_expert_fraction: float = 0.6
    """Per-env source share for fresh rollout chunks: per-atom future-
    grounded C from the expert motion buffer's keypoint window. Paired
    with the rollout_replay source — both are fully future-grounded."""

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

    ortho_coef: float = 100.0
    """Weight of the z_C orthonormality regulariser added to the successor
    critic loss. Pushes ``z_C @ z_C.T`` toward the identity so different
    atomic-constraint sets don't collapse onto a low-dimensional subspace.
    Matches BFM-Zero FB-CPR's ``ortho_coef=100`` on the BackwardMap output.
    Set 0.0 to disable."""

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

    eval_interval_env_steps: int = 0
    """Run the sparse-constraint tracking eval every this many env steps.
    0 disables eval. The eval queries the replay buffer to report per-τ and
    per-keypoint β-normalised position error. Cheap (a few hundred ms) —
    safe to set to e.g. 500_000 or 1_000_000 env steps."""

    eval_num_samples_per_bucket: int = 512
    """Number of (time, env) anchor points sampled per τ bucket by
    ``evaluate_tracking``."""

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


@configclass
class RslRlFBCprPolicyCfg:
    """Configuration for BFM-Zero's FB-CPR-Aux policy networks.

    Mirrors :class:`isaaclab_rl.rsl_rl.modules.fb_cpr_policy.FBCprNetworkCfg`
    with our runner/configclass conventions. Defaults match BFM-Zero's
    production ``train_bfm_zero()`` overrides (z_dim=256, residual arch,
    6 hidden layers, hidden_dim=2048, num_parallel=2, etc.)."""

    class_name: str = "FBCprAuxPolicy"

    # Latent dimension
    z_dim: int = 256
    norm_z: bool = True

    # Backward map (B)
    backward_hidden_dim: int = 256
    backward_hidden_layers: int = 1
    backward_norm: bool = True
    backward_model: str = "simple"  # {"simple", "residual"} — residual = LN residual MLP
    backward_input_keys: tuple[str, ...] = ("state", "privileged_state")

    # Forward map (F)
    forward_hidden_dim: int = 2048
    forward_model: Literal["residual", "simple"] = "residual"
    forward_hidden_layers: int = 6
    forward_embedding_layers: int = 2
    forward_num_parallel: int = 2
    forward_input_keys: tuple[str, ...] = (
        "state",
        "privileged_state",
        "last_action",
        "history_actor",
    )
    forward_gamma_embed_dim: int = 0  # >0 => gamma-conditioned F(s,z,a,gamma)

    # Actor
    actor_hidden_dim: int = 2048
    actor_model: Literal["residual", "simple"] = "residual"
    actor_hidden_layers: int = 6
    actor_embedding_layers: int = 2
    actor_std: float = 0.05
    actor_input_keys: tuple[str, ...] = (
        "state",
        "last_action",
        "history_actor",
    )

    # Critic (twin Q for discriminator reward)
    critic_hidden_dim: int = 2048
    critic_model: Literal["residual", "simple"] = "residual"
    critic_hidden_layers: int = 6
    critic_embedding_layers: int = 2
    critic_num_parallel: int = 2
    critic_input_keys: tuple[str, ...] = (
        "state",
        "privileged_state",
        "last_action",
        "history_actor",
    )
    # Quantile-regression distributional critic (Dabney et al. 2018). When
    # True, critic head emits ``critic_n_quantiles`` outputs in place of
    # the scalar Q; trained with quantile Huber loss. Actor consumes the
    # quantile mean for its policy-gradient objective. Off by default
    # (scalar Q); existing ckpts load unchanged.
    critic_distributional: bool = False
    critic_n_quantiles: int = 51
    critic_huber_kappa: float = 1.0

    # Aux critic (twin Q for aux env reward)
    aux_critic_hidden_dim: int = 2048
    aux_critic_model: Literal["residual", "simple"] = "residual"
    aux_critic_hidden_layers: int = 6
    aux_critic_embedding_layers: int = 2
    aux_critic_num_parallel: int = 2
    aux_critic_input_keys: tuple[str, ...] = (
        "state",
        "privileged_state",
        "last_action",
        "history_actor",
    )
    aux_critic_distributional: bool = False
    aux_critic_n_quantiles: int = 51
    aux_critic_huber_kappa: float = 1.0

    # Discriminator
    discriminator_hidden_dim: int = 1024
    discriminator_hidden_layers: int = 3
    discriminator_input_keys: tuple[str, ...] = (
        "state",
        "privileged_state",
    )
    discriminator_zero_obs_tail_dims: int = 0
    """Zero out the last N dims of the disc's concat-filtered obs at the
    input boundary. For BFM-Zero's standard obs layout (``state`` +
    ``privileged_state``) the last 93 dims are ``local_body_ang_vel``.
    Setting this to 93 masks that block so the disc cannot separate
    policy vs expert on the ω-distribution gap (sim PhysX ω vs mocap
    spline-derived ω). Base ``root_ang_vel`` (inside ``state``) is
    unaffected. Default 0 = disabled."""

    # Obs normalizer: per-key BatchNorm1d (affine=False) momentums.
    obs_normalizer_momentum: dict[str, float] = MISSING

    obs_normalizer_allow_mismatching_keys: bool = True

    # EMA aux-reward normalizer
    aux_reward_normalizer_translate: bool = False
    aux_reward_normalizer_scale: bool = True

    # Sequence length for expert encoding mean
    seq_length: int = 8

    # Reconstruction head (end-effector pos + rot6d decoded from z).
    # Each entry is ``(obs_key, start, end)`` — a half-open slice of the
    # named obs vector (e.g. ``priv_max_local_self`` slots for wrists/ankles).
    # Concatenated in list order; paired with
    # ``RslRlFBCprAlgorithmCfg.recons_coeff`` to scale the MSE loss.
    recon_targets: tuple[tuple[str, int, int], ...] = ()
    recon_hidden_dim: int = 256
    recon_hidden_layers: int = 2
    recon_model: str = "simple"  # {"simple", "residual"} — residual = same LN residual MLP as B (ignored if recon_linear)
    recon_linear: bool = False
    recon_square_augment: bool = False
    recon_target_scale: float = 1.0

    # Manifold attractor
    manifold_attractor: bool = False
    manifold_attractor_hidden_dim: int = 1024
    manifold_attractor_hidden_layers: int = 3
    manifold_attractor_input_keys: tuple[str, ...] = ("state", "privileged_state")

    # Soft FB
    soft_fb: bool = False
    entropy_critic_hidden_dim: int = 1024
    entropy_critic_hidden_layers: int = 3
    entropy_critic_input_keys: tuple[str, ...] = (
        "state", "privileged_state", "last_action", "history_actor",
    )
    actor_learned_std: bool = False
    actor_min_std: float = 0.01
    actor_max_std: float = 0.25


@configclass
class RslRlFBCprAlgorithmCfg:
    """Configuration for BFM-Zero's FB-CPR-Aux algorithm.

    Mirrors :class:`isaaclab_rl.rsl_rl.algorithms.fb_cpr.FBCprAuxAlgorithmCfg`.
    Defaults match BFM-Zero's production ``train_bfm_zero()`` overrides."""

    class_name: str = "FBCprAux"

    # Learning rates
    lr_f: float = 3e-4
    lr_b: float = 1e-5
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_aux_critic: float = 3e-4
    lr_discriminator: float = 1e-5

    lr_scale_with_world_size: bool = True
    """Whether startup LRs and coupled normalizer EMA rates include the
    sqrt(world_size) factor. Disable for exact base-LR DDP parity runs."""

    lr_scale_with_batch_size: bool = True
    """Whether the startup LR (and coupled momentum/EMA-tau) scaling includes the
    sqrt(batch_size / 1024) factor. When True (default) the multiplier is
    sqrt(world_size) * sqrt(batch_size/1024). Set False to scale by
    sqrt(world_size) ONLY — use when batch_size is set to num_envs (or otherwise
    off the 1024 reference) and you do NOT want the LR to chase batch_size."""

    obs_normalizer_scale_momentum: bool = True
    """Whether LR scaling also speeds up observation BatchNorm running moments."""

    target_tau_scale_with_world_size: bool = False
    """Whether FB and critic target-network Polyak rates scale with
    sqrt(world_size)."""

    target_tau_world_size_cap: int = 0
    """Maximum world size used for target-network Polyak-rate scaling. Values
    <= 0 leave the world-size contribution uncapped."""

    # LR anneling. When ``lr_anneal_enable=True`` and ``lr_anneal_steps>0``,
    # every optimizer's LR linearly decays from the DDP-scaled start
    # (``sqrt(world_size) * base_lr``) to the un-scaled ``base_lr`` over
    # ``lr_anneal_steps`` GLOBAL env-steps (i.e. the counter summed across
    # ranks — same schedule regardless of world_size). No-op under single-
    # rank (start == base). Both default off.
    lr_anneal_enable: bool = False
    lr_anneal_steps: int = 0

    # Optim
    weight_decay: float = 0.0
    weight_decay_discriminator: float = 0.0
    clip_grad_norm: float = 0.0

    # Target Polyak rates
    fb_target_tau: float = 0.01
    critic_target_tau: float = 0.005

    # Pessimism penalties
    fb_pessimism_penalty: float = 0.0
    critic_pessimism_penalty: float = 0.5
    aux_critic_pessimism_penalty: float = 0.5
    actor_pessimism_penalty: float = 0.5

    # TD3-style action-noise clip
    stddev_clip: float = 0.3

    # ISOLATION TEST (transformer actor): score the parallel actor loss at the
    # CURRENT token only, while still running the full H+1 transformer forward.
    # Isolates whether the Q_disc/Q_aux runaway is the past-token (parallel)
    # scoring vs the transformer/frame_norm/window. Default False.
    actor_score_current_only: bool = False

    # FB loss regularizers
    ortho_coef: float = 100.0
    q_loss_coef: float = 0.0
    recons_coeff: float = 0.0
    """MSE weight for the reconstruction head (end-effector pos+rot6d decoded
    from ``z = B(goal)``). Requires ``policy.recon_targets`` to be non-empty.
    Set to 0 to disable even when the head is built."""

    # gamma-conditioned F (requires policy.forward_gamma_embed_dim > 0)
    fb_gamma_conditioned: bool = False
    actor_gamma_short: float = 0.8
    actor_gamma_short_alpha: float = 0.5
    fb_stochastic_integral: bool = False  # softmax-weighted horizon integral actor FB term
    fb_integral_K: int = 8
    fb_integral_align_gamma: float = 0.98  # integral Q *= 1/(1-this) for scale alignment
    fb_integral_adaptive_tau: bool = False  # tau=sqrt(abs(mean target N)), clamped to >=1
    fb_integral_prior_lambda: float = 0.0
    """Exponential SI prior strength over ``h=-log(1-gamma)``. The actor adds
    ``-lambda * (h-h_min)`` to its weight logits, so positive values prefer
    shorter horizons. Zero preserves the original SI softmax."""

    fb_gamma_innovation_align: bool = False
    """Whether the FB update samples a second gamma and aligns the two Bellman
    innovations for each transition."""

    fb_gamma_innovation_align_coef: float = 1.0
    """MSE coefficient for cross-gamma Bellman-innovation alignment."""

    length_proportional_priors: bool = True
    """When True, ``FBCprExpertBuffer`` scales the initial uniform priors AND
    any ``update_priorities()`` call by per-motion length so the expected
    per-transition draw probability stays uniform across motions regardless
    of clip-length imbalance. Default True — important for datasets that
    mix long continuous motions with short clips (e.g. the dynamic 149-s
    motion alongside 8-13s LAFAN clips)."""

    distributed_expert: bool = False
    """When True AND the runner is under DDP (world_size>1), each rank
    loads only a disjoint shard of the expert dataset (seeded random
    permutation of motion IDs → rank[r::W]). Cuts per-rank GPU memory
    linearly with world_size; required for datasets bigger than fits on
    one GPU. Tracking-eval metrics are all-reduced so global numbers
    are reported."""

    expert_shard_seed: int = 42
    """Rank-invariant seed for distributed expert sharding. This must remain
    identical across ranks so ``perm[rank::world_size]`` forms disjoint shards."""

    # Discriminator
    grad_penalty_discriminator: float = 10.0

    # Actor-objective coefficients
    reg_coeff: float = 0.05
    reg_coeff_aux: float = 0.02
    scale_reg: bool = True
    actor_fb_scale: float = 1.0
    """Scale applied to the actor's FB objective only. The same scaled FB
    magnitude drives ``scale_reg`` so discriminator/auxiliary terms retain
    their relative weighting. FB training targets and losses are unchanged."""

    # Main batch for FB / actor / critic / aux_critic. Rounded down to
    # a multiple of seq_length at init.
    batch_size: int = 1024
    batch_size_eq_num_envs: bool = False
    """When True, the runner overrides ``batch_size`` with the per-rank
    ``num_envs`` at construction (before the FB off-diag mask + LR scaling are
    built from it). Ties the training minibatch to the rollout width."""
    # Disc batch sized as disc_num_slices * seq_length (sampled
    # separately from main batch). When None, disc uses main batch.
    disc_num_slices: int | None = None
    # Max per-side batch for manifold attractor.
    ma_max_batch: int = 1024
    discount: float = 0.98
    relabel_ratio: float | None = 0.8
    train_goal_ratio: float = 0.2
    expert_asm_ratio: float = 0.6

    # Rollout context
    update_z_every_step: int = 100
    use_mix_rollout: bool = True
    rollout_expert_trajectories: bool = True
    rollout_expert_trajectories_length: int = 250
    rollout_expert_trajectories_percentage: float = 0.5
    terrain_variant_root_h_prob: float = 0.50
    global_fb_zero_prob: float = 0.5
    z_buffer_size: int = 8192
    tracking_T_min: int = 1
    tracking_T_max: int = 16
    tracking_T_choices: tuple[int, ...] = ()
    tracking_T_choice_probs: tuple[float, ...] = ()
    disc_fixed_T: int = 0
    """Fixed expert/discriminator z-mean horizon. Zero preserves the legacy
    coupling to tracking_T_*; positive values keep discriminator T fixed while
    rollout tracking environments randomize their mean-z horizon episodically."""
    disc_positive_full_window: bool = False

    # AMP (bf16)
    amp: bool = False

    # Perf flags (safe defaults off).
    stream_parallel_phase1: bool = False
    """Parallelize phase-1 backwards (disc + F/B + aux_critic) across CUDA
    streams. Saves ~10-15% iter-time on fast intra-node fabrics like B200
    NVSwitch by overlapping 4 otherwise-serial backward chunks. Pair with
    ``merge_phase1_reduce=True`` — streams need the merged manual reduce
    since DDP bucket hooks serialize on the main stream."""

    compile_mode: str = ""
    """``torch.compile`` mode for the 5 trainable online networks (F, B,
    actor, critic, aux_critic). Empty = disabled. Options:
    "default" | "reduce-overhead" | "max-autotune".

    IMPORTANT: On PyTorch 2.7, ``reduce-overhead`` (CUDA graphs) is
    broken when combined with user CUDA streams (pytorch#180396,
    #180497; fix in release/2.12). When ``stream_parallel_phase1=True``
    the algorithm auto-downgrades ``reduce-overhead`` → ``default``.
    Use ``"default"`` when pairing with streams on 2.7."""

    compile_forward_map: bool = True
    """Whether the online F network is included in ``torch.compile``. Disable
    selectively for workloads whose actor objective expands F to a much larger
    batch while retaining compilation for B, actor, and both critics."""

    merge_phase1_reduce: bool = False
    """Merge the 4 phase-1 allreduces into one. Wins on slow/high-latency
    fabrics (EFA without GDR). Required if ``stream_parallel_phase1=True``.
    On NVSwitch alone, losing DDP bucket overlap costs more than latency
    savings — keep False."""

    # Aux rewards: name -> scaling coefficient. Default matches BFM-Zero.
    aux_rewards_scaling: dict[str, float] = MISSING

    # Optional override for ``FBCprRunner._BFM_KEY_GROUPS`` — lets a task
    # with extra obs terms (e.g. BFM-Terrain's ``height_scan``) route them
    # into new agent-input dict keys. Leave empty to use the flat-floor
    # BFM-Zero default.
    obs_key_groups: dict[str, tuple[str, ...]] = dict()

    # Manifold attractor
    manifold_attractor: bool = False
    manifold_attractor_coeff: float = 0.05
    lr_manifold_attractor: float = 1e-5
    grad_penalty_manifold_attractor: float = 10.0

    # Soft FB
    soft_fb: bool = False
    soft_fb_entropy_coef: float = 1.0
    soft_fb_expert_future_min: tuple[float, float] = (0.5, 1.0)
    lr_entropy_critic: float = 3e-4
    entropy_critic_target_tau: float = 0.005

    # Replay / seed
    replay_capacity: int = 5_120_000
    """Total replay capacity across all envs (flat circular). BFM's 5.12M."""

    replay_device: str = "cpu"
    """Device to hold the replay buffer on."""

    recompose_history_actor: bool = False
    """Store only the newest ``history_actor`` frame in replay and rebuild the
    full ``H*frame`` window on sample (byte-exact; see FBCprReplayBuffer). Cuts
    the replay's history_actor footprint ~H×. Requires the MLP actor
    (incompatible with the transformer actor-window path). Default off."""

    num_seed_steps: int = 10_240
    """Total env transitions collected with random actions before updates
    begin. Matches BFM's ``num_seed_steps=10240``."""

    resume_num_seed_steps: int | None = None
    """Pre-update collection budget (per-rank env-steps) when RESUMING without a
    restored replay buffer (load_replay=False or the sibling .replay.pt is
    missing). The buffer starts empty; updates are held off until this many
    transitions are collected — but ON-POLICY, using the resumed policy's normal
    training rollout (exploration + z-context), NOT uniform-random actions (the
    policy is already trained, so random actions would pollute the buffer). Gives
    a more stable restart on a well-filled buffer. ``None`` -> fall back to
    ``num_seed_steps``. Only affects the resume-without-replay path; fresh runs
    and replay-restored resumes are unchanged."""

    num_agent_updates: int = 16
    """Number of gradient updates per rollout-collection trigger. BFM's 16."""

    update_agent_every: int = 1024
    """Env steps between update triggers. BFM's ``update_agent_every=1024``."""

    # Expert dataset
    expert_dataset_path: str = MISSING
    """Path to a BFM-format expert dataset (``.pt``) produced by
    ``scripts/precompute_bfm_expert_dataset.py``."""

    expert_dataset_device: str = "cuda"
    """Device to hold the expert buffer on."""

    expert_keypoint_names: list[str] | None = None
    """Optional keypoint-list override for the load-time privileged_state
    compose of a *minimal* expert dataset. When ``None`` the buffer uses the
    precompute script's 31-body ``KEYPOINT_NAMES``. A variant (BFM-One) passes
    a shorter list (e.g. MI's 26-body ``EEF_LINKS``) to drop redundant
    intermediate links from the B-encode priv — the raw motion data is
    keypoint-agnostic, so the same dataset is reused with a smaller priv dim.
    MUST match the env's ``priv_max_local_self`` ``body_names`` exactly."""

    # --- Global-through-Anchoring (BFM-One-Anchored) only ---
    store_world_pose: bool = False
    """Store per-transition world SE(2) root pose in replay (anchor relabel)."""
    anchored_pose_key: str = "anchored_pose"
    anchor_pose_clamp: float = 10.0
    anchor_alpha_gt: float = 0.34
    anchor_beta_gh: float = 0.33
    anchor_random_xy_range: float = 10.0
    anchor_frame_body: bool = False
    """Reframe privileged_state body POS/ROT6D into the per-row anchor frame
    (train+expert) so B/F/critic/disc encode globally-positioned body pose."""
    spatial_cpr_coeff: float = 1.0
    goal_future_ratio: float = 0.4
    goal_nearby_ratio: float = 0.2
    goal_replay_ratio: float = 0.2
    goal_composed_ratio: float = 0.2
    goal_nearby_radius: float = 2.0

    # BFM-style tracking eval (fires every ``eval_every_steps`` env-steps).
    eval_every_steps: int = 9_600_000
    """Env-step interval between tracking evals. BFM production: 9.6M.
    Set to 0 to disable eval entirely. NO initial eval — the first eval
    fires only after ``tot_timesteps >= eval_every_steps``."""

    eval_rollout_length: int = 250
    """Per-motion tracking rollout length (steps). BFM's default 250."""

    eval_update_priorities: bool = True
    """If True, feed tracking MPJPE back into the expert buffer's
    sampling weights (prioritized RSI sampling). BFM production: True."""

    eval_priority_min: float = 0.5
    """Lower clamp for eval-based priority weights (BFM default 0.5)."""

    eval_priority_max: float = 2.0
    """Upper clamp for eval-based priority weights (BFM default 2.0)."""

    eval_priority_scale: float = 2.0
    """Scale applied to normalized MPJPE before ``exp``/``lin`` mapping
    (BFM default 2.0)."""

    eval_priority_mode: Literal["exp", "lin"] = "exp"
    """How the clamped+scaled MPJPE maps to priority weight (BFM: ``exp``)."""

    # Checkpoint pruning / replay-save cadence — read by ``FBCprRunner``.
    save_replay_every_n: int = 10
    """Save the heavy ``.replay.pt`` sibling file every ``save_replay_every_n``
    *light-save intervals* (i.e., every ``save_replay_every_n * save_interval``
    training iterations). Set to 1 to save the replay every time the
    light ckpt fires, or 0 to never save the replay."""

    keep_last_n_checkpoints: int = 10
    """Keep only the ``N`` newest ``model_<iter>.pt`` files (and their
    paired ``.replay.pt`` siblings). ``model_best.pt`` is never pruned.
    Set to 0 to disable pruning."""


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

    policy: RslRlPpoActorCriticCfg | RslRlDistillationStudentTeacherCfg | RslRlSparseSuccessorPolicyCfg | RslRlFBCprPolicyCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg | RslRlDistillationAlgorithmCfg | RslRlSparseSuccessorAlgorithmCfg | RslRlFBCprAlgorithmCfg = MISSING
    """The algorithm configuration."""

    clip_actions: float | None = None
    """The clipping value for actions. If ``None``, then no clipping is done.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    save_interval: int = MISSING
    """The number of iterations between saves."""

    log_env_steps_world_size_cap: int = 0
    """Maximum world-size contribution to the logged env-step x-axis. Values
    <= 0 use the full world size."""

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

    load_checkpoint: str = r"^model_[0-9]+\.pt$"
    """The checkpoint file to load. Default matches numbered model checkpoints
    while excluding their ``.replay.pt`` siblings.

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
