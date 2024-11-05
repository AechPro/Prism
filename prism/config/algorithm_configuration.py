from dataclasses import dataclass


@dataclass
class Config(object):
    env_name: str
    num_processes: int
    shared_memory_num_floats_per_process: int
    training_reward_ema: float
    timestep_limit: int
    timesteps_per_iteration: int
    timesteps_between_evaluations: int
    evaluation_timestep_horizon: int
    timesteps_per_report: int
    episode_timestep_limit: int

    distributional_loss_weight: float
    q_loss_weight: float
    batch_size: int
    gamma: float
    learning_rate: float
    max_grad_norm: float

    reward_clipping_type: str
    loss_squish_fn_id: str

    use_adam: bool
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float

    use_rmsprop: bool
    rmsprop_alpha: float
    rmsprop_epsilon: float

    q_loss_fn: str

    embedding_model_final_dim: int
    embedding_model_layer_sizes: int
    embedding_model_num_layers: int
    embedding_model_type: str
    embedding_model_act_fn_id: str

    use_ids: bool
    ids_use_random_samples: bool
    ids_beta: float
    ids_lambda: float
    ids_n_q_heads: int
    ids_q_head_feature_dim: int
    ids_n_q_head_model_layers: int
    ids_allow_distributional_gradients: bool
    ids_rho_lower_bound: float
    ids_epsilon: float
    ids_ensemble_variation_coef: float

    use_e_greedy: bool
    e_greedy_initial_epsilon: float
    e_greedy_final_epsilon: float
    e_greedy_decay_timesteps: int

    n_step_returns_length: int
    use_layer_norm: bool

    use_experience_replay: bool
    experience_replay_capacity: int
    num_initial_random_timesteps: int

    use_per: bool
    per_alpha: float
    per_beta_start: float
    per_beta_end: float
    per_beta_anneal_timesteps: int

    use_iqn: bool
    iqn_n_current_state_quantile_samples: int
    iqn_n_next_state_quantile_samples: int
    iqn_quantile_samples_per_action: int
    iqn_n_basis_elements: int
    iqn_quantile_model_feature_dim: int
    iqn_quantile_model_layers: int
    iqn_huber_loss_kappa: float
    iqn_risk_policy_id: str

    use_dqn: bool
    dqn_n_model_layers: int
    dqn_n_model_feature_dim: int

    use_c51: bool

    use_double_q_learning: bool
    use_target_network: bool
    target_update_period: int

    seed: int
    hours_per_checkpoint: float
    checkpoint_dir: str
    log_to_wandb: bool
    wandb_group_name: str
    wandb_run_name: str
    wandb_project_name: str
    device: str
    env_device: str
    use_cuda_graph: bool
    render: bool

    frame_stack_size: int
    atari_sticky_actions_prob: float
    atari_noops: int