from prism.config import Config

DEFAULT_CONFIG = Config(
    env_name="ALE/Defender-v5",
    num_processes=17,
    shared_memory_num_floats_per_process=100_000,
    training_reward_ema=0.9,
    timestep_limit=50_000_000,
    timesteps_per_iteration=4,
    timesteps_between_evaluations=1_000_000,  # 1M *steps* (4M frames)
    evaluation_timestep_horizon=125_000,  # 500K *frames*
    timesteps_per_report=100_000,
    episode_timestep_limit=108_000,

    run_through_redis=False,
    redis_host="localhost",
    redis_port=6379,
    redis_side="server",

    distributional_loss_weight=1,
    q_loss_weight=1,
    batch_size=32,
    gamma=0.99,
    learning_rate=6.25e-5,
    max_grad_norm=10.0,

    sparse_init_p=0.0,
    loss_squish_fn_id="none",
    reward_clipping_type="dopamine_clip",

    use_adam=True,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1.5e-4,

    use_rmsprop=False,
    rmsprop_alpha=0.95,
    rmsprop_epsilon=0.01,

    q_loss_fn="mse",  # "mse" or "huber"
    embedding_model_final_dim=3136,  # Atari network flatten dim. This will be overridden when the model is built.
    embedding_model_layer_sizes=512,
    embedding_model_num_layers=0,
    embedding_model_type="nature_atari_cnn",
    embedding_model_act_fn_id="relu",

    use_ids=True,
    ids_beta=1,
    ids_use_random_samples=False,
    ids_lambda=0.1,
    ids_n_q_heads=10,
    ids_q_head_feature_dim=512,
    ids_n_q_head_model_layers=2,
    ids_allow_distributional_gradients=True,
    ids_rho_lower_bound=0.25,
    ids_epsilon=1e-10,
    ids_ensemble_variation_coef=1e-6,

    use_e_greedy=False,
    e_greedy_initial_epsilon=1.0,
    e_greedy_final_epsilon=0.01,
    e_greedy_decay_timesteps=50_000_000,

    n_step_returns_length=3,
    use_layer_norm=True,

    use_experience_replay=True,
    experience_replay_capacity=1_000_000,
    num_initial_random_timesteps=20_000,  # 80K *FRAMES*

    use_per=False,
    per_alpha=0.5,
    per_beta_start=0.5,
    per_beta_end=0.5,
    per_beta_anneal_timesteps=1,

    use_iqn=True,
    iqn_n_current_state_quantile_samples=8,
    iqn_n_next_state_quantile_samples=8,
    iqn_quantile_samples_per_action=200,
    iqn_n_basis_elements=64,
    iqn_quantile_model_feature_dim=512,
    iqn_quantile_model_layers=1,
    iqn_risk_policy_id="neutral",
    iqn_huber_loss_kappa=1.0,

    use_dqn=False,
    dqn_n_model_layers=1,
    dqn_n_model_feature_dim=512,

    use_c51=False,

    use_double_q_learning=False,
    use_target_network=False,
    target_update_period=8_000,  # 32K *FRAMES*

    seed=123,
    hours_per_checkpoint=1.0,
    checkpoint_dir="data/checkpoints",
    log_to_wandb=False,
    wandb_group_name="debug",
    wandb_run_name="null",
    wandb_project_name="Prism",
    device="cuda:0",
    env_device="cpu",
    use_cuda_graph=True,
    render=False,

    frame_stack_size=4,
    atari_sticky_actions_prob=0,  # Not in Rainbow paper, but in the dopamine implementation of Rainbow.
    atari_noops=30,
)
