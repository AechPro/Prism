from prism.config import Config, DEFAULT_CONFIG

ROCKET_LEAGUE_CONFIG = Config(**DEFAULT_CONFIG.__dict__)
ROCKET_LEAGUE_CONFIG.shared_memory_num_floats_per_process = 100_000
ROCKET_LEAGUE_CONFIG.env_name = "rocket_league"
ROCKET_LEAGUE_CONFIG.render = True
ROCKET_LEAGUE_CONFIG.experience_replay_capacity = 10_000_000
ROCKET_LEAGUE_CONFIG.num_initial_random_timesteps = 500_000

ROCKET_LEAGUE_CONFIG.evaluation_timestep_horizon = 100_000
ROCKET_LEAGUE_CONFIG.timesteps_per_report = 100_000
ROCKET_LEAGUE_CONFIG.timesteps_between_evaluations = 1_000_000
ROCKET_LEAGUE_CONFIG.timesteps_per_iteration = 32
ROCKET_LEAGUE_CONFIG.batch_size = 32
ROCKET_LEAGUE_CONFIG.gamma = 0.995

ROCKET_LEAGUE_CONFIG.frame_stack_size = 1
ROCKET_LEAGUE_CONFIG.timestep_limit = 500_000_000_000
ROCKET_LEAGUE_CONFIG.per_beta_anneal_timesteps = 500_000_000_000

ROCKET_LEAGUE_CONFIG.embedding_model_type = "ffnn"
ROCKET_LEAGUE_CONFIG.embedding_model_num_layers = 3
ROCKET_LEAGUE_CONFIG.embedding_model_layer_sizes = 1024
ROCKET_LEAGUE_CONFIG.embedding_model_final_dim = 1024
ROCKET_LEAGUE_CONFIG.embedding_model_act_fn_id = "relu"

ROCKET_LEAGUE_CONFIG.use_adam = True
ROCKET_LEAGUE_CONFIG.q_loss_fn = "mse"
ROCKET_LEAGUE_CONFIG.learning_rate = 3e-4

ROCKET_LEAGUE_CONFIG.n_step_returns_length = 3

ROCKET_LEAGUE_CONFIG.use_ids = True
ROCKET_LEAGUE_CONFIG.ids_n_q_head_model_layers = 3
ROCKET_LEAGUE_CONFIG.ids_n_q_heads = 10
ROCKET_LEAGUE_CONFIG.ids_q_head_feature_dim = 1024
ROCKET_LEAGUE_CONFIG.ids_ensemble_variation_coef = 1e-5

ROCKET_LEAGUE_CONFIG.use_iqn = True
ROCKET_LEAGUE_CONFIG.iqn_n_current_state_quantile_samples = 32
ROCKET_LEAGUE_CONFIG.iqn_n_next_state_quantile_samples = 32
ROCKET_LEAGUE_CONFIG.iqn_quantile_samples_per_action = 32
ROCKET_LEAGUE_CONFIG.iqn_n_basis_elements = 64
ROCKET_LEAGUE_CONFIG.iqn_quantile_model_feature_dim = 1024
ROCKET_LEAGUE_CONFIG.iqn_quantile_model_layers = 2

ROCKET_LEAGUE_CONFIG.use_double_q_learning = False
ROCKET_LEAGUE_CONFIG.use_target_network = True
ROCKET_LEAGUE_CONFIG.target_update_period = 10_000

ROCKET_LEAGUE_CONFIG.reward_clipping_type = None
ROCKET_LEAGUE_CONFIG.loss_squish_fn_id = "none"

ROCKET_LEAGUE_CONFIG.use_cuda_graph = True
ROCKET_LEAGUE_CONFIG.use_layer_norm = True
ROCKET_LEAGUE_CONFIG.num_processes = 16

ROCKET_LEAGUE_CONFIG.wandb_run_name = "debug"
ROCKET_LEAGUE_CONFIG.wandb_project_name = "prism"
ROCKET_LEAGUE_CONFIG.wandb_group_name = "rocket_league"
