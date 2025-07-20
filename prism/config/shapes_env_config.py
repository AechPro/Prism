from prism.config import Config, DEFAULT_CONFIG

SHAPES_ENV_CONFIG = Config(**DEFAULT_CONFIG.__dict__)
SHAPES_ENV_CONFIG.shared_memory_num_floats_per_process = 100_000
SHAPES_ENV_CONFIG.env_name = "shapes_environment"
SHAPES_ENV_CONFIG.render = True
SHAPES_ENV_CONFIG.experience_replay_capacity = 10_000_000
SHAPES_ENV_CONFIG.num_initial_random_timesteps = 1

SHAPES_ENV_CONFIG.evaluation_timestep_horizon = 100_000
SHAPES_ENV_CONFIG.timesteps_per_report = 10_000
SHAPES_ENV_CONFIG.timesteps_between_evaluations = 100_000
SHAPES_ENV_CONFIG.timesteps_per_iteration = 32000
SHAPES_ENV_CONFIG.batch_size = 32
SHAPES_ENV_CONFIG.gamma = 0.99

SHAPES_ENV_CONFIG.distributional_loss_weight = 1
SHAPES_ENV_CONFIG.q_loss_weight = 1

SHAPES_ENV_CONFIG.frame_stack_size = 1
SHAPES_ENV_CONFIG.timestep_limit = 500_000_000_000
SHAPES_ENV_CONFIG.per_beta_anneal_timesteps = 500_000_000_000

SHAPES_ENV_CONFIG.embedding_model_type = "ffnn"
SHAPES_ENV_CONFIG.embedding_model_num_layers = 3
SHAPES_ENV_CONFIG.embedding_model_layer_sizes = 1024
SHAPES_ENV_CONFIG.embedding_model_final_dim = 1024
SHAPES_ENV_CONFIG.embedding_model_act_fn_id = "relu"

SHAPES_ENV_CONFIG.use_adam = True
SHAPES_ENV_CONFIG.q_loss_fn = "mse"
SHAPES_ENV_CONFIG.learning_rate = 3e-5

SHAPES_ENV_CONFIG.use_per = False

SHAPES_ENV_CONFIG.n_step_returns_length = 3

# SHAPES_ENV_CONFIG.use_e_greedy = True
# SHAPES_ENV_CONFIG.e_greedy_initial_epsilon = 1.0
# SHAPES_ENV_CONFIG.e_greedy_decay_timesteps = 250_000
# SHAPES_ENV_CONFIG.e_greedy_final_epsilon = 0.01

SHAPES_ENV_CONFIG.use_ids = True
SHAPES_ENV_CONFIG.ids_n_q_head_model_layers = 3
SHAPES_ENV_CONFIG.ids_n_q_heads = 10
SHAPES_ENV_CONFIG.ids_q_head_feature_dim = 512
SHAPES_ENV_CONFIG.ids_ensemble_variation_coef = 1e-5

SHAPES_ENV_CONFIG.use_iqn = True
SHAPES_ENV_CONFIG.iqn_n_current_state_quantile_samples = 16
SHAPES_ENV_CONFIG.iqn_n_next_state_quantile_samples = 16
SHAPES_ENV_CONFIG.iqn_quantile_samples_per_action = 32
SHAPES_ENV_CONFIG.iqn_n_basis_elements = 64
SHAPES_ENV_CONFIG.iqn_quantile_model_feature_dim = 512
SHAPES_ENV_CONFIG.iqn_quantile_model_layers = 2

SHAPES_ENV_CONFIG.use_double_q_learning = False
SHAPES_ENV_CONFIG.use_target_network = True
SHAPES_ENV_CONFIG.target_update_period = 10_000

SHAPES_ENV_CONFIG.reward_clipping_type = None
SHAPES_ENV_CONFIG.loss_squish_fn_id = "none"

SHAPES_ENV_CONFIG.use_cuda_graph = True
SHAPES_ENV_CONFIG.device = "cuda:0"
SHAPES_ENV_CONFIG.use_layer_norm = True
SHAPES_ENV_CONFIG.num_processes = 16

SHAPES_ENV_CONFIG.wandb_run_name = "debug"
SHAPES_ENV_CONFIG.wandb_project_name = "prism"
SHAPES_ENV_CONFIG.wandb_group_name = "shapes_environment"
