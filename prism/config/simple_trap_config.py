from prism.config import Config, DEFAULT_CONFIG

SIMPLE_TRAP_CONFIG = Config(**DEFAULT_CONFIG.__dict__)
SIMPLE_TRAP_CONFIG.shared_memory_num_floats_per_process = 100_000
SIMPLE_TRAP_CONFIG.env_name = "simple_trap"
SIMPLE_TRAP_CONFIG.render = True
SIMPLE_TRAP_CONFIG.experience_replay_capacity = 5_000_000
SIMPLE_TRAP_CONFIG.num_initial_random_timesteps = 5000

SIMPLE_TRAP_CONFIG.evaluation_timestep_horizon = 1_000
SIMPLE_TRAP_CONFIG.timesteps_per_report = 1_000
SIMPLE_TRAP_CONFIG.timesteps_between_evaluations = 10_000

SIMPLE_TRAP_CONFIG.frame_stack_size = 1
SIMPLE_TRAP_CONFIG.timestep_limit = 5_000_000
SIMPLE_TRAP_CONFIG.per_beta_anneal_timesteps = 5_000_000

SIMPLE_TRAP_CONFIG.embedding_model_type = "ffnn"
SIMPLE_TRAP_CONFIG.embedding_model_num_layers = 2
SIMPLE_TRAP_CONFIG.embedding_model_layer_sizes = 1024
SIMPLE_TRAP_CONFIG.embedding_model_final_dim = 1024

SIMPLE_TRAP_CONFIG.use_adam = True
SIMPLE_TRAP_CONFIG.use_rmsprop = False
SIMPLE_TRAP_CONFIG.q_loss_fn = "mse"
SIMPLE_TRAP_CONFIG.learning_rate = 0.0001

SIMPLE_TRAP_CONFIG.use_per = False
SIMPLE_TRAP_CONFIG.n_step_returns_length = 3

SIMPLE_TRAP_CONFIG.use_ids = True
SIMPLE_TRAP_CONFIG.ids_n_q_head_model_layers = 2
SIMPLE_TRAP_CONFIG.ids_n_q_heads = 10
SIMPLE_TRAP_CONFIG.ids_q_head_feature_dim = 256
SIMPLE_TRAP_CONFIG.ids_ensemble_variation_coef = 1e-6

SIMPLE_TRAP_CONFIG.use_e_greedy = False
SIMPLE_TRAP_CONFIG.e_greedy_decay_timesteps = 250000
SIMPLE_TRAP_CONFIG.e_greedy_final_epsilon = 0.1

SIMPLE_TRAP_CONFIG.use_iqn = True
SIMPLE_TRAP_CONFIG.iqn_n_current_state_quantile_samples = 32
SIMPLE_TRAP_CONFIG.iqn_n_next_state_quantile_samples = 32
SIMPLE_TRAP_CONFIG.iqn_quantile_samples_per_action = 32
SIMPLE_TRAP_CONFIG.iqn_n_basis_elements = 64
SIMPLE_TRAP_CONFIG.iqn_quantile_model_feature_dim = 256
SIMPLE_TRAP_CONFIG.iqn_quantile_model_layers = 1

SIMPLE_TRAP_CONFIG.use_double_q_learning = False
SIMPLE_TRAP_CONFIG.use_target_network = False
SIMPLE_TRAP_CONFIG.target_update_period = 4000

SIMPLE_TRAP_CONFIG.use_dqn = False
SIMPLE_TRAP_CONFIG.dqn_n_model_layers = 1
SIMPLE_TRAP_CONFIG.dqn_n_model_feature_dim = 1024

SIMPLE_TRAP_CONFIG.reward_clipping_type = None
SIMPLE_TRAP_CONFIG.loss_squish_fn_id = "none"
SIMPLE_TRAP_CONFIG.embedding_model_act_fn_id = "relu"

SIMPLE_TRAP_CONFIG.atari_sticky_actions_prob = 0.1
SIMPLE_TRAP_CONFIG.timesteps_per_iteration = 1

SIMPLE_TRAP_CONFIG.use_cuda_graph = True
SIMPLE_TRAP_CONFIG.use_layer_norm = True
SIMPLE_TRAP_CONFIG.num_processes = 4

