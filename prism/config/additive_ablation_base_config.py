from prism.config import Config, DEFAULT_CONFIG
ADDITIVE_ABLATION_BASE_CONFIG = Config(**DEFAULT_CONFIG.__dict__)

ADDITIVE_ABLATION_BASE_CONFIG.use_per = False
ADDITIVE_ABLATION_BASE_CONFIG.use_ids = False
ADDITIVE_ABLATION_BASE_CONFIG.use_dqn = False
ADDITIVE_ABLATION_BASE_CONFIG.use_double_q_learning = False
ADDITIVE_ABLATION_BASE_CONFIG.use_layer_norm = False
ADDITIVE_ABLATION_BASE_CONFIG.dqn_n_model_feature_dim = 256
ADDITIVE_ABLATION_BASE_CONFIG.dqn_n_model_layers = 2

# IQN
ADDITIVE_ABLATION_BASE_CONFIG.use_iqn = True
ADDITIVE_ABLATION_BASE_CONFIG.use_target_network = True
ADDITIVE_ABLATION_BASE_CONFIG.use_e_greedy = True
ADDITIVE_ABLATION_BASE_CONFIG.e_greedy_initial_epsilon = 1.0
ADDITIVE_ABLATION_BASE_CONFIG.e_greedy_decay_timesteps = 250_000
ADDITIVE_ABLATION_BASE_CONFIG.e_greedy_final_epsilon = 0.01
ADDITIVE_ABLATION_BASE_CONFIG.n_step_returns_length = 1
ADDITIVE_ABLATION_BASE_CONFIG.target_update_period = 4_000
ADDITIVE_ABLATION_BASE_CONFIG.iqn_n_current_state_quantile_samples = 32
ADDITIVE_ABLATION_BASE_CONFIG.iqn_n_next_state_quantile_samples = 32
ADDITIVE_ABLATION_BASE_CONFIG.iqn_quantile_samples_per_action = 32
ADDITIVE_ABLATION_BASE_CONFIG.iqn_n_basis_elements = 64
ADDITIVE_ABLATION_BASE_CONFIG.iqn_quantile_model_feature_dim = 256
ADDITIVE_ABLATION_BASE_CONFIG.iqn_quantile_model_layers = 1

# Basic setup
ADDITIVE_ABLATION_BASE_CONFIG.embedding_model_type = "minatar_cnn"
ADDITIVE_ABLATION_BASE_CONFIG.atari_sticky_actions_prob = 0.1
ADDITIVE_ABLATION_BASE_CONFIG.evaluation_timestep_horizon = 100_000
ADDITIVE_ABLATION_BASE_CONFIG.timesteps_per_report = 50_000
ADDITIVE_ABLATION_BASE_CONFIG.timesteps_between_evaluations = 1_000_000
ADDITIVE_ABLATION_BASE_CONFIG.timestep_limit = 3_000_000
ADDITIVE_ABLATION_BASE_CONFIG.experience_replay_capacity = 3_000_000
ADDITIVE_ABLATION_BASE_CONFIG.learning_rate = 0.0001
ADDITIVE_ABLATION_BASE_CONFIG.gamma = 0.99
ADDITIVE_ABLATION_BASE_CONFIG.batch_size = 64
ADDITIVE_ABLATION_BASE_CONFIG.timesteps_per_iteration = 4
ADDITIVE_ABLATION_BASE_CONFIG.num_initial_random_timesteps = 4_000
ADDITIVE_ABLATION_BASE_CONFIG.frame_stack_size = 1
ADDITIVE_ABLATION_BASE_CONFIG.adam_epsilon = 0.0003125
ADDITIVE_ABLATION_BASE_CONFIG.loss_squish_fn_id = "none",
ADDITIVE_ABLATION_BASE_CONFIG.reward_clipping_type = "dopamine clamp",

ADDITIVE_ABLATION_BASE_CONFIG.log_to_wandb = True
ADDITIVE_ABLATION_BASE_CONFIG.render = False
ADDITIVE_ABLATION_BASE_CONFIG.wandb_group_name = "Second Prism Additive Ablation Experiment"
ADDITIVE_ABLATION_BASE_CONFIG.num_processes = 16
