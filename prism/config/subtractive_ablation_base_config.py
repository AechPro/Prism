from prism.config import Config, DEFAULT_CONFIG
SUBTRACTIVE_ABLATION_BASE_CONFIG = Config(**DEFAULT_CONFIG.__dict__)

# IQN
SUBTRACTIVE_ABLATION_BASE_CONFIG.use_iqn = True
SUBTRACTIVE_ABLATION_BASE_CONFIG.iqn_n_current_state_quantile_samples = 32
SUBTRACTIVE_ABLATION_BASE_CONFIG.iqn_n_next_state_quantile_samples = 32
SUBTRACTIVE_ABLATION_BASE_CONFIG.iqn_quantile_samples_per_action = 32
SUBTRACTIVE_ABLATION_BASE_CONFIG.iqn_n_basis_elements = 64
SUBTRACTIVE_ABLATION_BASE_CONFIG.iqn_quantile_model_feature_dim = 256
SUBTRACTIVE_ABLATION_BASE_CONFIG.iqn_quantile_model_layers = 1

# IDS
SUBTRACTIVE_ABLATION_BASE_CONFIG.use_e_greedy = False
SUBTRACTIVE_ABLATION_BASE_CONFIG.use_ids = True
SUBTRACTIVE_ABLATION_BASE_CONFIG.ids_n_q_head_model_layers = 2
SUBTRACTIVE_ABLATION_BASE_CONFIG.ids_n_q_heads = 10
SUBTRACTIVE_ABLATION_BASE_CONFIG.ids_q_head_feature_dim = 256
SUBTRACTIVE_ABLATION_BASE_CONFIG.ids_ensemble_variation_coef = 1e-6

# LayerNorm
SUBTRACTIVE_ABLATION_BASE_CONFIG.use_layer_norm = True
SUBTRACTIVE_ABLATION_BASE_CONFIG.use_target_network = False
SUBTRACTIVE_ABLATION_BASE_CONFIG.use_double_q_learning = False

# PER
SUBTRACTIVE_ABLATION_BASE_CONFIG.use_per = True
SUBTRACTIVE_ABLATION_BASE_CONFIG.per_beta_start = 0.5
SUBTRACTIVE_ABLATION_BASE_CONFIG.per_beta_end = 0.5
SUBTRACTIVE_ABLATION_BASE_CONFIG.per_alpha = 0.5

# n-step
SUBTRACTIVE_ABLATION_BASE_CONFIG.n_step_returns_length = 3

# Basic setup
SUBTRACTIVE_ABLATION_BASE_CONFIG.embedding_model_type = "minatar_cnn"
SUBTRACTIVE_ABLATION_BASE_CONFIG.atari_sticky_actions_prob = 0.1
SUBTRACTIVE_ABLATION_BASE_CONFIG.evaluation_timestep_horizon = 100_000
SUBTRACTIVE_ABLATION_BASE_CONFIG.timesteps_per_report = 50_000
SUBTRACTIVE_ABLATION_BASE_CONFIG.timesteps_between_evaluations = 1_000_000
SUBTRACTIVE_ABLATION_BASE_CONFIG.timestep_limit = 3_000_000
SUBTRACTIVE_ABLATION_BASE_CONFIG.experience_replay_capacity = 3_000_000
SUBTRACTIVE_ABLATION_BASE_CONFIG.learning_rate = 0.0001
SUBTRACTIVE_ABLATION_BASE_CONFIG.gamma = 0.99
SUBTRACTIVE_ABLATION_BASE_CONFIG.batch_size = 64
SUBTRACTIVE_ABLATION_BASE_CONFIG.timesteps_per_iteration = 4
SUBTRACTIVE_ABLATION_BASE_CONFIG.num_initial_random_timesteps = 4_000
SUBTRACTIVE_ABLATION_BASE_CONFIG.frame_stack_size = 1
SUBTRACTIVE_ABLATION_BASE_CONFIG.adam_epsilon = 0.0003125
SUBTRACTIVE_ABLATION_BASE_CONFIG.loss_squish_fn_id = "none",
SUBTRACTIVE_ABLATION_BASE_CONFIG.reward_clipping_type = "dopamine clamp",

SUBTRACTIVE_ABLATION_BASE_CONFIG.log_to_wandb = True
SUBTRACTIVE_ABLATION_BASE_CONFIG.render = False
SUBTRACTIVE_ABLATION_BASE_CONFIG.wandb_group_name = "Prism Subtractive Ablation Experiment"
SUBTRACTIVE_ABLATION_BASE_CONFIG.num_processes = 16