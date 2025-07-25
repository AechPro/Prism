from prism.config import Config, DEFAULT_CONFIG

MINATAR_CONFIG = Config(**DEFAULT_CONFIG.__dict__)
MINATAR_CONFIG.env_name = "MinAtar/Freeway-v1"
MINATAR_CONFIG.embedding_model_type = "minatar_cnn"
MINATAR_CONFIG.experience_replay_capacity = 100_000
MINATAR_CONFIG.num_initial_random_timesteps = 5000
MINATAR_CONFIG.evaluation_timestep_horizon = 100_000
MINATAR_CONFIG.timesteps_per_report = 10_000
MINATAR_CONFIG.timesteps_between_evaluations = 10_000

MINATAR_CONFIG.frame_stack_size = 1
MINATAR_CONFIG.timestep_limit = 5_000_000
MINATAR_CONFIG.per_beta_anneal_timesteps = 5_000_000

MINATAR_CONFIG.use_adam = True
MINATAR_CONFIG.use_rmsprop = False
MINATAR_CONFIG.q_loss_fn = 'huber'
MINATAR_CONFIG.learning_rate = 0.00025

MINATAR_CONFIG.use_per = True
MINATAR_CONFIG.n_step_returns_length = 3

MINATAR_CONFIG.use_ids = True
MINATAR_CONFIG.ids_beta = 0.8
MINATAR_CONFIG.ids_q_head_feature_dim = 128
MINATAR_CONFIG.ids_use_random_samples = False

MINATAR_CONFIG.use_e_greedy = False
MINATAR_CONFIG.e_greedy_decay_timesteps = 100_000
MINATAR_CONFIG.e_greedy_final_epsilon = 0.1

MINATAR_CONFIG.use_iqn = True
MINATAR_CONFIG.iqn_quantile_model_feature_dim = 128

MINATAR_CONFIG.use_double_q_learning = False
MINATAR_CONFIG.use_target_network = False
MINATAR_CONFIG.target_update_period = 1000

MINATAR_CONFIG.use_dqn = False
MINATAR_CONFIG.dqn_n_model_layers = 1
MINATAR_CONFIG.dqn_n_model_feature_dim = 128

MINATAR_CONFIG.reward_clipping_type = "none"
MINATAR_CONFIG.loss_squish_fn_id = "none"
MINATAR_CONFIG.embedding_model_act_fn_id = "relu"
MINATAR_CONFIG.atari_sticky_actions_prob = 0.1
MINATAR_CONFIG.timesteps_per_iteration = 1

MINATAR_CONFIG.use_cuda_graph = True
MINATAR_CONFIG.use_layer_norm = True
MINATAR_CONFIG.num_processes = 8
