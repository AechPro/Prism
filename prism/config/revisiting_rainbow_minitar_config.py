from prism.config import Config, MINATAR_CONFIG

REVISITING_RAINBOW_MINATAR_CONFIG = Config(**MINATAR_CONFIG.__dict__)
REVISITING_RAINBOW_MINATAR_CONFIG.learning_rate = 0.0001
REVISITING_RAINBOW_MINATAR_CONFIG.frame_stack_size = 1
REVISITING_RAINBOW_MINATAR_CONFIG.gamma = 0.99
REVISITING_RAINBOW_MINATAR_CONFIG.n_step_returns_length = 3
REVISITING_RAINBOW_MINATAR_CONFIG.num_initial_random_timesteps = 1_000
REVISITING_RAINBOW_MINATAR_CONFIG.timesteps_per_iteration = 4
REVISITING_RAINBOW_MINATAR_CONFIG.target_update_period = 1_000
REVISITING_RAINBOW_MINATAR_CONFIG.use_adam = True
REVISITING_RAINBOW_MINATAR_CONFIG.adam_epsilon = 0.0003125
REVISITING_RAINBOW_MINATAR_CONFIG.timestep_limit = 10_000_000
REVISITING_RAINBOW_MINATAR_CONFIG.experience_replay_capacity = 100_000
REVISITING_RAINBOW_MINATAR_CONFIG.batch_size = 64
REVISITING_RAINBOW_MINATAR_CONFIG.iqn_n_current_state_quantile_samples = 32
REVISITING_RAINBOW_MINATAR_CONFIG.iqn_n_next_state_quantile_samples = 32
REVISITING_RAINBOW_MINATAR_CONFIG.iqn_quantile_samples_per_action = 32
REVISITING_RAINBOW_MINATAR_CONFIG.iqn_quantile_model_feature_dim = 64

# Here we have departed from the parameters used by (Obando-Ceron & Castro) in the revisiting Rainbow paper.
# If all the heads of the Q ensemble are linear functions of the same input, they converge to the same NN
# almost immediately and IDS can't work. To prevent this, we use 1 hidden layer with 256 nodes per head for
# both IDS and IQN. This requires us to test DQN and Rainbow for fair comparison.
REVISITING_RAINBOW_MINATAR_CONFIG.iqn_quantile_model_layers = 1
REVISITING_RAINBOW_MINATAR_CONFIG.iqn_quantile_model_feature_dim = 256

# This is 2 because we have to build one hidden layer and the output layer, so there is still only 1 hidden layer per
# head in the ensemble here.
REVISITING_RAINBOW_MINATAR_CONFIG.ids_n_q_head_model_layers = 2
REVISITING_RAINBOW_MINATAR_CONFIG.ids_q_head_feature_dim = 256

REVISITING_RAINBOW_MINATAR_CONFIG.wandb_group_name = "Revisiting Rainbow"

REVISITING_RAINBOW_MINATAR_CONFIG.env_name = "MinAtar/Breakout-v1"
REVISITING_RAINBOW_MINATAR_CONFIG.num_processes = 32


# DQN
REVISITING_RAINBOW_MINATAR_CONFIG.use_per = False
REVISITING_RAINBOW_MINATAR_CONFIG.use_ids = False
REVISITING_RAINBOW_MINATAR_CONFIG.use_iqn = False
REVISITING_RAINBOW_MINATAR_CONFIG.use_layer_norm = False
REVISITING_RAINBOW_MINATAR_CONFIG.use_double_q_learning = False
REVISITING_RAINBOW_MINATAR_CONFIG.use_dqn = True
REVISITING_RAINBOW_MINATAR_CONFIG.use_target_network = True
REVISITING_RAINBOW_MINATAR_CONFIG.use_e_greedy = True
REVISITING_RAINBOW_MINATAR_CONFIG.learning_rate = 0.00025
REVISITING_RAINBOW_MINATAR_CONFIG.e_greedy_decay_timesteps = 250_000
REVISITING_RAINBOW_MINATAR_CONFIG.e_greedy_final_epsilon = 0.01
REVISITING_RAINBOW_MINATAR_CONFIG.dqn_n_model_feature_dim = 256
REVISITING_RAINBOW_MINATAR_CONFIG.n_step_returns_length = 1
REVISITING_RAINBOW_MINATAR_CONFIG.dqn_n_model_layers = 2
REVISITING_RAINBOW_MINATAR_CONFIG.double_q_learning_target_update_period = 1_000
