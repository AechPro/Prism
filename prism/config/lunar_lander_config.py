from prism.config import Config, DEFAULT_CONFIG

LUNAR_LANDER_CFG = Config(**DEFAULT_CONFIG.__dict__)
LUNAR_LANDER_CFG.env_name = "LunarLander-v2"
LUNAR_LANDER_CFG.embedding_model_type = "ffnn"
LUNAR_LANDER_CFG.loss_squish_fn_id = "obs_look_further"
LUNAR_LANDER_CFG.evaluation_timestep_horizon = 10_000
LUNAR_LANDER_CFG.timesteps_between_evaluations = 10_000
LUNAR_LANDER_CFG.num_initial_random_timesteps = 500
LUNAR_LANDER_CFG.timesteps_per_report = 1_000
LUNAR_LANDER_CFG.per_beta_anneal_timesteps = 100_000
LUNAR_LANDER_CFG.reward_clipping_type = None
LUNAR_LANDER_CFG.num_processes = 4
LUNAR_LANDER_CFG.episode_timestep_limit = 2000
LUNAR_LANDER_CFG.frame_stack_size = 1
LUNAR_LANDER_CFG.embedding_model_final_dim = 512
LUNAR_LANDER_CFG.embedding_model_layer_sizes = 512
LUNAR_LANDER_CFG.embedding_model_num_layers = 2