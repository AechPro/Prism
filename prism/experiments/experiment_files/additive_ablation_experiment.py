from prism.experiments import Experiment
from prism.config import ADDITIVE_ABLATION_BASE_CONFIG, Config
import os


class AdditiveAblationExperiment(Experiment):
    def __init__(self, num_seeds=5):
        self.wandb_project = "Second Prism Additive Ablation Experiment"

        super().__init__(base_config=ADDITIVE_ABLATION_BASE_CONFIG, num_seeds=num_seeds)

        # idx = 0
        # while idx < len(self.configs):
        #     cfg = self.configs[idx]
        #
        #     if cfg.wandb_group_name == "IDS + Variation MinAtar/Freeway-v1":
        #         break
        #
        #     # if "Breakout" in cfg.wandb_group_name:
        #     #     break
        #
        #     idx += 1
        # self.configs = self.configs[idx:]

    def get_next_config(self):
        return self.configs.pop(0)

    def is_done(self):
        return len(self.configs) == 0

    def _setup_configs(self):
        for env_name in self.envs_to_test:
            # # IQN
            for i in range(self.num_seeds):
                # Copy base config
                cfg = Config(**self.base_config.__dict__)

                # Modify base config
                cfg.seed = cfg.seed + i
                cfg.wandb_project_name = self.wandb_project
                cfg.wandb_group_name = "IQN {}".format(env_name)
                cfg.wandb_run_name = "Seed {}".format(cfg.seed)
                cfg.checkpoint_dir = os.path.join("data",
                                                  "{}".format(cfg.wandb_project_name),
                                                  "{}".format(cfg.wandb_group_name),
                                                  "{}".format(cfg.wandb_run_name))
                cfg.env_name = env_name
                self.configs.append(cfg)

            # IQN + n-step
            for i in range(self.num_seeds):
                cfg = Config(**self.base_config.__dict__)
                cfg.n_step_returns_length = 3
                cfg.seed = cfg.seed + i
                cfg.wandb_project_name = self.wandb_project
                cfg.wandb_group_name = "IQN + n-step {}".format(env_name)
                cfg.wandb_run_name = "Seed {}".format(cfg.seed)
                cfg.checkpoint_dir = os.path.join("data",
                                                  "{}".format(cfg.wandb_project_name),
                                                  "{}".format(cfg.wandb_group_name),
                                                  "{}".format(cfg.wandb_run_name))
                cfg.env_name = env_name
                self.configs.append(cfg)

            # IQN + PER
            for i in range(self.num_seeds):
                cfg = Config(**self.base_config.__dict__)
                cfg.use_per = True
                cfg.per_beta_start = 0.5
                cfg.per_beta_end = 0.5
                cfg.per_beta_anneal_timesteps = 1
                cfg.per_alpha = 0.5

                cfg.seed = cfg.seed + i
                cfg.wandb_project_name = self.wandb_project
                cfg.wandb_group_name = "IQN + PER {}".format(env_name)
                cfg.wandb_run_name = "Seed {}".format(cfg.seed)
                cfg.checkpoint_dir = os.path.join("data",
                                                  "{}".format(cfg.wandb_project_name),
                                                  "{}".format(cfg.wandb_group_name),
                                                  "{}".format(cfg.wandb_run_name))
                cfg.env_name = env_name
                self.configs.append(cfg)

            # IQN + LayerNorm - target model
            for i in range(self.num_seeds):
                cfg = Config(**self.base_config.__dict__)
                cfg.use_layer_norm = True
                cfg.use_target_network = False

                cfg.seed = cfg.seed + i
                cfg.wandb_project_name = self.wandb_project
                cfg.wandb_group_name = "IQN + LayerNorm - target {}".format(env_name)
                cfg.wandb_run_name = "Seed {}".format(cfg.seed)
                cfg.checkpoint_dir = os.path.join("data",
                                                  "{}".format(cfg.wandb_project_name),
                                                  "{}".format(cfg.wandb_group_name),
                                                  "{}".format(cfg.wandb_run_name))
                cfg.env_name = env_name
                self.configs.append(cfg)

            # IQN + Double Q Learning
            for i in range(self.num_seeds):
                cfg = Config(**self.base_config.__dict__)
                cfg.use_target_network = True
                cfg.use_double_q_learning = True

                cfg.seed = cfg.seed + i
                cfg.wandb_project_name = self.wandb_project
                cfg.wandb_group_name = "IQN + Double Q Learning {}".format(env_name)
                cfg.wandb_run_name = "Seed {}".format(cfg.seed)
                cfg.checkpoint_dir = os.path.join("data",
                                                  "{}".format(cfg.wandb_project_name),
                                                  "{}".format(cfg.wandb_group_name),
                                                  "{}".format(cfg.wandb_run_name))
                cfg.env_name = env_name
                self.configs.append(cfg)

            # IDS
            for i in range(self.num_seeds):
                cfg = Config(**self.base_config.__dict__)
                cfg.use_ids = True

                # This says 2 because it builds both the input and output linear layers, separated by a ReLU, so it is
                # actually building each Q head with one hidden layer connected to the output layer.
                cfg.ids_n_q_head_model_layers = 2
                cfg.ids_n_q_heads = 10
                cfg.ids_ensemble_variation_coef = 0
                cfg.ids_q_head_feature_dim = 256

                cfg.seed = cfg.seed + i
                cfg.wandb_project_name = self.wandb_project
                cfg.wandb_group_name = "IDS {}".format(env_name)
                cfg.wandb_run_name = "Seed {}".format(cfg.seed)
                cfg.checkpoint_dir = os.path.join("data",
                                                  "{}".format(cfg.wandb_project_name),
                                                  "{}".format(cfg.wandb_group_name),
                                                  "{}".format(cfg.wandb_run_name))
                cfg.env_name = env_name
                self.configs.append(cfg)

            # IDS + Ensemble Variation
            for i in range(self.num_seeds):
                cfg = Config(**self.base_config.__dict__)
                cfg.use_dqn = False
                cfg.use_iqn = True
                cfg.use_ids = True
                cfg.ids_n_q_head_model_layers = 2
                cfg.ids_n_q_heads = 10
                cfg.ids_q_head_feature_dim = 256
                cfg.ids_ensemble_variation_coef = 1e-6

                cfg.seed = cfg.seed + i
                cfg.wandb_project_name = self.wandb_project
                cfg.wandb_group_name = "IDS + Variation {}".format(env_name)
                cfg.wandb_run_name = "Seed {}".format(cfg.seed)
                cfg.checkpoint_dir = os.path.join("data",
                                                  "{}".format(cfg.wandb_project_name),
                                                  "{}".format(cfg.wandb_group_name),
                                                  "{}".format(cfg.wandb_run_name))
                cfg.env_name = env_name
                self.configs.append(cfg)

    def __str__(self):
        return "Additive Ablation Experiment"
