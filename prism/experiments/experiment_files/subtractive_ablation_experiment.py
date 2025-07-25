from prism.experiments import Experiment
from prism.config import SUBTRACTIVE_ABLATION_BASE_CONFIG, Config
import os


class SubtractiveAblationExperiment(Experiment):
    def __init__(self, num_seeds=5):
        self.wandb_project = "Prism Subtractive Ablation Experiment"
        super().__init__(base_config=SUBTRACTIVE_ABLATION_BASE_CONFIG, num_seeds=num_seeds)

        # idx = 0
        # while idx < len(self.configs):
        #     cfg = self.configs[idx]
        #
        #     if cfg.wandb_group_name == "Prism MinAtar/Freeway-v1":
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
            # Prism
            for i in range(self.num_seeds):
                # Copy base config
                cfg = Config(**self.base_config.__dict__)

                cfg.use_layer_norm = True
                cfg.use_per = False
                cfg.use_target_network = True
                cfg.target_update_period = 4_000

                # Modify base config
                cfg.seed = cfg.seed + i
                cfg.wandb_project_name = self.wandb_project
                cfg.wandb_group_name = "Prism - PER + target {}".format(env_name)
                cfg.wandb_run_name = "Seed {}".format(cfg.seed)
                cfg.checkpoint_dir = os.path.join("data",
                                                  "{}".format(cfg.wandb_project_name),
                                                  "{}".format(cfg.wandb_group_name),
                                                  "{}".format(cfg.wandb_run_name))
                cfg.env_name = env_name
                self.configs.append(cfg)

            # Prism - n-step
            # for i in range(self.num_seeds):
            #     cfg = Config(**self.base_config.__dict__)
            #
            #     cfg.n_step_returns_length = 1
            #
            #     cfg.seed = cfg.seed + i
            #     cfg.wandb_project_name = self.wandb_project
            #     cfg.wandb_group_name = "Prism - n-step {}".format(env_name)
            #     cfg.wandb_run_name = "Seed {}".format(cfg.seed)
            #     cfg.checkpoint_dir = os.path.join("data",
            #                                       "{}".format(cfg.wandb_project_name),
            #                                       "{}".format(cfg.wandb_group_name),
            #                                       "{}".format(cfg.wandb_run_name))
            #     cfg.env_name = env_name
            #     self.configs.append(cfg)
            #
            # # Prism - PER
            # for i in range(self.num_seeds):
            #     cfg = Config(**self.base_config.__dict__)
            #
            #     cfg.use_per = False
            #
            #     cfg.seed = cfg.seed + i
            #     cfg.wandb_project_name = self.wandb_project
            #     cfg.wandb_group_name = "Prism - PER {}".format(env_name)
            #     cfg.wandb_run_name = "Seed {}".format(cfg.seed)
            #     cfg.checkpoint_dir = os.path.join("data",
            #                                       "{}".format(cfg.wandb_project_name),
            #                                       "{}".format(cfg.wandb_group_name),
            #                                       "{}".format(cfg.wandb_run_name))
            #     cfg.env_name = env_name
            #     self.configs.append(cfg)
            #
            # # Prism - LayerNorm + target model
            # for i in range(self.num_seeds):
            #     cfg = Config(**self.base_config.__dict__)
            #
            #     cfg.use_layer_norm = False
            #     cfg.use_target_network = True
            #     cfg.target_update_period = 4_000
            #
            #     cfg.seed = cfg.seed + i
            #     cfg.wandb_project_name = self.wandb_project
            #     cfg.wandb_group_name = "Prism - LayerNorm + target {}".format(env_name)
            #     cfg.wandb_run_name = "Seed {}".format(cfg.seed)
            #     cfg.checkpoint_dir = os.path.join("data",
            #                                       "{}".format(cfg.wandb_project_name),
            #                                       "{}".format(cfg.wandb_group_name),
            #                                       "{}".format(cfg.wandb_run_name))
            #     cfg.env_name = env_name
            #     self.configs.append(cfg)
            #
            # # Prism - Ensemble Variation
            # for i in range(self.num_seeds):
            #     cfg = Config(**self.base_config.__dict__)
            #
            #     cfg.ids_ensemble_variation_coef = 0
            #
            #     cfg.seed = cfg.seed + i
            #     cfg.wandb_project_name = self.wandb_project
            #     cfg.wandb_group_name = "Prism - Ensemble Variation {}".format(env_name)
            #     cfg.wandb_run_name = "Seed {}".format(cfg.seed)
            #     cfg.checkpoint_dir = os.path.join("data",
            #                                       "{}".format(cfg.wandb_project_name),
            #                                       "{}".format(cfg.wandb_group_name),
            #                                       "{}".format(cfg.wandb_run_name))
            #     cfg.env_name = env_name
            #     self.configs.append(cfg)
            #
            # # Prism - IDS
            # for i in range(self.num_seeds):
            #     cfg = Config(**self.base_config.__dict__)
            #
            #     cfg.use_ids = False
            #     cfg.use_e_greedy = True
            #     cfg.e_greedy_initial_epsilon = 1.0
            #     cfg.e_greedy_decay_timesteps = 250_000
            #     cfg.e_greedy_final_epsilon = 0.01
            #
            #     cfg.seed = cfg.seed + i
            #     cfg.wandb_project_name = self.wandb_project
            #     cfg.wandb_group_name = "Prism - IDS {}".format(env_name)
            #     cfg.wandb_run_name = "Seed {}".format(cfg.seed)
            #     cfg.checkpoint_dir = os.path.join("data",
            #                                       "{}".format(cfg.wandb_project_name),
            #                                       "{}".format(cfg.wandb_group_name),
            #                                       "{}".format(cfg.wandb_run_name))
            #     cfg.env_name = env_name
            #     self.configs.append(cfg)

    def __str__(self):
        return "Subtractive Ablation Experiment"
