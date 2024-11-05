

def eval_ablation_experiment():
    import os
    import yaml
    import wandb
    from prism.evals import CheckpointAgentEvaluator
    from prism.config import ADDITIVE_ABLATION_BASE_CONFIG

    project_name = "Second Prism Additive Ablation Experiment"
    base_path = os.path.join("data", project_name)
    wandb_folder_path = "wandb"

    for ablation_name in os.listdir(base_path):
        ablation_path = os.path.join(base_path, ablation_name)

        for env_name in os.listdir(ablation_path):
            env_path = os.path.join(ablation_path, env_name)

            for seed_name in os.listdir(env_path):
                subdir1 = os.listdir(os.path.join(env_path, seed_name))[0]
                subdir2 = os.listdir(os.path.join(env_path, seed_name, subdir1))[0]

                checkpoints_path = os.path.join(env_path, seed_name, subdir1, subdir2)
                wandb_run = None
                config_object = None
                print(checkpoints_path)

                for wandb_run_folder in os.listdir(wandb_folder_path):
                    if "run" not in wandb_run_folder:
                        continue

                    cfg_path = os.path.join(wandb_folder_path, wandb_run_folder, "files", "config.yaml")
                    with (open(cfg_path) as f):
                        run_config = yaml.safe_load(f)
                        # if "oxv4r5ca" in wandb_run_folder:
                        #     print(run_config)
                        #     print(run_config["wandb_project_name"]["value"], project_name)
                        #     print(run_config["wandb_group_name"]["value"], ablation_name)
                        #     print(run_config["wandb_run_name"]["value"], seed_name)
                        if "wandb_project_name" not in run_config.keys():
                            continue

                        if run_config["wandb_project_name"]["value"] == project_name and \
                                run_config["wandb_group_name"]["value"] == "{}/{}".format(ablation_name, env_name)\
                                and run_config["wandb_run_name"]["value"] == seed_name:
                            print("found wandb run", wandb_run_folder)

                            run_id_str = wandb_run_folder[wandb_run_folder.rfind("-") + 1:]
                            wandb_run = wandb.init(project=project_name, id=run_id_str, resume="must")

                            config_object = ADDITIVE_ABLATION_BASE_CONFIG
                            for attr, yaml_dict in run_config.items():
                                if hasattr(config_object, attr):
                                    setattr(config_object, attr, yaml_dict["value"])
                            break

                if wandb_run is None:
                    print("FAILED TO RECOVER WANDB RUN FOR", project_name, ablation_name, env_name, seed_name)
                    continue

                evaluator = CheckpointAgentEvaluator(config=config_object,
                                                     checkpoint_dir=checkpoints_path,
                                                     wandb_run=wandb_run)
                evaluator.run_evals(break_after_all_checkpoints=True)


def run_learner():
    from prism.config import (DEFAULT_CONFIG,
                              LUNAR_LANDER_CFG,
                              MINATAR_CONFIG,
                              SIMPLE_TRAP_CONFIG,
                              REVISITING_RAINBOW_MINATAR_CONFIG,
                              ROCKET_LEAGUE_CONFIG)
    from prism import Learner

    config = ROCKET_LEAGUE_CONFIG
    config.log_to_wandb = True

    learner = Learner()
    learner.configure(config)
    learner.learn()


def run_experiments():
    from prism.experiments import ExperimentRunner
    from prism.experiments.experiment_files import AdditiveAblationExperiment, SubtractiveAblationExperiment
    runner = ExperimentRunner()
    # runner.register_experiment(AdditiveAblationExperiment(num_seeds=5))
    runner.register_experiment(SubtractiveAblationExperiment(num_seeds=5))
    runner.run_experiments()


def run_evaluator():
    from prism.evals import CheckpointAgentEvaluator
    from prism.config import MINATAR_CONFIG, SIMPLE_TRAP_CONFIG, DEFAULT_CONFIG, ROCKET_LEAGUE_CONFIG

    cfg = ROCKET_LEAGUE_CONFIG
    cfg.render = True
    cfg.device = "cpu"
    checkpoint_dir = "data/checkpoints/{}".format(cfg.env_name)

    evaluator = CheckpointAgentEvaluator(config=cfg, checkpoint_dir=checkpoint_dir)
    evaluator.run_evals()


def main():
    import os
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # run_evaluator()
    run_learner()
    # run_experiments()
    # eval_ablation_experiment()


if __name__ == "__main__":
    main()
