import os
from prism.evals import SyncAgentEvaluator
import time
import numpy as np


class CheckpointAgentEvaluator(object):
    def __init__(self, config, checkpoint_dir, wandb_run=None):
        self.agent = None
        self.evaluator = None
        self.checkpoint_dir = checkpoint_dir
        self.unprocessed_checkpoints = []
        self.processed_checkpoints = []
        self.wandb_run = wandb_run
        self._init(config)

    def _init(self, config):
        from prism.factory import agent_factory, env_factory

        env = env_factory.build_environment(config)
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n
        self.agent = agent_factory.build_agent(config=config,
                                               obs_shape=obs_shape,
                                               n_actions=n_actions)

        self.evaluator = SyncAgentEvaluator(self.agent, env, 0,
                                            config.evaluation_timestep_horizon,
                                            config.frame_stack_size,
                                            device=config.device)

    def run_evals(self, break_after_all_checkpoints=True):
        while True:
            self.scan_checkpoints()
            if len(self.unprocessed_checkpoints) == 0:
                if break_after_all_checkpoints:
                    break

                time.sleep(1)
                continue

            checkpoint = self.unprocessed_checkpoints.pop(0)
            self.agent.load(checkpoint)

            print("Evaluating checkpoint", checkpoint)
            ep_rews = self.evaluator.evaluate_agent(verbose=False)
            if self.wandb_run is not None:
                cumulative_timesteps = int(checkpoint[checkpoint.rfind("_")+1:])
                self.wandb_run.log({"Report/Rewards-Eval Reward": np.mean(ep_rews),
                                    "Report/Metrics-Cumulative Timesteps": cumulative_timesteps})

            print("EVALUATION OF CHECKPOINT {} COMPLETE.\n"
                  "MEAN REWARD: {}\n"
                  "STD REWARD: {}\n".format(checkpoint, np.mean(ep_rews), np.std(ep_rews)))

            self.processed_checkpoints.append(checkpoint)

        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.evaluator.env.close()

    def scan_checkpoints(self):
        added_any = False
        for folder_name in os.listdir(self.checkpoint_dir):
            if "agent_checkpoint" not in folder_name:
                continue

            checkpoint_path = os.path.join(self.checkpoint_dir, folder_name)
            if checkpoint_path not in self.processed_checkpoints and checkpoint_path not in self.unprocessed_checkpoints:
                files_exist = os.path.exists(os.path.join(checkpoint_path, "agent", "model.pt")) and \
                                os.path.exists(os.path.join(checkpoint_path, "agent", "optimizer.pt")) and \
                                os.path.exists(os.path.join(checkpoint_path, "agent", "state.pkl"))
                if files_exist:
                    print("Adding new checkpoint", checkpoint_path)
                    added_any = True
                    self.unprocessed_checkpoints.append(checkpoint_path)

        if added_any:
            self.unprocessed_checkpoints.sort(key=lambda x: int(x.split("_")[-1]), reverse=True)

