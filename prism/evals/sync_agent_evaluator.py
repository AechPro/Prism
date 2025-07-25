import torch
import numpy as np
from prism.util.observation_stacker import ObservationStacker
import multiprocessing as mp


class SyncAgentEvaluator(object):
    def __init__(self, agent, env, timesteps_between_evaluations, evaluation_timestep_horizon, frame_stack=1, device="cpu"):
        self.agent = agent
        self.env = env
        self.timesteps_between_evaluations = timesteps_between_evaluations
        self.evaluation_timestep_horizon = evaluation_timestep_horizon
        self.timesteps_since_last_eval = np.inf
        self.frame_stack = frame_stack
        self.obs_stacker = ObservationStacker(frame_stack=frame_stack, device=device)

    def maybe_evaluate_agent(self, ts_this_iter, cumulative_timesteps):
        self.timesteps_since_last_eval += ts_this_iter
        if self.timesteps_since_last_eval >= self.timesteps_between_evaluations:
            self.timesteps_since_last_eval = 0
            return self.evaluate_agent()
        return None

    @torch.no_grad()
    def evaluate_agent(self, verbose=False):
        agent = self.agent
        env = self.env
        device = agent.model.device
        n_ts = self.evaluation_timestep_horizon
        obs_stacker = self.obs_stacker
        collected_ts = 0

        obs, info = env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        obs_stacker.reset(obs)

        ep_rews = []
        current_ep_rew = 0
        agent.eval()

        # Collect at least one episode.
        while collected_ts < n_ts or len(ep_rews) == 0:
            action = agent.forward(obs_stacker.obs_stack)
            obs, rew, done, trunc, info = env.step(action)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            obs_stacker.stack(obs)

            current_ep_rew += rew[0]
            if done or trunc:
                ep_rews.append(current_ep_rew)
                if verbose:
                    print(len(ep_rews), current_ep_rew)

                current_ep_rew = 0
                obs, info = env.reset()
                obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
                obs_stacker.reset(obs)

            collected_ts += 1

        agent.train()
        return ep_rews

    def close(self):
        pass
