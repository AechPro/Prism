import torch


class ObservationStacker(object):
    def __init__(self, frame_stack, device="cpu"):
        self.frame_stack = frame_stack
        self.device = device
        self.obs_stack = None

    def stack(self, obs, n_agents=1):
        if self.obs_stack is None:
            self.reset(obs, n_agents)
        else:
            if self.frame_stack == 1:
                self.obs_stack = obs
            else:
                self.obs_stack = torch.roll(self.obs_stack, shifts=-1, dims=1)
                self.obs_stack[:, -1, ...] = obs

    def reset(self, initial_obs, n_agents=1):
        if self.frame_stack == 1:
            self.obs_stack = initial_obs
        else:
            obs_shape = initial_obs.shape
            frame_stack_shape = (n_agents, self.frame_stack, *obs_shape[1:])
            self.obs_stack = torch.zeros(frame_stack_shape, dtype=torch.float32, device=self.device)
            self.obs_stack[:, -1, ...] = initial_obs


def timing_test():
    import gymnasium as gym
    import time

    stacker = ObservationStacker(frame_stack=4)
    env = gym.make("ALE/Defender-v5")
    obs, info = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32)

    stacker.reset(obs)
    n_timing_trials = 100
    n_steps_per_trial = 10000
    for i in range(n_timing_trials):
        t1 = time.perf_counter()

        for j in range(n_steps_per_trial):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            obs = torch.as_tensor(obs, dtype=torch.float32)

            stacker.stack(obs)

            if terminated or truncated:
                obs, info = env.reset()
                obs = torch.as_tensor(obs, dtype=torch.float32)

                stacker.reset(obs)

        t2 = time.perf_counter()
        print(f"Time elapsed: {t2 - t1:.5f}")


if __name__ == "__main__":
    timing_test()