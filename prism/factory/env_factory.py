from prism.config import Config


class DebugEnv(object):
    def __init__(self):
        from gymnasium.spaces import Box, Discrete
        import numpy as np
        self.np = np
        self.num_steps = 0
        self.n_obs = 3
        self.n_acts = 3
        self.ts_lim = 100
        self.observation_space = Box(low=0, high=self.ts_lim, shape=(self.n_obs,), dtype=np.int32)
        self.action_space = Discrete(n=self.n_acts)

    def step(self, action):
        self.num_steps += 1
        done = self.num_steps >= self.ts_lim
        obs = self.np.ones(self.n_obs, dtype=self.np.float32) * self.num_steps
        rew = self.num_steps
        # print("debug environment returning", rew, "|", obs, "|", self.num_steps)
        return obs, rew, done, False, {}

    def reset(self, seed=None):
        self.num_steps = 0
        obs = self.np.ones(self.n_obs, dtype=self.np.float32) * self.num_steps
        return obs, {}

    def close(self):
        pass


def build_environment(config: Config):
    if "debug" in config.env_name:
        return DebugEnv()

    import gymnasium
    render_mode = "human" if config.render else None
    if "ALE" in config.env_name:
        from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
        gym_env = gymnasium.make(config.env_name,
                                 max_episode_steps=config.episode_timestep_limit,
                                 repeat_action_probability=config.atari_sticky_actions_prob,
                                 frameskip=1,
                                 render_mode=render_mode)
        gym_env = AtariPreprocessing(gym_env, noop_max=config.atari_noops,
                                     terminal_on_life_loss=True,
                                     frame_skip=config.frame_stack_size,
                                     grayscale_obs=True,
                                     scale_obs=True)

        env = gym_env

    elif "MinAtar" in config.env_name:
        env = gymnasium.make(config.env_name,
                             sticky_action_prob=config.atari_sticky_actions_prob,
                             difficulty_ramping=True,
                             render_mode=render_mode)

    elif "simple_trap" in config.env_name:
        from custom_environments.simple_trap_env.simple_trap_environment import SimpleTrapEnvironment
        env = SimpleTrapEnvironment(render_this_env=config.render)

    elif "rocket" in config.env_name:
        from custom_environments.rocket_league.rocketsim_env import RocketSimEnv
        env = RocketSimEnv(render_this_env=config.render)

    else:
        env = gymnasium.make(config.env_name, render_mode=render_mode)

    env.reset(seed=config.seed)
    return env

    # from torchrl.envs.transforms import RewardSum, TransformedEnv, CatFrames
    # from torchrl.envs import ParallelEnv, Compose
    #
    # def create_env_fn():
    #     from torchrl.envs import GymEnv, GymWrapper
    #     if "ALE" in config.env_name:
    #         from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
    #         import gymnasium
    #         gym_env = gymnasium.make(config.env_name,
    #                                  max_episode_steps=config.episode_timestep_limit,
    #                                  repeat_action_probability=config.atari_sticky_actions_prob,
    #                                  frameskip=1)
    #
    #         gym_env = AtariPreprocessing(gym_env, noop_max=config.atari_noops,
    #                                      terminal_on_life_loss=True,
    #                                      frame_skip=config.frame_stack_size,
    #                                      grayscale_obs=True,
    #                                      scale_obs=True)
    #
    #         return GymWrapper(gym_env, categorical_action_encoding=True, device=config.env_device)
    #     else:
    #         return GymEnv(config.env_name, device=config.env_device, max_episode_steps=config.episode_timestep_limit)
    #
    # print("Creating env with {} workers".format(config.num_processes))
    # env = ParallelEnv(num_workers=config.num_processes, create_env_fn=create_env_fn, device=config.env_device)
    #
    # # Unsure why this is sometimes necessary. Cannot reproduce outside my specific configuration.
    # # Signs indicate that the process connections are being written to in the middle of a read, so the action received
    # # by one of the processes comes through as junk and an error is thrown. Increasing this timeout causes the env
    # # manager to wait longer for the synchronize event to happen, which prevents any additional writes while the action
    # # data is going through. However, because I can't reproduce in isolation by spamming random actions up the connection,
    # # it's not obvious what is actually going on here. Increasing the timeout does fix the problem, but I'm not sure why.
    # env._timeout = 1000000.0
    #
    # if "ALE" in config.env_name:
    #     transforms = Compose(RewardSum(reset_keys=["terminated"]),)
    #                          # CatFrames(N=config.frame_stack_size, dim=-3,
    #                          #           in_keys=["observation"], out_keys=["obs_stack"]))
    # else:
    #     transforms = RewardSum(reset_keys=["terminated"])
    #
    # env = TransformedEnv(env, transforms)
    # return env


def multiprocessing_memory_overwrite_error_repro():
    from torchrl.envs import ParallelEnv
    import numpy as np
    import torch

    n_workers = 8

    def create_env_fn():
        from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
        import gymnasium
        from torchrl.envs import GymWrapper
        gym_env = gymnasium.make("ALE/Defender-v5",
                                 max_episode_steps=108_000,
                                 repeat_action_probability=0,
                                 frameskip=1)

        gym_env = AtariPreprocessing(gym_env, noop_max=30,
                                     terminal_on_life_loss=True,
                                     frame_skip=4,
                                     grayscale_obs=True,
                                     scale_obs=True)

        return GymWrapper(gym_env, categorical_action_encoding=True, device="cpu")

    print("Making envs...")
    env = ParallelEnv(num_workers=n_workers, create_env_fn=create_env_fn, device="cpu")
    action = torch.as_tensor(np.random.randint(0, 18, n_workers), dtype=torch.long, device="cpu") * 0
    zeros = torch.zeros_like(action, dtype=torch.long, device="cpu")

    td = env.reset()
    print("Starting spam...")
    for i in range(150_000):
        # action += 1
        # action = torch.where(action >= 18, zeros, action)
        # td["action"] = action
        td["action"] = torch.ones(n_workers, dtype=torch.long, device="cpu") * np.random.randint(0, 18)
        td, next_td = env.step_and_maybe_reset(td)
        td = next_td
    print("Done!")


def test():
    from prism.config import DEFAULT_CONFIG
    cfg = DEFAULT_CONFIG
    cfg.num_processes = 1
    cfg.env_name = "ALE/Pong-v5"

    env = build_environment(cfg)
    print(env.rand_step(env.reset()))
    print()
    print(env.observation_spec)
    print(env.action_spec)


if __name__ == "__main__":
    # test()
    multiprocessing_memory_overwrite_error_repro()
