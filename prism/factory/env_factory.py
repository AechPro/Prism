from prism.config import Config


def build_environment(config: Config):
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
        from minatar.gym import register_envs
        register_envs()
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

    elif "shapes_env" in config.env_name:
        from custom_environments.shapes import ShapesEnvironment
        env = ShapesEnvironment.random_game(width=7, height=9, num_colors=4, seed=config.seed)
        env.render_this_env = config.render

    else:
        env = gymnasium.make(config.env_name, render_mode=render_mode)

    env.reset(seed=config.seed)
    return env
