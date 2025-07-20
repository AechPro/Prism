

def build_algorithm(config):
    from prism.factory import collector_factory
    collector = collector_factory.build_collector(config)
    obs_shape, n_acts, n_agents = collector.get_env_info()
    config.redis_side = "server"

    import torch
    import os
    import numpy as np
    import random
    from prism.factory import agent_factory, exp_buffer_factory
    from prism.util.checkpointer import Checkpointer
    from prism.util import Logger

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    agent = agent_factory.build_agent(config, obs_shape, n_acts)

    agent.model.should_build_forward_cuda_graph = False
    collector.signal_processes_start_collecting(agent)
    agent.model.should_build_forward_cuda_graph = config.use_cuda_graph and "cuda" in config.device

    exp_buffer = exp_buffer_factory.build_exp_buffer(config)

    checkpoint_dir = os.path.join(config.checkpoint_dir, config.env_name)
    checkpointer = Checkpointer(checkpoint_dir, agent, exp_buffer,
                                config.timesteps_between_evaluations,
                                config.hours_per_checkpoint)

    logger = Logger(config, None)

    return agent, collector, exp_buffer, logger, checkpointer
