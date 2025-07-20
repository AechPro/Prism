from prism.config import Config
from prism.util import Logger
from torchrl.data import PrioritizedReplayBuffer, LazyMemmapStorage, ReplayBuffer
from torchrl.data import LazyTensorStorage, SliceSampler, PrioritizedSliceSampler, ListStorage
from tensordict import TensorDict
import torch
from torchrl.envs import SerialEnv, GymEnv, CatFrames, SqueezeTransform, UnsqueezeTransform, Compose, StepCounter


def build_exp_buffer(config: Config):
    if config.run_through_redis:
        if config.redis_side == "server":
            from prism.async_components.async_experience_buffer import AsyncExperienceBufferInterface
            return AsyncExperienceBufferInterface(config.redis_host, config.redis_port, config.device)

        elif config.redis_side == "client":
            from prism.async_components.async_experience_buffer import AsyncExperienceBuffer
            return AsyncExperienceBuffer(config.redis_host, config.redis_port)

    from prism.experience import TimestepBuffer

    if config.use_per:
        td_buffer = PrioritizedReplayBuffer(
            storage=ListStorage(max_size=config.experience_replay_capacity),
            collate_fn=lambda x: x,
            alpha=config.per_alpha,
            beta=config.per_beta_start,
            batch_size=config.batch_size)
    else:
        td_buffer = ReplayBuffer(
            storage=ListStorage(max_size=config.experience_replay_capacity),
            collate_fn=lambda x: x,
            batch_size=config.batch_size)

    buffer = TimestepBuffer(td_buffer, frame_stack=config.frame_stack_size, device=config.device,
                            n_step=config.n_step_returns_length, gamma=config.gamma)
    return buffer