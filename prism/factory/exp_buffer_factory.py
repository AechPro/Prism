from prism.config import Config
from prism.util import Logger
from torchrl.data import PrioritizedReplayBuffer, LazyMemmapStorage, ReplayBuffer
from torchrl.data import LazyTensorStorage, SliceSampler, PrioritizedSliceSampler, ListStorage
from tensordict import TensorDict
import torch
from torchrl.envs import SerialEnv, GymEnv, CatFrames, SqueezeTransform, UnsqueezeTransform, Compose, StepCounter


def build_exp_buffer(config: Config):
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


def catframes_speed_test(with_catframes=True):
    from tensordict import TensorDict
    import time

    if with_catframes:
        obs_shape = (16, 84, 84, 1)
        transform = Compose(
            CatFrames(N=4, dim=-1, in_keys=["observation"], done_key="done", padding="constant"),
            CatFrames(N=4, dim=-1, in_keys=[("next", "observation")], done_key="done", padding="constant"))
    else:
        obs_shape = (16, 84, 84, 4)
        transform = None

    sampler = PrioritizedSliceSampler(
        max_capacity=50_000,
        alpha=0.5,
        beta=0.4,
        strict_length=False,
        num_slices=32 // 4,
        span=[True, False])

    exp_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=50_000, device="cpu"),
        sampler=sampler,
        batch_size=32,
        transform=transform
    )
    fake_data = TensorDict(
        {
            "observation": torch.zeros(obs_shape, dtype=torch.float32),
            "next": {"observation": torch.zeros(obs_shape, dtype=torch.float32),
                     "done": torch.zeros((obs_shape[0], 1), dtype=torch.bool)},
        },
        batch_size=[obs_shape[0]],
    )

    for _ in range(25):
        exp_buffer.extend(fake_data)

    import cProfile
    prof = cProfile.Profile()
    t1 = time.perf_counter()
    prof.enable()
    data, info = exp_buffer.sample(return_info=True)
    prof.disable()
    prof.dump_stats("data/exp_buffer.pstat")
    t2 = time.perf_counter()
    print(f"Sampling took {t2 - t1} seconds.")

    exp_buffer.empty()


def _make_fake_tensordict(obs_shape):
    fake_data = TensorDict(
        {
            "observation": torch.zeros(obs_shape[0], 1, obs_shape[-1], dtype=torch.float32),
            "action": torch.zeros(obs_shape[0], 1, 4, dtype=torch.long),
            "episode_reward": torch.zeros(obs_shape[0], 1, 1, dtype=torch.float32),
            "gamma": torch.zeros(obs_shape[0], 1, dtype=torch.float32),
            "nonterminal": torch.zeros(obs_shape[0], 1, dtype=torch.bool),
            "truncated": torch.zeros(obs_shape[0], 1, 1, dtype=torch.bool),
            "terminated": torch.zeros(obs_shape[0], 1, 1, dtype=torch.bool),
            "done": torch.zeros(obs_shape[0], 1, 1, dtype=torch.bool),
            "steps_to_next_obs": torch.zeros(obs_shape[0], 1, dtype=torch.int64),
            "next": TensorDict(
                {"observation": torch.zeros((obs_shape[0], 1, obs_shape[-1]), dtype=torch.float32),
                 "done": torch.zeros((obs_shape[0], 1, 1), dtype=torch.bool),
                 "truncated": torch.zeros((obs_shape[0], 1, 1), dtype=torch.bool),
                 "terminated": torch.zeros((obs_shape[0], 1, 1), dtype=torch.bool),

                 "episode_reward": torch.zeros((obs_shape[0], 1, 1), dtype=torch.float32),
                 "reward": torch.zeros((obs_shape[0], 1, 1), dtype=torch.float32),
                 "original_reward": torch.zeros((obs_shape[0], 1, 1), dtype=torch.float32),
                 }, batch_size=[obs_shape[0], 1], device="cpu"),
        }, batch_size=[obs_shape[0], 1], device="cpu")
    return fake_data


def sampling_time_test():
    import time

    # Create a wandb run to graph the sampling time.
    import wandb
    wandb_run = wandb.init(project="torchrl-bug-testing", group="debug", name="sampling_time_test", reinit=True)

    # Basic variables.
    ts_per_obs = 4
    batch_size = 32
    buffer_size = 250_000
    obs_shape = (ts_per_obs, 8)

    # Set up a prioritized replay buffer by using a slice sampler.
    exp_buffer_slice_sampler = ReplayBuffer(
        storage=LazyTensorStorage(max_size=buffer_size, device="cpu"),
        sampler=PrioritizedSliceSampler(
            max_capacity=buffer_size,
            alpha=0.5,
            beta=0.4,
            strict_length=False,
            num_slices=batch_size,
            compile=False,
            span=[True, False]),
        batch_size=batch_size,
        transform=None
    )

    # Set up an instance of the native prioritized replay buffer.
    exp_buffer_normal = PrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=buffer_size, device="cpu"),
        alpha=0.5,
        beta=0.4,
        batch_size=batch_size)

    # Initialize counters.
    report_timer = time.perf_counter()
    slice_time = 0
    normal_time = 0
    n_iter = 0
    n_ts = 0

    for i in range(1_000_000):

        # Create fake data.
        fake_data = _make_fake_tensordict(obs_shape).flatten()

        # Count the number of timesteps in the fake batch.
        n_ts += fake_data.batch_size[0]

        # Add the batch to both buffers.
        exp_buffer_slice_sampler.extend(fake_data)
        exp_buffer_normal.extend(fake_data)

        # Measure the time it takes to sample from the buffer using a slice sampler.
        t1 = time.perf_counter()
        batch, info = exp_buffer_slice_sampler.sample(return_info=True)
        slice_time += time.perf_counter() - t1

        # Measure the time it takes to sample from the native prioritized replay buffer.
        t1 = time.perf_counter()
        batch, info = exp_buffer_normal.sample(return_info=True)
        normal_time += time.perf_counter() - t1

        # Count the number of iterations that have passed since our last report, so we can average the counters.
        n_iter += 1
        if time.perf_counter() - report_timer > 1:
            # Log to wandb.
            wandb_run.log({"Prioritized Slice Sampler": slice_time / n_iter,
                           "Prioritized Replay Buffer": normal_time / n_iter,
                           "n_ts": n_ts})

            # Print the results.
            print(
                f"Prioritized Slice Sampler: {slice_time / n_iter}\nPrioritized Replay Buffer: {normal_time / n_iter}\n")

            # Reset the counters.
            slice_time = 0
            normal_time = 0
            n_iter = 0
            report_timer = time.perf_counter()


def test():
    from prism.factory import env_factory
    from torchrl.envs import StepCounter
    from prism.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG
    cfg.num_processes = 1
    cfg.experience_replay_capacity = 10000
    exp_buffer = build_exp_buffer(cfg)
    env = env_factory.build_environment(cfg)
    # env = GymEnv("CartPole-v1", device=None)
    env = env.append_transform(StepCounter())

    out = env.rollout(10000, break_when_any_done=False)
    # ones = torch.ones_like(out["next"]["reward"])
    del out["obs_stack"]
    del out["next"]["obs_stack"]
    # out["obs_stack"] = ones
    # out["next"]["obs_stack"] = ones
    exp_buffer.extend(out.view(-1))

    s, info = exp_buffer.sample(return_info=True)
    # s = s[info[("next", "truncated")].squeeze(-1)]

    print(s)
    print()


if __name__ == "__main__":
    sampling_time_test()
    # print("WITHOUT CATFRAMES")
    # catframes_speed_test(with_catframes=False)
    #
    # print("WITH CATFRAMES")
    # catframes_speed_test(with_catframes=True)
