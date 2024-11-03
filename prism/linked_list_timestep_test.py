import torch
from tensordict import TensorDict
from torchrl.envs import GymEnv, ParallelEnv
from torchrl.data import PrioritizedReplayBuffer, LazyTensorStorage, ReplayBuffer, ListStorage
from tensordict import tensorclass
import weakref
import torch
from prism.experience import Timestep, TimestepBuffer
import time


def create_env_fn():
    return GymEnv("LunarLander-v2", device="cpu")


def test():
    n_proc = 2
    batch_size = 32
    env = ParallelEnv(num_workers=n_proc, create_env_fn=create_env_fn, device="cpu")

    torchrl_buffer = PrioritizedReplayBuffer(
        storage=ListStorage(250_000),
        collate_fn=lambda x: x,
        batch_size=batch_size,
        alpha=0.5,
        beta=0.4
    )
    buffer = TimestepBuffer(torchrl_buffer, frame_stack=7, device="cpu", n_step=3, gamma=0.99)

    timestep_id = 0

    proc_ts_map = {}
    td = env.reset()

    for i in range(n_proc):
        timestep = Timestep(timestep_id)
        timestep.obs = td["observation"][i]
        timestep_id += 1
        proc_ts_map[i] = timestep

    initial_timestep = Timestep(-1)
    initial_timestep.next = weakref.ref(proc_ts_map[0])

    observations = []
    for _ in range(32):
        td = env.rand_action(td)
        td, next_td = env.step_and_maybe_reset(td)
        observations.append((td["observation"], td["next"]["observation"]))
        # print(td["observation"], "->", td["next"]["observation"])

        for i in range(n_proc):
            current_ts = proc_ts_map[i]

            next_timestep = Timestep(timestep_id)
            timestep_id += 1

            next_timestep.obs = td["next"]["observation"][i]

            current_ts.action = td["action"][i]
            current_ts.reward = td["next"]["reward"][i].item()
            current_ts.done = td["next"]["terminated"][i].item()
            current_ts.truncated = td["next"]["truncated"][i].item()

            if not current_ts.done:
                next_timestep.prev = weakref.ref(current_ts)
                current_ts.next = weakref.ref(next_timestep)

            proc_ts_map[i] = next_timestep
            t1 = time.perf_counter()
            buffer.extend(current_ts)
            print("extend", time.perf_counter() - t1)

        td = next_td

    data = buffer.sample(batch_size=32)
    x = 0
    # t1 = time.perf_counter()
    # n_iters = 1000
    # for i in range(n_iters):
    #     data = buffer.sample()
    # print((time.perf_counter() - t1) / n_iters)

    # print(data)

    # for ts in data["timestep"]:
    #     while ts is not None:
    #         print(ts)
    #         if ts.next is not None:
    #             ts = ts.next()  # callable because this is a weakref.
    #         else:
    #             break
    #
    #     print("Next buffer entry")


if __name__ == "__main__":
    test()



