import numpy as np
import time
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray

from numpy.ma.core import shape

from multiprocessing_experience_collection.environment_process import run_env
from multiprocessing_experience_collection import EnvProcessMemoryInterface, CollectorProcessInterface


class RandomAgent(object):
    def __init__(self, n_acts):
        self.n_acts = n_acts
        self.rng = np.random.RandomState(123)
        import torch
        self.torch = torch

    def forward(self, obs):
        return self.torch.as_tensor(self.rng.randint(0, self.n_acts, obs.shape[0]), dtype=self.torch.long)


class MultiprocessExperienceCollector(object):
    def __init__(self, num_procs, num_floats_per_process, config, inference_buffer_size, obs_stack_size,
                 device="cpu", training_reward_ema=0.9):

        self._device = device
        self._timestep_id = 0
        self._inference_buffer_size = inference_buffer_size
        self._running_processes = {}
        self._timestep_class_callable = None
        self._torch = None
        self._obs_tensor = None
        self._obs_shape = None
        self._waiting_pids = []
        self._current_idx = 0
        self._random_agent = None
        self._training_reward_ema = training_reward_ema
        self._training_reward = None
        self.overflow = []

        # self.loggable_timers = {}
        # self.reset_loggable_timers()

        self._init_processes(num_procs, num_floats_per_process, config, obs_stack_size)

    def get_env_info(self):
        obs_shape, n_acts, n_agents = self._running_processes[0].get_env_info()
        self._random_agent = RandomAgent(n_acts)
        return obs_shape, n_acts, n_agents

    def signal_processes_start_collecting(self, agent):
        # Signal all processes to send the initial observations from the first reset.
        for pid, interface in self._running_processes.items():
            interface.signal_start()

        self._init_torch_imports()

        # Collect all the initial observations.
        initial_obs_group = []
        for pid, interface in self._running_processes.items():
            # Wait for reset.
            obs, n_agents = interface.wait_for_reset_obs(self._new_timestep)
            initial_obs_group.append(obs)

            # Obs shape should be (n_agents, *obs_shape)
            self._obs_shape = obs.shape[1:]

        initial_obs = self._torch.stack(initial_obs_group, dim=0).to(self._device)

        # Initial obs should now be (n_procs, n_agents, *obs_shape), so the actions will be (n_procs, n_agents, 1),
        actions = agent.forward(initial_obs)

        for pid, interface in self._running_processes.items():
            if interface.send_action(actions[pid]) == EnvProcessMemoryInterface.PROC_CRASHED_FLAG:
                print("PROC {} CRASHED".format(pid))
                interface.close()
                del self._running_processes[pid]

        self._obs_tensor = self._torch.zeros((self._inference_buffer_size, *self._obs_shape),
                                             dtype=self._torch.float32, device=self._device)

    def collect_timesteps(self, n_timesteps, agent, exp_buffer, random=False):
        if random:
            agent = self._random_agent

        n_collected = 0
        obs_tensor = self._obs_tensor
        procs = self._running_processes
        new_timestep = self._new_timestep

        pids = self._waiting_pids
        idx = self._current_idx
        overflow = self.overflow

        while len(overflow) > 0 and n_collected < n_timesteps and exp_buffer is not None:
            timestep = overflow.pop(0)
            exp_buffer.extend(timestep)
            n_collected += 1

        with ((self._torch.no_grad())):
            while n_collected < n_timesteps:
                for pid, interface in procs.items():
                    # t1 = time.perf_counter()
                    data = interface.receive_env_step(new_timestep)
                    # self.loggable_timers["Receive Env Step Time"].append(time.perf_counter() - t1)

                    if data == EnvProcessMemoryInterface.PROC_CRASHED_FLAG:
                        print("PROC {} CRASHED".format(pid))
                        interface.close()
                        del procs[pid]
                        break

                    if data is not None:
                        complete_timesteps, obs = data
                        for complete_timestep in complete_timesteps:
                            if complete_timestep.done or complete_timestep.truncated:
                                if self._training_reward is None:
                                    self._training_reward = complete_timestep.episodic_reward
                                else:
                                    self._training_reward = self._training_reward_ema * self._training_reward + \
                                                            (1 - self._training_reward_ema) * complete_timestep.episodic_reward

                            # t1 = time.perf_counter()
                            if n_collected >= n_timesteps:
                                overflow.append(complete_timestep)
                            else:
                                n_collected += 1
                                if exp_buffer is not None:
                                    exp_buffer.extend(complete_timestep)
                            # self.loggable_timers["Buffer Extend Time"].append(time.perf_counter() - t1)

                        pid_index_buffer = []
                        pids.append(pid)
                        for i in range(obs.shape[0]):
                            obs_tensor[idx].copy_(obs[i], non_blocking=True)
                            idx += 1
                            pid_index_buffer.append(pid)

                            if idx >= self._inference_buffer_size:
                                # state_recv_time = time.perf_counter() - state_recv_timer
                                # action_send_timer = time.perf_counter()

                                actions = agent.forward(obs_tensor).cpu().numpy()
                                action_tensor_index = 0
                                for j in range(len(pids)):
                                    proc = procs[pids[j]]
                                    action = actions[action_tensor_index:action_tensor_index + proc.current_n_agents]
                                    action_tensor_index += proc.current_n_agents

                                    if proc.send_action(action) == EnvProcessMemoryInterface.PROC_CRASHED_FLAG:

                                        print("PROC {} CRASHED".format(pids[j]))
                                        procs[pids[j]].close()
                                        del procs[pids[j]]

                                        import sys
                                        sys.exit(-1)

                                idx = 0
                                pids = []

                        # action_send_time = time.perf_counter() - action_send_timer
                        # self.loggable_timers["State Recv Time"].append(state_recv_time)
                        # state_recv_timer = time.perf_counter()
                        # self.loggable_timers["Action Send Time"].append(action_send_time)

        self._waiting_pids = pids
        self._current_idx = idx

        return n_collected

    def close(self):
        for pid, interface in self._running_processes.items():
            interface.close()

    def log(self, logger):
        # for key, value in self.loggable_timers.items():
        #     logger.log_data(data=np.mean(value), group_name="Report/Metrics", var_name=key)
        # self.reset_loggable_timers()

        # for pid, interface in self._running_processes.items():
        #     for key, value in interface.loggable_timers.items():
        #         logger.log_data(data=np.mean(value), group_name="Report/Metrics".format(pid), var_name="Proc{} {}".format(pid, key))
        #     interface.reset_loggable_timers()

        logger.log_data(data=self._training_reward,
                        group_name="Report/Rewards",
                        var_name="Training Reward")

    # def reset_loggable_timers(self):
    #     self.loggable_timers.clear()
    #     self.loggable_timers["State Recv Time"] = []
    #     self.loggable_timers["Action Send Time"] = []
    #     self.loggable_timers["Buffer Extend Time"] = []
    #     self.loggable_timers["Receive Env Step Time"] = []

    def _init_torch_imports(self):
        # We need to do this to avoid importing torch before the processes start.
        import torch
        from prism.experience import Timestep
        self._timestep_class_callable = Timestep
        self._torch = torch

        for pid, interface in self._running_processes.items():
            interface.init_torch()

    def _new_timestep(self):
        self._timestep_id += 1
        return self._timestep_class_callable(self._timestep_id)

    def _init_processes(self, num_procs, num_floats_per_process, config, obs_stack_size):
        shared_memory = RawArray('f', num_floats_per_process * num_procs)

        can_fork = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if can_fork else "spawn"
        context = mp.get_context(start_method)

        for i in range(num_procs):
            memory_offset = num_floats_per_process * i
            pid = i
            process = context.Process(
                target=run_env,
                args=(pid, config, shared_memory, memory_offset, num_floats_per_process),
            )
            process.start()
            local_memory_slice = np.frombuffer(shared_memory, dtype='float32', count=num_floats_per_process,
                                               offset=memory_offset)
            process_memory_interface = EnvProcessMemoryInterface(local_memory_slice)
            collector_process_interface = CollectorProcessInterface(process,
                                                                    process_memory_interface,
                                                                    obs_stack_size)

            self._running_processes[pid] = collector_process_interface


def create_test_env():
    from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
    import gymnasium
    gym_env = gymnasium.make("ALE/Boxing-v5",
                             max_episode_steps=108_000,
                             repeat_action_probability=0,
                             frameskip=1)

    gym_env = AtariPreprocessing(gym_env, noop_max=30,
                                 terminal_on_life_loss=True,
                                 frame_skip=4,
                                 grayscale_obs=True,
                                 scale_obs=True)
    return gym_env

    # import gymnasium
    # return gymnasium.make("LunarLander-v2")


def test():
    import os
    from prism.config import LUNAR_LANDER_CFG
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    collector = MultiprocessExperienceCollector(num_procs=12,
                                                num_floats_per_process=100_000,
                                                inference_buffer_size=6,
                                                obs_stack_size=4,
                                                device="cpu",
                                                config=LUNAR_LANDER_CFG)
    try:
        print("Retrieving env info...")
        obs_shape, n_acts = collector.get_env_info()

        print("Creating fake agent...")
        agent = RandomAgent(n_acts)

        print("Waiting for all envs to send reset states...")
        collector.signal_processes_start_collecting(agent)

        print("Beginning collection....")
        n_ts = 10_000
        sps_measurements = []
        for i in range(100):
            t1 = time.perf_counter()
            collector.collect_timesteps(n_ts, agent)
            collection_time = time.perf_counter() - t1
            sps = n_ts / collection_time
            sps_measurements.append(sps)
            print("Collected {} timesteps in {} seconds | {} | {}".format(n_ts, collection_time, sps,
                                                                          np.mean(sps_measurements)))
            time.sleep(1)

        print("Done collecting!")
        # for ts in timesteps:
        #     print(ts)

    finally:
        collector.close()


def test_interaction():
    import torch

    class TestAgent(object):
        def __init__(self):
            self.step = 0

        def forward(self, obs):
            self.step += 1
            return torch.as_tensor([obs[0][0].item() for i in range(len(obs))], dtype=torch.long)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    from prism.factory import exp_buffer_factory
    from prism.config import LUNAR_LANDER_CFG
    config = LUNAR_LANDER_CFG
    config.env_name = "debug"
    config.num_processes = 4

    exp_buffer = exp_buffer_factory.build_exp_buffer(config)
    inference_buffer_size = 4  # config.num_processes  # int(round(config.num_processes * 0.9))
    collector = MultiprocessExperienceCollector(
        num_procs=config.num_processes,
        num_floats_per_process=config.shared_memory_num_floats_per_process,
        config=config,
        inference_buffer_size=inference_buffer_size,
        obs_stack_size=config.frame_stack_size,
        device=config.device,
        training_reward_ema=config.training_reward_ema
    )

    agent = TestAgent()
    collector.signal_processes_start_collecting(agent)

    time.sleep(1)
    for i in range(100):
        ts = collector.collect_timesteps(1, agent, exp_buffer)
    timesteps = exp_buffer.buffer._storage._storage
    for ts in timesteps:
        assert ts[0].obs[0].item() == ts[0].action, "OBS-ACTION MISMATCH".format(ts[0].obs[0].item(), ts[0].action)
        assert ts[0].reward == ts[0].action + 1, "REWARD-ACTION MISMATCH".format(ts[0].reward, ts[0].action)
        print(ts[0])
    collector.close()


if __name__ == "__main__":
    test_interaction()
