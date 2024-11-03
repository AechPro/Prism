import numpy as np
import time
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
from multiprocessing_experience_collection.environment_process import run_env
from multiprocessing_experience_collection import EnvProcessMemoryInterface, CollectorProcessInterface
from multiprocessing_experience_collection.experience_collector import RandomAgent


class MultiAgentExperienceCollector(object):
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
        self._random_agent = None
        self._training_reward_ema = training_reward_ema
        self._training_reward = None

        self._waiting_observations = []
        self._waiting_timesteps = []
        self._waiting_pids = []
        self._action_pids = []
        self._buffer_idx = 0

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
        initial_pids = []
        for pid, interface in self._running_processes.items():
            # Wait for reset.
            obs = interface.wait_for_reset_obs(self._new_timestep)
            for i in range(interface.current_n_agents):
                initial_obs_group.append(obs[i])
                initial_pids.append(pid)

            # Obs shape should be (n_agents, *obs_shape)
            self._obs_shape = obs.shape[1:]

        initial_obs = self._torch.stack(initial_obs_group, dim=0).to(self._device)

        # Initial obs should now be (n_procs, n_agents, *obs_shape), so the actions will be (n_procs, n_agents, 1),
        actions = agent.forward(initial_obs)
        for i in range(len(initial_pids)):
            error_code = self._running_processes[initial_pids[i]].send_action(actions[i])
            if error_code == EnvProcessMemoryInterface.PROC_CRASHED_FLAG:
                print("PROC {} CRASHED".format(initial_pids[i]))
                self._running_processes[initial_pids[i]].close()
                del self._running_processes[initial_pids[i]]

        self._obs_tensor = self._torch.zeros((self._inference_buffer_size, *self._obs_shape),
                                             dtype=self._torch.float32, device=self._device)

    def collect_timesteps(self, n_timesteps, agent, exp_buffer, random=False):
        if random:
            agent = self._random_agent

        n_collected = 0
        waiting_observations = self._waiting_observations
        waiting_timesteps = self._waiting_timesteps
        waiting_pids = self._waiting_pids
        action_pids = self._action_pids
        buffer_idx = self._buffer_idx

        while n_collected < n_timesteps:
            for pid, interface in self._running_processes.items():
                data = interface.receive_env_step(self._new_timestep)
                if data == EnvProcessMemoryInterface.PROC_CRASHED_FLAG:
                    print("PROC {} CRASHED".format(pid))
                    interface.close()
                    del self._running_processes[pid]

                elif type(data) is tuple:
                    timesteps, obs_stack = data
                    for i in range(len(timesteps)):
                        waiting_timesteps.append(timesteps[i])
                        waiting_observations.append(obs_stack[i])
                        waiting_pids.append(pid)

            for _ in range(len(waiting_timesteps)):
                timestep = waiting_timesteps.pop(0)
                obs = waiting_observations.pop(0)
                pid = waiting_pids.pop(0)

                if timestep.done or timestep.truncated:
                    if self._training_reward is None:
                        self._training_reward = timestep.episodic_reward
                    else:
                        self._training_reward = self._training_reward_ema * self._training_reward + \
                                                (1 - self._training_reward_ema) * timestep.episodic_reward

                if exp_buffer is not None:
                    exp_buffer.extend(timestep)

                action_pids.append(pid)
                self._obs_tensor[buffer_idx].copy_(obs, non_blocking=True)
                buffer_idx += 1

                if buffer_idx >= self._inference_buffer_size:
                    actions = agent.forward(self._obs_tensor).cpu().numpy()

                    for j in range(len(action_pids)):
                        pid = action_pids[j]
                        if self._running_processes[pid].send_action(
                                actions[j]) == EnvProcessMemoryInterface.PROC_CRASHED_FLAG:
                            print("PROC {} CRASHED".format(pid))
                            self._running_processes[pid].close()
                            del self._running_processes[pid]

                    buffer_idx = 0
                    action_pids = []

                n_collected += 1
                if n_collected >= n_timesteps:
                    break

        self._waiting_observations = waiting_observations
        self._waiting_timesteps = waiting_timesteps
        self._waiting_pids = waiting_pids
        self._action_pids = action_pids
        self._buffer_idx = buffer_idx

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
    collector = MultiAgentExperienceCollector(num_procs=12,
                                              num_floats_per_process=100_000,
                                              inference_buffer_size=6,
                                              obs_stack_size=4,
                                              device="cpu",
                                              config=LUNAR_LANDER_CFG)
    try:
        print("Retrieving env info...")
        obs_shape, n_acts, n_agents = collector.get_env_info()

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
    collector = MultiAgentExperienceCollector(
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
