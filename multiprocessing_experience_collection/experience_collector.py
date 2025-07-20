import numpy as np
import time
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
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


class ExperienceCollector(object):
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
        logger.log_data(data=self._training_reward,
                        group_name="Report/Rewards",
                        var_name="Training Reward")

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
