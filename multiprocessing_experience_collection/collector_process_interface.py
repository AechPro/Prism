from multiprocessing_experience_collection import EnvProcessMemoryInterface
import weakref
import time


class CollectorProcessInterface(object):
    def __init__(self, process, process_memory_interface, obs_stack_size, reward_clipping_type=None, device="cpu"):
        self._process = process
        self._process_memory_interface = process_memory_interface
        self._obs_stack_size = obs_stack_size
        self._current_timesteps = None
        self._obs_stacker = None
        self._torch = None
        self._reward_clipping_type = reward_clipping_type
        self.loggable_timers = {}
        # self.reset_loggable_timers()
        self.device = device
        self.current_n_agents = 1
        self.current_action_index = 0

    # def reset_loggable_timers(self):
    #     self.loggable_timers.clear()
    #     # self.loggable_timers["Crash Check Timer"] = []
    #     self.loggable_timers["Env Step Memory Read Time"] = []
    #     # self.loggable_timers["Obs Stack Time"] = []
    #     # self.loggable_timers["Reward Processing Time"] = []
    #     # self.loggable_timers["Timestep Creation Time"] = []

    def start_waiting(self):
        self._process_memory_interface.set_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX,
                                                EnvProcessMemoryInterface.PROC_ORDER_WAIT_FLAG)

    def stop_waiting(self):
        self._process_memory_interface.set_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX,
                                                EnvProcessMemoryInterface.PROC_ORDER_TAKE_STEP_FLAG)

    def signal_start(self):
        self._process_memory_interface.set_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX,
                                                EnvProcessMemoryInterface.PROC_ORDER_START_FLAG)

    def wait_for_reset_obs(self, new_timestep_fn):
        while self._process_memory_interface.get_flag(
                EnvProcessMemoryInterface.PROC_STATUS_IDX) != EnvProcessMemoryInterface.PROC_STATUS_WROTE_RESET_FLAG:
            time.sleep(0.01)

        obs, n_agents = self._process_memory_interface.read_reset_obs()
        obs = self._torch.as_tensor(obs, dtype=self._torch.float32, device=self.device)

        self._current_timesteps = []
        for i in range(n_agents):
            timestep = new_timestep_fn()
            timestep.obs = obs[i]
            self._current_timesteps.append(timestep)

        self._obs_stacker.reset(obs, n_agents)
        self.current_n_agents = n_agents

        return self._obs_stacker.obs_stack

    def get_env_info(self):
        self._process_memory_interface.set_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX,
                                                EnvProcessMemoryInterface.PROC_ORDER_GET_ENV_INFO_FLAG)

        # Spin until we get the env info.
        while self._process_memory_interface.get_flag(
                EnvProcessMemoryInterface.PROC_STATUS_IDX) != EnvProcessMemoryInterface.PROC_STATUS_WROTE_ENV_INFO_FLAG:
            time.sleep(0.01)

        obs_shape, n_actions, n_agents = self._process_memory_interface.read_env_info()
        self.current_n_agents = n_agents
        return obs_shape, n_actions, n_agents

    def init_torch(self):
        import torch
        from prism.util.observation_stacker import ObservationStacker
        self._torch = torch
        self._obs_stacker = ObservationStacker(frame_stack=self._obs_stack_size, device=self.device)

    def receive_env_step(self, create_timestep_fn):
        has_all_actions = True
        for timestep in self._current_timesteps:
            if timestep.action is None:
                has_all_actions = False
                break

        if not has_all_actions:
            return None

        if self._process_memory_interface.get_flag(
                EnvProcessMemoryInterface.CRASHED_FLAG_IDX) == EnvProcessMemoryInterface.PROC_CRASHED_FLAG:
            return EnvProcessMemoryInterface.PROC_CRASHED_FLAG

        current_status = self._process_memory_interface.get_flag(EnvProcessMemoryInterface.PROC_STATUS_IDX)
        if current_status == EnvProcessMemoryInterface.PROC_STATUS_WAITING_FOR_ACTION_FLAG:

            # t1 = time.perf_counter()
            obs, rews, done, trunc, next_obs, n_current_agents, n_next_agents = self._process_memory_interface.read_step_data()
            self.current_n_agents = n_current_agents
            # self.loggable_timers["Env Step Memory Read Time"].append(time.perf_counter() - t1)

            # t1 = time.perf_counter()
            next_obs = self._torch.tensor(next_obs, dtype=self._torch.float32, device=self.device)
            if done or trunc:
                self._obs_stacker.reset(next_obs, n_next_agents)
            else:
                self._obs_stacker.stack(next_obs, n_next_agents)
            # self.loggable_timers["Obs Stack Time"].append(time.perf_counter() - t1)

            # t1 = time.perf_counter()
            timesteps_to_return = []
            for i in range(n_next_agents):
                # If there are more agents in the next timestep than in the current timestep.
                if i >= n_current_agents:
                    # Create a new timestep.
                    next_timestep = create_timestep_fn()
                    next_timestep.obs = next_obs[i]

                    # Add the timestep to the list of tracked timesteps.
                    if i >= len(self._current_timesteps):
                        self._current_timesteps.append(next_timestep)
                    else:
                        self._current_timesteps[i] = next_timestep

                # If there are more agents in the current timestep than in the next timestep.
                else:
                    # Select the current timestep.
                    current_timestep = self._current_timesteps[i]

                    # If there is a previous timestep, accumulate the episodic reward.
                    if current_timestep.prev is not None:
                        prev = current_timestep.prev()
                        if prev is not None:
                            current_timestep.episodic_reward = rews[i] + prev.episodic_reward

                    # Reward clipping.
                    if self._reward_clipping_type == "sign":
                        rews[i] = rews[i] / abs(rews[i])
                    elif self._reward_clipping_type == "dopamine_clamp":
                        rews[i] = min(max(rews[i], -1.0), 1.0)

                    # self.loggable_timers["Reward Processing Time"].append(time.perf_counter() - t1)

                    # t1 = time.perf_counter()

                    # Create the next timestep.
                    next_timestep = create_timestep_fn()
                    next_timestep.obs = next_obs[i]

                    # Fill in the current timestep.
                    current_timestep.reward = rews[i]
                    current_timestep.done = done
                    current_timestep.truncated = trunc

                    # If truncated, create a new truncated timestep.
                    if trunc:
                        truncated_timestep = create_timestep_fn()

                        # Here we're using the current observation in the truncated timestep because this will be the
                        # observation we have not yet acted on when the truncation flag was raised.
                        truncated_timestep.obs = self._torch.as_tensor(obs[i],
                                                                       dtype=self._torch.float32,
                                                                       device=self.device)

                        truncated_timestep.prev = weakref.ref(current_timestep)
                        current_timestep.next = truncated_timestep

                    elif not done:
                        next_timestep.prev = weakref.ref(current_timestep)
                        current_timestep.next = weakref.ref(next_timestep)

                    # self.loggable_timers["Timestep Creation Time"].append(time.perf_counter() - t1)
                    self._current_timesteps[i] = next_timestep
                    timesteps_to_return.append(current_timestep)

            return timesteps_to_return, self._obs_stacker.obs_stack

        return None

    def send_action(self, action):
        if (self._process_memory_interface.get_flag(EnvProcessMemoryInterface.CRASHED_FLAG_IDX) ==
                EnvProcessMemoryInterface.PROC_CRASHED_FLAG):
            return EnvProcessMemoryInterface.PROC_CRASHED_FLAG

        self._current_timesteps[self.current_action_index].action = action
        self._process_memory_interface.write_action(action, self.current_action_index)

        self.current_action_index += 1
        if self.current_action_index >= self.current_n_agents:
            self.current_action_index = 0
            self._process_memory_interface.send_actions()

        return None

    def close(self):
        self._process_memory_interface.set_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX,
                                                EnvProcessMemoryInterface.PROC_ORDER_HALT_FLAG)
        self._process.join()
