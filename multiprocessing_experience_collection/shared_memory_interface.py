import numpy as np


class SharedMemoryInterface(object):
    PROC_CRASHED_FLAG = 17713.0

    def __init__(self, memory_buffer, n_flags):
        self.local_buffer_slice = memory_buffer
        self._flags = self.local_buffer_slice[:n_flags]
        self._data = self.local_buffer_slice[n_flags:]
        self._static_obs_arr = None
        self._static_next_obs_arr = None

    def get_flag(self, flag_idx):
        return self._flags[flag_idx]

    def get_data(self, data_range):
        return self._data[data_range[0]:data_range[1]]

    def set_flag(self, flag_idx, flag_value):
        self._flags[flag_idx] = flag_value

    def set_data(self, data_range, data):
        self._data[data_range[0]:data_range[1]] = data[:]


class EnvProcessMemoryInterface(SharedMemoryInterface):
    # Flag definitions.
    PROC_STATUS_WAITING_FOR_ACTION_FLAG = 11389.0
    PROC_STATUS_WROTE_ENV_INFO_FLAG = 41517.0
    PROC_STATUS_WROTE_RESET_FLAG = 51519.0
    PROC_STATUS_STEPPING_ENV_FLAG = 98314.0
    PROC_STATUS_NULL_FLAG = -1.0

    PROC_ORDER_HALT_FLAG = 92924.0
    PROC_ORDER_GET_ENV_INFO_FLAG = 14141.0
    PROC_ORDER_START_FLAG = 99198.0
    PROC_ORDER_TAKE_STEP_FLAG = 17171.0
    PROC_ORDER_WAIT_FLAG = 19191.0

    PROC_ORDER_NULL_FLAG = -1.0

    PROC_STATUS_IDX = 0
    PROC_ORDER_IDX = 1
    CRASHED_FLAG_IDX = 2

    def __init__(self, memory_buffer):
        super().__init__(memory_buffer, 3)
        self.one_hot = False

    def send_actions(self):
        self.set_flag(EnvProcessMemoryInterface.PROC_STATUS_IDX, EnvProcessMemoryInterface.PROC_STATUS_NULL_FLAG)
        self.set_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX, EnvProcessMemoryInterface.PROC_ORDER_TAKE_STEP_FLAG)

    def write_action(self, action, offset=0):
        """
        Write an action for an environment to a memory buffer.
        :param offset: Offset into which the action should be written.
        :param action: Actions to write.
        :return:
        """
        if self.one_hot:
            self._write_np_array(action, offset)
        else:
            self._data[offset] = int(action)

    def read_action(self, n_agents):
        """
        Read an action for an environment from a memory buffer.
        :return: Action that was read.
        """

        idx = 0
        actions = []
        for i in range(n_agents):
            if self.one_hot:
                action, idx = self._read_np_array(idx)
            else:
                action = int(self._data[idx])
                idx += 1
            actions.append(action)

        if n_agents == 1:
            return actions[0]

        return np.asarray(actions)

    def write_step_data(self, obs, rew, done, truncated, next_obs, n_current_agents, n_next_agents):
        """
        Write step data to a memory buffer.

        :param obs: Environment observation.
        :param rew: Reward for action taken at this observation.
        :param done: Whether this timestep is terminal.
        :param truncated: Whether the trajectory has been reset without a terminal state.
        :param next_obs: The next observation from the environment.
        :param n_current_agents: The number of agents in the current observation.
        :param n_next_agents: The number of agents in the next observation.
        :return:
        """
        data = self._data
        idx = 0

        if n_current_agents == 1:
            obs = obs[np.newaxis, ...]
            rew = [rew]

        if n_next_agents == 1:
            next_obs = next_obs[np.newaxis, ...]

        data[idx] = n_current_agents
        idx += 1

        data[idx] = n_next_agents
        idx += 1

        idx = self._write_np_array(obs, idx)

        data[idx:idx+n_current_agents] = rew
        idx += n_current_agents

        data[idx] = float(done)
        idx += 1

        data[idx] = float(truncated)
        idx += 1

        self._write_np_array(next_obs, idx)
        self.set_flag(EnvProcessMemoryInterface.PROC_STATUS_IDX, EnvProcessMemoryInterface.PROC_STATUS_WAITING_FOR_ACTION_FLAG)

    def read_step_data(self):
        data = self._data

        n_current_agents = int(data[0])
        n_next_agents = int(data[1])
        idx = 2
        obs, idx = self._read_np_array(idx, copy=False)
        rews = data[idx:idx + n_current_agents]
        idx += n_current_agents
        done = data[idx]
        idx += 1
        truncated = data[idx]
        idx += 1
        next_obs, _ = self._read_np_array(idx, copy=False)

        return obs, [float(rew) for rew in rews], bool(done), bool(truncated), next_obs, n_current_agents, n_next_agents

    def write_env_info(self, obs_shape, n_actions, n_agents):
        data = self._data
        data[0] = len(obs_shape)
        data[1:1+len(obs_shape)] = obs_shape
        data[1+len(obs_shape)] = n_actions
        data[2+len(obs_shape)] = n_agents
        self.set_flag(EnvProcessMemoryInterface.PROC_STATUS_IDX, EnvProcessMemoryInterface.PROC_STATUS_WROTE_ENV_INFO_FLAG)

    def read_env_info(self):
        data = self._data
        len_obs_shape = int(data[0])
        obs_shape = data[1:1+len_obs_shape]
        n_actions = data[1+len_obs_shape]
        n_agents = data[2+len_obs_shape]
        self.set_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX, EnvProcessMemoryInterface.PROC_ORDER_TAKE_STEP_FLAG)
        return obs_shape, n_actions, n_agents

    def write_reset_obs(self, initial_observation, n_agents):
        if n_agents == 1:
            initial_observation = initial_observation[np.newaxis, ...]

        idx = self._write_np_array(initial_observation, 0)
        self._data[idx] = n_agents

        self.set_flag(EnvProcessMemoryInterface.PROC_STATUS_IDX, EnvProcessMemoryInterface.PROC_STATUS_WROTE_RESET_FLAG)

    def read_reset_obs(self):
        obs, idx = self._read_np_array(0)
        n_agents = int(self._data[idx])
        return obs, n_agents

    def _write_np_array(self, array, idx):
        data = self._data

        n_elements = int(array.size)
        shape = array.shape
        len_shape = len(shape)
        data[idx] = len_shape
        idx += 1
        data[idx:idx + len_shape] = shape
        idx += len_shape
        data[idx:idx + n_elements] = array.flatten()
        return idx + n_elements

    def _read_np_array(self, idx, copy=True):
        data = self._data

        len_shape = int(data[idx])
        idx += 1
        shape = []

        n_elements = 1
        for arg in data[idx:idx + len_shape]:
            shape.append(int(arg))
            n_elements *= arg
        n_elements = int(n_elements)

        idx += len_shape

        array = data[idx:idx + n_elements].reshape(shape)
        if copy:
            array = array.copy()
        idx += n_elements
        return array, idx
