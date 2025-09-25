import time
from prism.async_components.redis import RedisInterface
from prism.experience import Timestep, TimestepBuffer
from prism.factory import exp_buffer_factory
import torch
from tensordict import TensorDict
from collections import deque


class AsyncExperienceBuffer(object):
    def __init__(self, redis_host, redis_port):
        self._redis_interface = RedisInterface(redis_host, redis_port)
        self._batch_size = None
        self._experience_buffer = None
        self._waiting_timestep_id_map = {}
        self._time_between_command_pings = 1.0
        self._time_since_last_command_ping = 0.0
        self._n_collected = 0
        self._last_collect_call_timer = 0
        self._time_between_collect_calls = 0.01
        self._hanging_prev = None

    def wait_for_config(self):
        config = self._redis_interface.get_config()
        print("ASYNC BUFFER GOT CONFIG", config)
        config.run_through_redis = False
        while config is None:
            print("Async buffer waiting for config...")
            time.sleep(0.1)
            config = self._redis_interface.get_config()

        self._batch_size = config.batch_size
        self._experience_buffer = exp_buffer_factory.build_exp_buffer(config)

    def run(self):
        self.wait_for_config()
        running = True
        while running:
            self._get_latest_timesteps()
            if self._n_collected < self._batch_size:
                time.sleep(0.01)
                continue

            batch = self._experience_buffer.sample(batch_size=self._batch_size)
            self._transmit_batch(batch)

            if time.perf_counter() - self._time_since_last_command_ping > self._time_between_command_pings:
                current_command = self._redis_interface.get_current_command()
                running = current_command != RedisInterface.SHUTDOWN_COMMAND
                self._time_since_last_command_ping = time.perf_counter()

    def _get_latest_timesteps(self):
        if time.perf_counter() - self._last_collect_call_timer < self._time_between_collect_calls:
            return

        serialized_timesteps = self._redis_interface.get_timesteps()
        # print("Deserializing timesteps...", len(serialized_timesteps))
        (deserialized_timesteps,
         waiting_timestep_id_map) = Timestep.deserialize_linked_list(serialized_timesteps,
                                                                     self._waiting_timestep_id_map)
        # print("Linked list deserialized", len(deserialized_timesteps))

        self._waiting_timestep_id_map.clear()
        self._waiting_timestep_id_map = waiting_timestep_id_map

        for timestep in deserialized_timesteps:
            self._n_collected += 1
            self._experience_buffer.extend(timestep)

        # print(self._n_collected)
        self._last_collect_call_timer = time.perf_counter()

    def _transmit_batch(self, batch):
        serialized = []
        serialized += self._serialize_tensor(batch["observation"])
        serialized += self._serialize_tensor(batch["next"]["observation"])
        serialized += self._serialize_tensor(batch["next"]["reward"])
        serialized += self._serialize_tensor(batch["nonterminal"])
        serialized += self._serialize_tensor(batch["gamma"])
        serialized += self._serialize_tensor(batch["action"])

        self._redis_interface.add_batch(serialized)


    def _serialize_tensor(self, tensor):
        n_elements = int(tensor.numel())
        shape = tensor.shape
        len_shape = len(shape)
        data = [len_shape, *shape, n_elements, *tensor.flatten().tolist()]

        return data


class AsyncExperienceBufferInterface(object):
    def __init__(self, redis_host, redis_port, device):
        self._redis_interface = RedisInterface(redis_host, redis_port)
        self._batch = None
        self._batch_buffer = []
        self._timestep_buffer = []
        self.device = device
        self._obs = None
        self._next_obs = None
        self._reward = None
        self._nonterminal = None
        self._gamma = None
        self._action = None
        # Keep a small window of strong references to avoid weakref death across flush boundaries
        self._strong_ref_window = deque(maxlen=32)

    def set_static_batch(self, batch):
        self._batch = batch
        self._obs = self._batch["observation"]
        self._next_obs = self._batch["next"]["observation"]
        self._reward = self._batch["next"]["reward"]
        self._nonterminal = self._batch["nonterminal"]
        self._gamma = self._batch["gamma"]
        self._action = self._batch["action"]

    def get_static_batch(self):
        return self._batch

    def extend(self, timestep):
        self._timestep_buffer.append(timestep)
        # Hold strong refs for prev/self/next to keep links alive until after serialization
        self._strong_ref_window.append(timestep)
        if timestep.prev is not None:
            prev_ts = timestep.prev if isinstance(timestep.prev, Timestep) else timestep.prev()
            if prev_ts is not None:
                self._strong_ref_window.append(prev_ts)
        if timestep.next is not None:
            next_ts = timestep.next if isinstance(timestep.next, Timestep) else timestep.next()
            if next_ts is not None:
                self._strong_ref_window.append(next_ts)
                
        flush_threshold = 10
        if len(self._timestep_buffer) >= flush_threshold:
            # print("Transmitting timesteps...", len(self._timestep_buffer))
            self._redis_interface.submit_timesteps(self._timestep_buffer)
            self._timestep_buffer = []

    def sample(self, return_info=False):
        serialized_batch = self._redis_interface.pop_latest_training_batch()
        while serialized_batch is None:
            time.sleep(0.01)
            serialized_batch = self._redis_interface.pop_latest_training_batch()

        batch_tensors = self._deserialize_batch(serialized_batch)

        self._obs.copy_(batch_tensors[0], non_blocking=True)
        self._next_obs.copy_(batch_tensors[1], non_blocking=True)
        self._reward.copy_(batch_tensors[2], non_blocking=True)
        self._nonterminal.copy_(batch_tensors[3], non_blocking=True)
        self._gamma.copy_(batch_tensors[4], non_blocking=True)
        self._action.copy_(batch_tensors[5], non_blocking=True)

        if return_info:
            return self._batch, {}
        return self._batch

    def _deserialize_batch(self, serialized_batch):
        idx = 0
        obs, idx = self._deserialize_tensor(serialized_batch, idx)
        next_obs, idx = self._deserialize_tensor(serialized_batch, idx)
        reward, idx = self._deserialize_tensor(serialized_batch, idx)
        nonterminal, idx = self._deserialize_tensor(serialized_batch, idx)
        gamma, idx = self._deserialize_tensor(serialized_batch, idx)
        action, idx = self._deserialize_tensor(serialized_batch, idx)
        nonterminal = nonterminal.to(dtype=torch.bool)
        action = action.to(dtype=torch.long)

        if self._batch is None:
            obs_buf = torch.zeros_like(obs, device=self.device)
            next_obs_buf = torch.zeros_like(next_obs, device=self.device)
            reward_buf = torch.zeros_like(reward, device=self.device)
            nonterminal_buf = torch.zeros_like(nonterminal, device=self.device, dtype=torch.bool)
            gamma_buf = torch.zeros_like(gamma, device=self.device)
            action_buf = torch.zeros_like(action, device=self.device, dtype=torch.long)

            batch = TensorDict({
                "observation": obs_buf,

                "next": TensorDict({
                    "observation": next_obs_buf,
                    "reward": reward_buf}, device=self.device),

                "nonterminal": nonterminal_buf,
                "gamma": gamma_buf,
                "action": action_buf}, device=self.device)
            self.set_static_batch(batch)
        batch = (obs, next_obs, reward, nonterminal, gamma, action)
        return batch

    def _deserialize_tensor(self, serialized_batch, idx):
        len_shape = int(serialized_batch[idx])
        idx += 1

        shape = serialized_batch[idx:idx + len_shape]
        idx += len_shape

        n_elements = int(serialized_batch[idx])
        idx += 1

        tensor = torch.as_tensor(serialized_batch[idx:idx + n_elements], dtype=torch.float32, device='cpu').view(shape)
        idx += n_elements
        return tensor, idx

    def empty(self):
        pass