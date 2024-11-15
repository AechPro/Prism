import time
from prism.async_components.redis import RedisInterface
from prism.experience import Timestep, TimestepBuffer
from prism.factory import exp_buffer_factory


class AsyncExperienceBuffer(object):
    def __init__(self, config):
        self._batch_size = config.batch_size
        self._redis_interface = RedisInterface(config["redis_host"], config["redis_port"])
        self._experience_buffer = exp_buffer_factory.build_exp_buffer(config)
        self._waiting_timestep_id_map = {}

    def run(self):
        running = True
        while running:
            self._get_latest_timesteps()
            self._experience_buffer.sample(batch_size=self._batch_size)
            self._transmit_batch()

    def _get_latest_timesteps(self):
        serialized_timesteps = self._redis_interface.get_timesteps()

        (deserialized_timesteps,
         waiting_timestep_id_map) = Timestep.deserialize_linked_list(serialized_timesteps,
                                                                     self._waiting_timestep_id_map)

        self._waiting_timestep_id_map.clear()
        self._waiting_timestep_id_map = waiting_timestep_id_map

        for timestep in deserialized_timesteps:
            self._experience_buffer.extend(timestep)

    def _transmit_batch(self):
        serialized = []
        serialized += self._serialize_tensor(self._experience_buffer._obs)
        serialized += self._serialize_tensor(self._experience_buffer._next_obs)
        serialized += self._serialize_tensor(self._experience_buffer._reward)
        serialized += self._serialize_tensor(self._experience_buffer._nonterminal)
        serialized += self._serialize_tensor(self._experience_buffer._gamma)
        serialized += self._serialize_tensor(self._experience_buffer._action)

    def _serialize_tensor(self, tensor):
        n_elements = int(tensor.numel)
        shape = tensor.shape
        len_shape = len(shape)
        data = [len_shape, *shape, n_elements, *tensor.flatten().tolist()]

        return data
