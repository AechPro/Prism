from redis import Redis
from prism.async_components import compression_methods
import time


class RedisInterface(object):
    TIMESTEPS_KEY = "timesteps"
    MODEL_PARAMS_KEY = "model"
    CURRENT_EPOCH_KEY = "current_epoch"
    CONFIG_KEY = "config"
    CURRENT_COMMAND_KEY = "current_command"
    TOTAL_TIMESTEPS_COLLECTED_KEY = "total_timesteps_collected"
    ENV_INFO_KEY = "env_info"
    TRAINING_REWARD_KEY = "training_reward"

    START_COLLECTING_COMMAND = "start_collecting"
    SHUTDOWN_COMMAND = "shutdown"

    def __init__(self, host='localhost', port=6379):
        self.redis = Redis(host=host, port=port)
        self.serializer = compression_methods.MessageSerializer()
        self.max_queue_size = 100_000
        self.last_known_epoch = None
        self.waiting_timestep_id_map = {}

    def get_training_reward(self):
        return self.redis.get(RedisInterface.TRAINING_REWARD_KEY)

    def set_training_reward(self, reward):
        self.redis.set(RedisInterface.TRAINING_REWARD_KEY, reward)

    def get_current_command(self):
        return self.redis.get(RedisInterface.CURRENT_COMMAND_KEY)

    def set_current_command(self, command):
        self.redis.set(RedisInterface.CURRENT_COMMAND_KEY, command)

    def get_config(self):
        return self.redis.get(RedisInterface.CONFIG_KEY)

    def set_config(self, config):
        self.redis.set(RedisInterface.CONFIG_KEY, config)

    def get_env_info(self):
        env_info_vector = self.redis.get(RedisInterface.ENV_INFO_KEY)
        while env_info_vector is None:
            print("Awaiting env info...")
            time.sleep(1)
            env_info_vector = self.redis.get(RedisInterface.ENV_INFO_KEY)

        env_info_vector = self.serializer.unpack(env_info_vector)

        n_elements_in_shape = env_info_vector[0]
        obs_shape = env_info_vector[1:n_elements_in_shape + 1]
        n_acts = env_info_vector[n_elements_in_shape + 1]
        n_agents = env_info_vector[n_elements_in_shape + 2]

        print("Env info:", obs_shape, n_acts, n_agents)
        return obs_shape, n_acts, n_agents

    def set_env_info(self, obs_shape, n_acts, n_agents):
        n_elements_in_shape = len(obs_shape)
        env_info_vector = [n_elements_in_shape, *obs_shape, n_acts, n_agents]
        self.redis.set(RedisInterface.ENV_INFO_KEY, self.serializer.pack(env_info_vector))

    def get_latest_model(self):
        current_epoch = self.redis.get(RedisInterface.CURRENT_EPOCH_KEY)

        if current_epoch is not None:
            current_epoch = int(current_epoch)

            if current_epoch != self.last_known_epoch:
                serialized_model_params = self.redis.get(RedisInterface.MODEL_PARAMS_KEY)
                self.last_known_epoch = current_epoch

                return self.serializer.unpack(serialized_model_params)
        return None

    def set_latest_model(self, serialized_model, current_epoch):
        pipe = self.redis.pipeline()
        pipe.set(RedisInterface.CURRENT_EPOCH_KEY, current_epoch)
        pipe.set(RedisInterface.MODEL_PARAMS_KEY, self.serializer.pack(serialized_model))
        pipe.get(RedisInterface.TOTAL_TIMESTEPS_COLLECTED_KEY)
        total_timesteps = pipe.execute()[0]

        return total_timesteps

    def submit_timesteps(self, timesteps):
        serialized = []
        for timestep in timesteps:
            serialized += timestep.serialize()
        packed_timesteps = self.serializer.pack(serialized)

        pipe = self.redis.pipeline()
        pipe.lpush(RedisInterface.TIMESTEPS_KEY, packed_timesteps)
        pipe.ltrim(RedisInterface.TIMESTEPS_KEY, 0, self.max_queue_size)
        pipe.incrby(RedisInterface.TOTAL_TIMESTEPS_COLLECTED_KEY, len(timesteps))
        pipe.execute()

    def get_timesteps(self):
        pipe = self.redis.pipeline()

        pipe.lrange(RedisInterface.TIMESTEPS_KEY, 0, -1)
        pipe.delete(RedisInterface.TIMESTEPS_KEY)

        packed_timesteps = pipe.execute()[0]

        serialized_timesteps = []
        for packed_list in packed_timesteps:
            serialized_timesteps += self.serializer.unpack(packed_list)

        return serialized_timesteps
