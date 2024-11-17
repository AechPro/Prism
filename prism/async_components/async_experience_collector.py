import time
from prism.async_components.redis import RedisInterface
from prism.factory import collector_factory, agent_factory
from prism.async_components import AsyncExperienceBufferInterface


class AsyncExperienceCollector(object):
    def __init__(self, redis_host, redis_port):
        self._redis_interface = RedisInterface(redis_host, redis_port)
        self._agent = None
        self._collector = None
        self._config = None
        self._exp_buffer_interface = AsyncExperienceBufferInterface(redis_host, redis_port, "cpu")
        self._time_between_command_pings = 10.0
        self._time_since_last_command_ping = 0.0

    def wait_for_config(self):
        config = self._redis_interface.get_config()
        print("GOT CONFIG", config)
        while config is None:
            print("Async collector waiting for config...")
            time.sleep(0.1)
            config = self._redis_interface.get_config()
        config.run_through_redis = False
        self._collector = collector_factory.build_collector(config)
        self._config = config

    def setup(self):
        self.wait_for_config()
        obs_shape, n_acts, n_agents = self._collector.get_env_info()
        self._redis_interface.set_env_info(obs_shape, n_acts, n_agents)
        self._agent = agent_factory.build_agent(self._config, obs_shape, n_acts)

        latest_model = self._redis_interface.get_latest_model()
        while latest_model is None:
            print("Async collector waiting for latest model...")
            time.sleep(0.1)
            latest_model = self._redis_interface.get_latest_model()

        self._agent.deserialize_model(latest_model)
        self._agent.model.should_build_forward_cuda_graph = False
        self._collector.signal_processes_start_collecting(self._agent)
        self._agent.model.should_build_forward_cuda_graph = self._config.use_cuda_graph and "cuda" in self._config.device

    def run(self):
        self.setup()
        self._collector.collect_timesteps(self._config.num_initial_random_timesteps, None,
                                          self._exp_buffer_interface, random=True)

        running = True
        while running:
            t1 = time.perf_counter()
            self._collector.collect_timesteps(100, self._agent, self._exp_buffer_interface, random=False)
            print(time.perf_counter() - t1, "seconds to collect 100 timesteps")
            print(time.perf_counter() - self._time_since_last_command_ping, "|", self._time_between_command_pings)

            if time.perf_counter() - self._time_since_last_command_ping > self._time_between_command_pings:
                current_command = self._redis_interface.get_current_command()
                running = current_command != RedisInterface.SHUTDOWN_COMMAND

                latest_model = self._redis_interface.get_latest_model()
                if latest_model is not None:
                    self._agent.deserialize_model(latest_model)
                    print("Deserializing model")
                self._time_since_last_command_ping = time.perf_counter()

class AsyncExperienceCollectorInterface(object):
    def __init__(self, config):
        self._redis_interface = RedisInterface(config.redis_host, config.redis_port)
        self._redis_interface.clear_redis()

        original_side = config.redis_side
        config.redis_side = "client"
        serialized_config = config.serialize()
        config.redis_side = original_side

        self._redis_interface.set_config(serialized_config)

        self._last_timestep_measurement = 0
        self._last_update_timestamp = 0
        self._time_between_updates = 15

    def get_env_info(self):
        obs_shape, n_acts, n_agents = self._redis_interface.get_env_info()
        return obs_shape, n_acts, n_agents

    def signal_processes_start_collecting(self, agent):
        self._redis_interface.set_latest_model(agent.serialize_model(), agent.n_updates)
        self._redis_interface.set_current_command(RedisInterface.START_COLLECTING_COMMAND)

    def collect_timesteps(self, n_timesteps, agent, exp_buffer, random=False):
        n_collected = 0

        if time.perf_counter() - self._last_update_timestamp > self._time_between_updates:
            total_timesteps = self._redis_interface.set_latest_model(agent.serialize_model(), agent.n_updates)

            if total_timesteps is not None and total_timesteps > self._last_timestep_measurement:
                n_collected = total_timesteps - self._last_timestep_measurement
                self._last_timestep_measurement = total_timesteps
            self._last_update_timestamp = time.perf_counter()

        return n_collected

    def close(self):
        self._redis_interface.set_current_command(RedisInterface.SHUTDOWN_COMMAND)

    def log(self, logger):
        rew = self._redis_interface.get_training_reward()
        if rew is None:
            return

        logger.log_data(data=rew,
                        group_name="Report/Rewards",
                        var_name="Training Reward")
