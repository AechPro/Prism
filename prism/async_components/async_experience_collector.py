import time
from prism.async_components.redis import RedisInterface
from prism.experience import Timestep


class AsyncExperienceCollector(object):
    def __init__(self, config):
        self._redis_interface = RedisInterface(config["redis_host"], config["redis_port"])
        self._redis_interface.set_config(config["config"])
        self._last_timestep_measurement = 0
        self._last_update_timestamp = 0
        self._time_between_updates = 1

    def get_env_info(self):
        obs_shape, n_acts, n_agents = self._redis_interface.get_env_info()
        return obs_shape, n_acts, n_agents

    def signal_processes_start_collecting(self, agent):
        self._redis_interface.set_latest_model(agent.serialize_model(), agent.n_updates)
        self._redis_interface.set_current_command(RedisInterface.START_COLLECTING_COMMAND)

    def collect_timesteps(self, n_timesteps, agent, exp_buffer, random=False):
        n_collected = 0

        if time.perf_counter() - self._last_update_timestamp > self._time_between_updates:
            self._last_update_timestamp = time.perf_counter()
            total_timesteps = self._redis_interface.set_latest_model(agent.serialize_model(), agent.n_updates)

            if total_timesteps is not None and total_timesteps > self._last_timestep_measurement:
                n_collected = total_timesteps - self._last_timestep_measurement
                self._last_timestep_measurement = total_timesteps

        return n_collected

    def close(self):
        self._redis_interface.set_current_command(RedisInterface.SHUTDOWN_COMMAND)

    def log(self, logger):
        logger.log_data(data=self._redis_interface.get_training_reward(),
                        group_name="Report/Rewards",
                        var_name="Training Reward")
