from abc import ABC, abstractmethod


class Experiment(ABC):
    def __init__(self, base_config, num_seeds=5):
        self.base_config = base_config
        self.num_seeds = num_seeds

        # Gym envs: "CartPole-v1", "LunarLander-v2", "Acrobot-v1", "MountainCar-v0"
        self.envs_to_test = ["MinAtar/SpaceInvaders-v1", "MinAtar/Breakout-v1",
                             "MinAtar/Asterix-v1", "MinAtar/Seaquest-v1", "MinAtar/Freeway-v1"]

        self.configs = []
        self._setup_configs()

    @abstractmethod
    def is_done(self):
        raise NotImplementedError

    @abstractmethod
    def get_next_config(self):
        raise NotImplementedError

    @abstractmethod
    def _setup_configs(self):
        raise NotImplementedError

    def __str__(self):
        return "ABC Experiment"
