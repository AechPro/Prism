import numpy as np


class AnnealingStrategy(object):
    def __init__(self):
        self.current_step = 0

    def update(self, n_steps):
        self.current_step += n_steps
        return self.get_value()

    def get_value(self):
        raise NotImplementedError

    def get_state(self):
        return self.current_step

    def set_state(self, state):
        self.current_step = state


class LinearAnneal(AnnealingStrategy):
    def __init__(self, start_value, stop_value, max_steps):
        super().__init__()
        self.start = start_value
        self.stop = stop_value
        self.max_steps = max_steps

    def get_value(self):
        if self.current_step >= self.max_steps or self.max_steps == 0:
            return self.stop

        alpha = min(1, self.current_step / self.max_steps)
        return self.start * (1 - alpha) + self.stop*alpha


class CosineAnneal(AnnealingStrategy):
    def __init__(self, scaling_coefficient, period_steps):
        super().__init__()
        self.scale = scaling_coefficient
        self.period = period_steps

    def get_value(self):
        return self.scale * (np.cos(np.pi * self.current_step / self.period) / 2 + 0.5)
