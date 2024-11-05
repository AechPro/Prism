import torch


class ObservationStacker(object):
    def __init__(self, frame_stack, device="cpu"):
        self.frame_stack = frame_stack
        self.device = device
        self.obs_stack = None

    def stack(self, obs, n_agents=1):
        if self.obs_stack is None:
            self.reset(obs, n_agents)
        else:
            if self.frame_stack == 1:
                self.obs_stack = obs
            else:
                self.obs_stack = torch.roll(self.obs_stack, shifts=-1, dims=1)
                self.obs_stack[:, -1, ...] = obs

    def reset(self, initial_obs, n_agents=1):
        if self.frame_stack == 1:
            self.obs_stack = initial_obs
        else:
            obs_shape = initial_obs.shape
            frame_stack_shape = (n_agents, self.frame_stack, *obs_shape[1:])
            self.obs_stack = torch.zeros(frame_stack_shape, dtype=torch.float32, device=self.device)
            self.obs_stack[:, -1, ...] = initial_obs
