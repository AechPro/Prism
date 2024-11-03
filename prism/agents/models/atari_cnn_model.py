import torch
import torch.nn as nn
import time


class NatureAtariCnn(nn.Module):
    def __init__(self, frame_stack, feature_dim=512, act_fn=nn.ReLU, device="cpu", channels_first=True,
                 use_layer_norm=False, logger=None):

        super().__init__()
        self.channels_first = channels_first
        if use_layer_norm:
            embedding_net = nn.Sequential(
                nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4),
                act_fn(),

                nn.LayerNorm(normalized_shape=(32, 20, 20)),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                act_fn(),

                nn.LayerNorm(normalized_shape=(64, 9, 9)),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                act_fn(),
                nn.Flatten(),
            )
        else:
            embedding_net = nn.Sequential(
                nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4),
                act_fn(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                act_fn(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                act_fn(),
                nn.Flatten(),
            )
        with torch.no_grad():
            self.output_dim = embedding_net(torch.zeros(1, frame_stack, 84, 84)).numel()

        self.model = embedding_net.to(device)
        # if use_layer_norm:
        #     self.model = nn.Sequential(embedding_net, nn.LayerNorm(n_flatten), nn.Linear(n_flatten, feature_dim), nn.ReLU()).to(device)
        # else:
        #     self.model = nn.Sequential(embedding_net, nn.Linear(n_flatten, feature_dim), nn.ReLU()).to(device)

    def forward(self, x):
        if not self.channels_first:
            x = x.transpose(1, -1)
        return self.model(x)

    def log(self, logger):
        pass
