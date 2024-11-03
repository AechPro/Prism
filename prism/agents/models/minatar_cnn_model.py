import torch
import torch.nn as nn
import time


class MinAtarModel(nn.Module):
    def __init__(self, in_channels, act_fn=nn.ReLU, device="cpu", channels_first=True,
                 use_layer_norm=False):

        super().__init__()
        self.transposed_shape = (-1, in_channels, 10, 10)
        self.channels_first = channels_first
        if use_layer_norm:
            embedding_net = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
                act_fn(),
                nn.Flatten(),
            )
        else:
            embedding_net = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
                act_fn(),
                nn.Flatten(),
            )

        with torch.no_grad():
            self.output_dim = embedding_net(torch.zeros(1, in_channels, 10, 10)).numel()

        self.model = embedding_net.to(device)

    def forward(self, x):
        # if not self.channels_first:
        #     x = x.transpose(1, -1)
        # x = x.view(self.transposed_shape)
        x = x.permute(0, 3, 1, 2).float()

        return self.model(x)

    def log(self, logger):
        pass
