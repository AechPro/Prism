import torch
import torch.nn as nn
import time
import math


class NatureAtariCnn(nn.Module):
    def __init__(self, frame_stack, feature_dim=512, act_fn=nn.ReLU, device="cpu", channels_first=True,
                 use_layer_norm=False, logger=None, sparse_init_p=0.0):

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
            if sparse_init_p > 0:
                for layer in embedding_net:
                    if isinstance(layer, nn.Conv2d):
                        nn.init.zeros_(layer.bias)
                        tensor = layer.weight
                        channels_out, channels_in, h, w = tensor.shape
                        fan_in, fan_out = channels_in * h * w, channels_out * h * w
                        num_zeros = int(math.ceil(sparse_init_p * fan_in))
                        tensor.uniform_(-math.sqrt(1.0 / fan_in), math.sqrt(1.0 / fan_in))
                        for out_channel_idx in range(channels_out):
                            indices = torch.randperm(fan_in)
                            zero_indices = indices[:num_zeros]
                            tensor[out_channel_idx].reshape(channels_in * h * w)[zero_indices].mul_(0)

        self.model = embedding_net.to(device)

    def forward(self, x):
        if not self.channels_first:
            x = x.transpose(1, -1)
        return self.model(x)

    def log(self, logger):
        pass
