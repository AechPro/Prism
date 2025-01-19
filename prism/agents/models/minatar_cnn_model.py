import torch
import torch.nn as nn
import math
import time


class MinAtarModel(nn.Module):
    def __init__(self, in_channels, act_fn=nn.ReLU, device="cpu", use_layer_norm=False, sparse_init_p=0.0):

        super().__init__()
        self.transposed_shape = (-1, in_channels, 10, 10)
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
        x = x.permute(0, 3, 1, 2).float()

        return self.model(x)

    def log(self, logger):
        pass
