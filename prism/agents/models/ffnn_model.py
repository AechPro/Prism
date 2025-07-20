import torch
import torch.nn as nn
import numpy as np


class PIDActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.kp = nn.Parameter(torch.tensor(1.0))
        self.ki = nn.Parameter(torch.tensor(0.1))
        self.kd = nn.Parameter(torch.tensor(0.1))

        self.integral = None
        self.prev_x = None

    def forward(self, x):
        if self.integral is None or x.numel() != self.prev_x.numel():
            self.integral = torch.zeros_like(x, requires_grad=False)

        if self.prev_x is None or x.numel() != self.prev_x.numel():
            derivative = 0
        else:
            derivative = x - self.prev_x

        output = self.kp * x + self.ki * (self.integral + x) + self.kd * derivative

        self.integral += x.detach().clone()
        self.prev_x = x.detach().clone()

        return output

    def reset(self):
        self.integral = None
        self.prev_x = None


class PNorm(nn.Module):
    def forward(self, x):
        if len(x.shape) == 2:
            norm = x.norm(dim=1, keepdim=True)
        else:
            norm = x.norm()
        return x / (norm + 1e-12)


class FFNNModel(nn.Module):
    def __init__(self, n_input_features,
                 n_output_features,
                 n_layers, layer_width,
                 use_layer_norm,
                 use_p_norm=False,
                 apply_layer_norm_first_layer=True,
                 output_act_fn=None,
                 act_fn=nn.ReLU,
                 sparse_init_p=0.0,
                 device="cpu"):

        super().__init__()
        self.device = device

        layers = []
        in_widths = [n_input_features] + [layer_width for _ in range(n_layers - 1)]
        out_widths = [layer_width for _ in range(n_layers-1)] + [n_output_features]
        for i in range(n_layers):
            if use_layer_norm:
                if apply_layer_norm_first_layer and i == 0:
                    layers.append(nn.LayerNorm(in_widths[i]))
                elif i != 0:
                    layers.append(nn.LayerNorm(in_widths[i]))

            layers.append(nn.Linear(in_widths[i], out_widths[i]))
            if i != n_layers - 1:
                layers.append(act_fn())

        if output_act_fn is not None:
            layers.append(output_act_fn())

        if use_p_norm:
            layers.insert(-2, PNorm())

        self.model = nn.Sequential(*layers).to(device)

        if sparse_init_p > 0.0:
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    mask = torch.bernoulli(torch.ones_like(layer.weight) * (1 - sparse_init_p))
                    layer.weight.data *= mask

    def forward(self, x):
        if type(x) is not torch.Tensor:
            if type(x) is not np.array:
                x = np.asarray(x, dtype=np.float32)
            x = torch.from_numpy(x).to(self.device)
        x = x.view(x.shape[0], -1)
        return self.model(x)

    def log(self, logger):
        pass
