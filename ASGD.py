from torch.optim import Optimizer
import torch
import numpy as np


class ASGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, {"lr": lr})
        self.lr = lr
        self.coef = 1
        self._compute_coef()

    @torch.no_grad()
    def step(self, closure=None):
        flat_grad = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    flat_grad.append(p.grad.view(-1))
                    # flat_grad = torch.cat((flat_grad, p.grad.view(-1)))

        flat_grad = torch.cat(flat_grad, dim=0)
        norm = flat_grad.norm()
        coef = self.lr * self.coef / norm

        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                step = p.numel()
                grad_slice = flat_grad[idx:idx+step].view_as(p)
                p.sub_(coef*grad_slice)
                idx += step

    def _compute_coef(self):
        d = 0
        for group in self.param_groups:
            for p in group["params"]:
                d += p.numel()
        self.coef = np.sqrt(d)/5
