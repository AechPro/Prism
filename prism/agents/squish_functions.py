import torch


def symlog(x):
    return torch.sign(x) * torch.log(x.abs() + 1)


def symexp(x):
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


def obs_look_further_squish_fn(x):
    # https://arxiv.org/pdf/1805.11593.pdf
    return torch.sign(x) * (torch.sqrt(x.abs() + 1) - 1) + 0.01 * x


def obs_look_further_squish_fn_inverse(y):
    return torch.sign(y) * (torch.square((torch.sqrt(1 + 4 * 0.01 * (torch.abs(y) + 1 + 0.01)) - 1) / (2 * 0.01)) - 1)


def passthrough(x):
    return x