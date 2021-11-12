import torch
import torch.nn as nn

from .Flow import Flow


class Shuffle(Flow):
    def __init__(self, p):
        super(Shuffle, self).__init__()
        self.register_buffer("p", p)
        self.register_buffer("inv_p", __inverse_permutation(p))

    def forward_flow(self, x):
        return x[:, p],  0.

    def inverse_flow(self, x):
        return x[:, inv_p], 0.


class Reverse(Flow):
    def forward_flow(self, x):
        return torch.flip(x, [len(x.shape) - 1]), 0.

    def inverse_flow(self, x):
        return torch.flip(x, [len(x.shape) - 1]), 0.



def __inverse_permutation(p):
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.shape, device=p.device, dtype=a.dtype)
    return inv
