import torch
import torch.nn

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_feature, mask, bias=True, device=None, dtype=None):
        # TODO
        self.register_buffer("mask", mask)
        assert(len(mask.shape) == 1)

    def forward(self, x):
        raise NotImplementedError


class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask,
            stride=1, padding=0, dilation=1, groups=1, bias=True,
            padding_mode="zeros", device=None, dtype=None)
        # TODO
        self.register_buffer("mask", mask)
        assert(len(mask.shape) == 3)

    def forward(self, x):
        return NotImplementedError
