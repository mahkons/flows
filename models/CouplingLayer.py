import torch
import torch.nn as nn

from .Flow import Flow
from .ResNet import ResnetBlock

class CouplingLayer(Flow):
    def __init__(self, image_shape, hidden_channels, num_resnet, mask):
        super(CouplingLayer, self).__init__()
        assert(image_shape == mask.shape)
        image_channels = image_shape[0]

        self.register_buffer("mask", mask)
        self.register_parameter("log_scale_scale", nn.Parameter(torch.tensor(0., dtype=torch.float)))

        modules_scale = [nn.Conv2d(image_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU()] \
            + [ResnetBlock(hidden_channels) for _ in range(num_resnet)] \
            + [nn.Conv2d(hidden_channels, image_channels, kernel_size=3, padding=1)]

        modules_translate = [nn.Conv2d(image_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU()] \
            + [ResnetBlock(hidden_channels) for _ in range(num_resnet)] \
            + [nn.Conv2d(hidden_channels, image_channels, kernel_size=3, padding=1)]

        self.scale_net = nn.Sequential(*modules_scale, nn.Tanh())
        self.translate_net = nn.Sequential(*modules_translate)

    def forward_flow(self, x):
        masked_x = x * self.mask
        log_s = self.log_scale_scale * self.scale_net(masked_x)
        t = self.translate_net(masked_x)
        return masked_x + (1 - self.mask) * (x * torch.exp(log_s) + t), (log_s * (1 - self.mask)).sum(dim=(1,2,3))

    def inverse_flow(self, x):
        masked_x = x * self.mask
        log_s = self.log_scale_scale * self.scale_net(masked_x)
        t = self.translate_net(masked_x)
        return masked_x + (1 - self.mask) * ((x - t) * torch.exp(-log_s)), -(log_s * (1 - self.mask)).sum(dim=(1,2,3))
