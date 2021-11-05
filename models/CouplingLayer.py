import torch
import torch.nn as nn

from .Flow import Flow

class CouplingLayer(Flow):
    def __init__(self, image_channels, hidden_channels, mask):
        super(CouplingLayer, self).__init__()

        self.register_buffer("mask", mask)
        self.register_parameter("scale_scale", torch.tensor(1., dtype=torch.float))

        self.scale_net = nn.Sequential(
            nn.Conv2d(image_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, image_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Conv2d(image_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, image_channels, kernel_size=3, padding=1)
        )

    def forward_flow(self, x):
        masked_x = x * self.mask
        log_s = self.scale_scale * self.scale_net(masked_x)
        t = self.translate_net(masked_x)
        return masked_x + (1 - self.mask) * (x * torch.exp(log_s) + t)

    def inverse_flow(self, x):
        masked_x = x * self.mask
        log_s = self.scale_scale * self.scale_net(masked_x)
        t = self.translate_net(masked_x)
        return masked_x + (1 - self.mask) * (x * torch.exp(log_s) + t)
