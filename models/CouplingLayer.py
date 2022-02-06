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

        modules = [nn.utils.weight_norm(nn.Conv2d(image_channels, hidden_channels, kernel_size=3, padding=1)), nn.ReLU()] \
            + [ResnetBlock(hidden_channels, True, True) for _ in range(num_resnet)] \
            + [nn.utils.weight_norm(nn.Conv2d(hidden_channels, 2 * image_channels, kernel_size=3, padding=1))]

        self.model = nn.Sequential(*modules)

    def forward_flow(self, x):
        masked_x = x * self.mask
        log_s, t = self.model(masked_x).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * self.log_scale_scale
        return masked_x + (1 - self.mask) * (x * torch.exp(log_s) + t), (log_s * (1 - self.mask)).sum(dim=(1,2,3))

    def inverse_flow(self, x):
        masked_x = x * self.mask
        log_s, t = self.model(masked_x).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * self.log_scale_scale
        return masked_x + (1 - self.mask) * ((x - t) * torch.exp(-log_s)), -(log_s * (1 - self.mask)).sum(dim=(1,2,3))


class CouplingLayerLinear(Flow):
    def __init__(self, input_shape, hidden_shape, num_hidden, mask):
        super(CouplingLayerLinear, self).__init__()
        assert(mask.shape == torch.Size([input_shape]))

        self.register_buffer("mask", mask)
        self.register_parameter("log_scale_scale", nn.Parameter(torch.tensor(0., dtype=torch.float)))

        modules = [nn.utils.weight_norm(nn.Linear(input_shape, hidden_shape)), nn.ReLU()] \
            + sum([[nn.utils.weight_norm(nn.Linear(hidden_shape, hidden_shape)), nn.ReLU()] for _ in range(num_hidden)], []) \
            + [nn.utils.weight_norm(nn.Linear(hidden_shape, 2 * input_shape))]

        self.model = nn.Sequential(*modules)

    def forward_flow(self, x):
        masked_x = x * self.mask
        log_s, t = self.model(masked_x).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * self.log_scale_scale
        return masked_x + (1 - self.mask) * (x * torch.exp(log_s) + t), (log_s * (1 - self.mask)).sum(dim=1)

    def inverse_flow(self, x):
        masked_x = x * self.mask
        log_s, t = self.model(masked_x).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * self.log_scale_scale
        return masked_x + (1 - self.mask) * ((x - t) * torch.exp(-log_s)), -(log_s * (1 - self.mask)).sum(dim=1)
