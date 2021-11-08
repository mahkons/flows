import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, n_channels, with_batch_norm=False, with_weight_norm=False):
        super(ResnetBlock, self).__init__()

        conv_layer = lambda: nn.Conv2d(n_channels, n_channels, 3, padding=1)

        modules = [nn.utils.weight_norm(conv_layer())] if with_weight_norm else [conv_layer()] \
                + [nn.BatchNorm2d(n_channels)] if with_batch_norm else [] \
                + [nn.ReLU()] \
                + [nn.utils.weight_norm(conv_layer())] if with_weight_norm else [conv_layer()]
                + [nn.BatchNorm2d(n_channels)] if with_batch_norm else [] \

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return F.relu(x + self.model(x))

