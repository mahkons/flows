import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResnetBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
        )

    def forward(self, x):
        return F.relu(x + self.model(x))

