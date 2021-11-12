import torch
import torch.nn as nn

from .Flow import Flow

class ActNormImage(Flow):
    def __init__(self, dim):
        super(ActNormImage, self).__init__()
        self.mean = nn.Parameter(torch.zeros((dim, 1, 1), dtype=torch.float))
        self.log_s = nn.Parameter(torch.zeros((dim, 1, 1), dtype=torch.float))

    def forward_flow(self, x):
        return x * torch.exp(-self.log_s) - self.mean, -self.log_s.sum().repeat(x.shape[0]) * x.shape[2] * x.shape[3]

    def inverse_flow(self, x):
        return (x + self.mean) * torch.exp(self.log_s), self.log_s.sum().repeat(x.shape[0]) * x.shape[2] * x.shape[3]

    def data_init(self, x):
        self.mean.data = x.mean(dim=(0, 2, 3))[:, None, None]
        d = ((x - self.mean) ** 2).mean(dim=(0, 2, 3))[:, None, None]
        self.log_s.data = torch.log(torch.sqrt(d))

        return self.forward_flow(x)[0]


class ActNorm(Flow):
    def __init__(self, dim):
        super(ActNorm, self).__init__()
        self.mean = nn.Parameter(torch.zeros((dim,), dtype=torch.float))
        self.log_s = nn.Parameter(torch.zeros((dim,), dtype=torch.float))

    def forward_flow(self, x):
        return x * torch.exp(-self.log_s) - self.mean, -self.log_s.sum().repeat(x.shape[0])

    def inverse_flow(self, x):
        return (x + self.mean) * torch.exp(self.log_s), self.log_s.sum().repeat(x.shape[0])

    def data_init(self, x):
        self.mean.data = x.mean(dim=(0,))
        d = ((x - self.mean) ** 2).mean(dim=(0,))
        self.log_s.data = torch.log(torch.sqrt(d))

        return self.forward_flow(x)[0]


