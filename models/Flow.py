import torch
import torch.nn as nn

class Flow(nn.Module):
    def __init__(self):
        super(Flow, self).__init__()
        self.device = torch.device("cpu")

    def to(self, device):
        super(Flow, self).to(device)
        self.device = device

    def forward_flow(self, x):
        """ returns forward flow computation result and log of jacobian """
        raise NotImplementedError

    def inverse_flow(self, x):
        """ returns inverse flow computation result and log of jacobian  """
        raise NotImplementedError



class SequentialFlow(Flow):
    def __init__(self, modules):
        super(SequentialFlow, self).__init__()

        self.flow_modules = list(modules)
        for idx, module in enumerate(self.flow_modules):
            assert(isinstance(module, Flow))
            self.add_module(str(idx), module)

    def forward_flow(self, x):
        log_sum = torch.zeros(x.shape[0], dtype=torch.float, device=self.device)
        for module in self.flow_modules:
            x, log_jac = module.forward_flow(x)
            log_sum += log_jac
        return x, log_sum

    def inverse_flow(self, x):
        log_sum = torch.zeros(x.shape[0], dtype=torch.float, device=self.device)
        for module in reversed(self.flow_modules):
            x, log_jac = module.inverse_flow(x)
            log_sum += log_jac
        return x, log_sum

