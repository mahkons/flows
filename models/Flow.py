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

    def data_init(self, x):
        return self.forward_flow(x)[0]



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

    def data_init(self, x):
        for module in self.flow_modules:
            x = module.data_init(x)
        return x


class ConditionalFlow(Flow):
    def __init__(self):
        super(ConditionalFlow, self).__init__()

    def forward_flow(self, x, condition=None):
        """ returns forward flow computation result and log of jacobian """
        raise NotImplementedError

    def inverse_flow(self, x, condition=None):
        """ returns inverse flow computation result and log of jacobian  """
        raise NotImplementedError

    def data_init(self, x, condition=None):
        return self.forward_flow(x, condition)[0]



class SequentialConditionalFlow(ConditionalFlow):
    def __init__(self, modules):
        super(SequentialConditionalFlow, self).__init__()

        self.flow_modules = list(modules)
        for idx, module in enumerate(self.flow_modules):
            assert(isinstance(module, ConditionalFlow))
            self.add_module(str(idx), module)

    def forward_flow(self, x, condition):
        log_sum = torch.zeros(x.shape[0], dtype=torch.float, device=self.device)
        for module in self.flow_modules:
            x, log_jac = module.forward_flow(x, condition)
            log_sum += log_jac
        return x, log_sum

    def inverse_flow(self, x, condition):
        log_sum = torch.zeros(x.shape[0], dtype=torch.float, device=self.device)
        for module in reversed(self.flow_modules):
            x, log_jac = module.inverse_flow(x, condition)
            log_sum += log_jac
        return x, log_sum

    def data_init(self, x, condition):
        for module in self.flow_modules:
            x = module.data_init(x, condition)
        return x



