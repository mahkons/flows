import torch
import torch.nn as nn
import torchvision.transforms as T

from models import SequentialFlow, Flow, CouplingLayerLinear, ActNorm
from utils import get_mask

SCALE_L2_REG_COEFF = 5e-5
MAX_GRAD_NORM = 100.

class RealNVPLinear():
    def __init__(self, input_shape, hidden_shape, num_hidden, lr, device):
        self.input_shape = input_shape
        self.device = device

        mask = torch.arange(input_shape) % 2
        self.model = SequentialFlow([
                CouplingLayerLinear(input_shape, hidden_shape, num_hidden, mask),
                ActNorm(input_shape),
                CouplingLayerLinear(input_shape, hidden_shape, num_hidden, 1 - mask),
                ActNorm(input_shape),
                CouplingLayerLinear(input_shape, hidden_shape, num_hidden, mask),
                ActNorm(input_shape),
                CouplingLayerLinear(input_shape, hidden_shape, num_hidden, 1 - mask),
                ActNorm(input_shape),

                CouplingLayerLinear(input_shape, hidden_shape, num_hidden, mask),
                ActNorm(input_shape),
                CouplingLayerLinear(input_shape, hidden_shape, num_hidden, 1 - mask),
                ActNorm(input_shape),
                CouplingLayerLinear(input_shape, hidden_shape, num_hidden, mask),
                ActNorm(input_shape),

                CouplingLayerLinear(input_shape, hidden_shape, num_hidden, mask),
                ActNorm(input_shape),
                CouplingLayerLinear(input_shape, hidden_shape, num_hidden, 1 - mask),
                ActNorm(input_shape),
                CouplingLayerLinear(input_shape, hidden_shape, num_hidden, mask),
        ])
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step, 1000) / 1000.)
        self.prior = torch.distributions.Normal(torch.tensor(0., device=device),
                torch.tensor(1., device=device))

        self.initialized = False


    def train(self, inputs):
        inputs = inputs.to(self.device)
        if not self.initialized:
            with torch.no_grad():
                self.model.data_init(inputs)
            self.initialized = True

        log_prob = self.get_log_prob(inputs).mean()

        l2reg = sum([SCALE_L2_REG_COEFF * (module.log_scale_scale ** 2).sum()
            for module in self.model.flow_modules if isinstance(module, CouplingLayerLinear)])

        loss = l2reg - log_prob
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        self.scheduler.step()

        return -log_prob.item(), l2reg.item()

    def get_log_prob(self, inputs):
        inputs = inputs.to(self.device)
        prediction, log_jac = self.model.forward_flow(inputs)
        log_prob = self.prior.log_prob(prediction).sum(dim=1) + log_jac
        return log_prob

    def sample(self, batch_size):
        with torch.no_grad():
            z = self.prior.sample([batch_size] + list(self.input_shape))
            x, _ = self.model.inverse_flow(z)
            return x.reshape((3, 28, 28))


    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])


"""
    for celeba center crop and resize as in paper
    uniform noise to dequantize input
    logit(a + (1 - 2a) * image) as in paper
"""
class RealNVPImageTransform():
    def __init__(self, dataset):
        if dataset == "celeba":
            self.base_transform = T.Compose([T.ToTensor(), T.CenterCrop((148, 148)), T.Resize((64, 64)), T.RandomHorizontalFlip()])
            self.alpha = 0.05
        elif dataset == "mnist":
            self.base_transform = T.Compose([T.ToTensor(), T.RandomHorizontalFlip()])
            self.alpha = 0.01
        else:
            raise AttributeError("Unknown dataset")


    def __call__(self, image):
        image = self.base_transform(image)
        noise = (torch.rand_like(image) - 0.5) * (1/256.)
        image = (image + noise).clip(0., 1.)
        return torch.logit(self.alpha +  (1 - 2 * self.alpha) * image)




