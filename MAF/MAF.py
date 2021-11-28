import torch
import torch.nn as nn
import torchvision.transforms as T

from models import MADE, Shuffle, SequentialConditionalFlow, ActNorm

MAX_GRAD_NORM = 100.
WEIGHT_DECAY = 1e-6

class MAF(nn.Module):
    def __init__(self, flow_dim, cond_dim, hidden_dim, num_blocks, lr, device):
        super(MAF, self).__init__()
        self.flow_dim = flow_dim
        self.device = device

        self.model = SequentialConditionalFlow(sum(
            [[MADE(flow_dim, cond_dim, hidden_dim), ActNorm(flow_dim), Shuffle(torch.randperm(flow_dim))] \
                    for _ in range(num_blocks - 1)] \
            + [[MADE(flow_dim, cond_dim, hidden_dim)]], 
        []))
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step, 1000) / 1000.)
        self.prior = torch.distributions.Normal(torch.tensor(0., device=device),
                torch.tensor(1., device=device))

        self.initialized = False


    def train(self, inputs, conditions):
        if not self.initialized:
            with torch.no_grad():
                self.model.data_init(inputs, conditions)
            self.initialized = True
        loss = -self.get_log_prob(inputs, conditions).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def get_log_prob(self, inputs, conditions):
        prediction, log_jac = self.model.forward_flow(inputs, conditions)
        log_prob = self.prior.log_prob(prediction).sum(dim=1) + log_jac
        return log_prob

    def sample(self, batch_size, conditions):
        with torch.no_grad():
            z = self.prior.sample([batch_size, self.flow_dim])
            x, _ = self.model.inverse_flow(z, conditions)
        return x


    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])


"""
    uniform noise to dequantize input
    logit(a + (1 - 2a) * image) as in paper
"""
class MAFImageTransform():
    def __init__(self, dataset):
        if dataset == "mnist":
            self.base_transform = T.Compose([T.ToTensor(), T.RandomHorizontalFlip()])
            self.alpha = 0.01
        else:
            raise AttributeError("Unknown dataset")


    def __call__(self, image):
        image = self.base_transform(image)
        noise = (torch.rand_like(image) - 0.5) * (1/256.)
        image = (image + noise).clip(0., 1.)
        return torch.logit(self.alpha +  (1 - 2 * self.alpha) * image)




