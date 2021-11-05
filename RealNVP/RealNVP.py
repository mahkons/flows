import torch
import torch.nn as nn
import torchvision.transforms as T

from models import SequentialFlow, CouplingLayer
from utils import get_mask


class RealNVP():
    def __init__(self, image_shape, hidden_channels, num_coupling, lr, device):
        assert(len(image_shape) == 3)
        self.image_shape = image_shape
        self.device = device

        mask = get_mask(image_shape, "checkerboard", device)
        self.model = SequentialFlow([
            CouplingLayer(image_shape[0], hidden_channels, mask if i % 2 == 0 else 1 - mask)
            for i in range(num_coupling)
        ])
        self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.prior = torch.distributions.Normal(torch.tensor(0., device=device),
                torch.tensor(1., device=device))

    def train(self, images):
        images = images.to(self.device)
        prediction, log_jac = self.model.forward_flow(images)
        log_prob = self.prior.log_prob(prediction).sum(dim=(1,2,3)) + log_jac

        loss = -log_prob.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sample(self, batch_size):
        with torch.no_grad():
            z = self.prior.sample([batch_size] + list(self.image_shape))
            x, _ = self.model.inverse_flow(z)
            return x


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
    center crop and resize as in paper
    uniform noise to dequantize input
    logit(0.05 + 0.95 * image) as in paper
    scale to -1., 1.
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
        return torch.logit(self.alpha +  (1 - 2 * self.alpha) * image, eps=1e-3)




