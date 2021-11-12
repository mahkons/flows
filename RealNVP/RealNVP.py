import torch
import torch.nn as nn
import torchvision.transforms as T

from models import SequentialFlow, Flow, CouplingLayer, ActNormImage
from utils import get_mask

SCALE_L2_REG_COEFF = 5e-5
MAX_GRAD_NORM = 100.

class RealNVP():
    def __init__(self, image_shape, hidden_channels, num_resnet, lr, device):
        assert(len(image_shape) == 3)
        self.image_shape = image_shape
        self.device = device

        squeezed_shape = (image_shape[0] * 4, image_shape[1] // 2, image_shape[2] // 2)
        mask = get_mask(image_shape, "checkerboard", device)
        channelwise_mask = get_mask(squeezed_shape, "channelwise", device)
        self.model = SequentialFlow([
                CouplingLayer(image_shape, hidden_channels, num_resnet, mask),
                ActNormImage(image_shape[0]),
                CouplingLayer(image_shape, hidden_channels, num_resnet, 1 - mask),
                ActNormImage(image_shape[0]),
                CouplingLayer(image_shape, hidden_channels, num_resnet, mask),
                ActNormImage(image_shape[0]),
                CouplingLayer(image_shape, hidden_channels, num_resnet, 1 - mask),
                ActNormImage(image_shape[0]),
                Squeeze(),

                CouplingLayer(squeezed_shape, hidden_channels, num_resnet, channelwise_mask),
                ActNormImage(squeezed_shape[0]),
                CouplingLayer(squeezed_shape, hidden_channels, num_resnet, 1 - channelwise_mask),
                ActNormImage(squeezed_shape[0]),
                CouplingLayer(squeezed_shape, hidden_channels, num_resnet, channelwise_mask),
                ActNormImage(squeezed_shape[0]),
                Unsqueeze(),

                CouplingLayer(image_shape, hidden_channels, num_resnet, mask),
                ActNormImage(image_shape[0]),
                CouplingLayer(image_shape, hidden_channels, num_resnet, 1 - mask),
                ActNormImage(image_shape[0]),
                CouplingLayer(image_shape, hidden_channels, num_resnet, mask),
        ])
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=1000)
        self.prior = torch.distributions.Normal(torch.tensor(0., device=device),
                torch.tensor(1., device=device))

        self.initialized = False


    def train(self, images):
        images = images.to(self.device)
        if not self.initialized:
            with torch.no_grad():
                self.model.data_init(images)
            self.initialized = True

        log_prob = self.get_log_prob(images).mean()

        l2reg = sum([SCALE_L2_REG_COEFF * (module.log_scale_scale ** 2).sum()
            for module in self.model.flow_modules if isinstance(module, CouplingLayer)])

        loss = l2reg - log_prob
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        self.scheduler.step()

        return -log_prob.item(), l2reg.item()

    def get_log_prob(self, images):
        images = images.to(self.device)
        prediction, log_jac = self.model.forward_flow(images)
        log_prob = self.prior.log_prob(prediction).sum(dim=(1,2,3)) + log_jac
        return log_prob

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


def squeeze__(image):
    b, c, h, w = image.shape
    image = image.reshape(b, c, h//2, 2, w//2, 2).permute(0, 1, 3, 5, 2, 4)
    return image.reshape(b, 4 * c, h//2, w//2)

def unsqueeze__(image):
    b, c, h, w = image.shape
    image = image.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return image.reshape(b, c // 4, 2 * h, 2 * w)


class Squeeze(Flow):
    def forward_flow(self, image):
        return squeeze__(image), torch.zeros(image.shape[0], device=image.device)

    def inverse_flow(self, image):
        return unsqueeze__(image), torch.zeros(image.shape[0], device=image.device)


class Unsqueeze(Flow):
    def forward_flow(self, image):
        return unsqueeze__(image), torch.zeros(image.shape[0], device=image.device)

    def inverse_flow(self, image):
        return squeeze__(image), torch.zeros(image.shape[0], device=image.device)



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




