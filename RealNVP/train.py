import sys
sys.path.append("..")

import torch
import torchvision
import matplotlib.pyplot as plt
import math

from RealNVP import RealNVP, RealNVPImageTransform
from RealNVP_linear import RealNVPLinear
from utils import init_logger, log

BATCH_SIZE = 2 # increase to 64 to train
LR = 5e-4
NUM_RESNET = 2
HIDDEN_CHANNELS = 1024
device = torch.device("cpu")

def sample(model):
    model.model.eval()
    alpha = 0.01
    images = model.sample(10)
    images = ((torch.sigmoid(images) - alpha) / (1 - 2 * alpha)).clip(0., 1.)
    grid_img = torchvision.utils.make_grid(images, nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
    plt.show()


def test(test_dataset, model):
    dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False, num_workers=4)
    sum = 0.
    for images, _ in dataloader:
        with torch.no_grad():
            sum += model.get_log_prob(torch.flatten(images, start_dim=1)).mean().item()
    return sum / len(dataloader)
    

def train(train_dataset, test_dataset):
    dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    image_shape = train_dataset[0][0].shape
    D = torch.prod(torch.tensor(image_shape)).item()
    model = RealNVPLinear(
            input_shape=D,
            hidden_shape=HIDDEN_CHANNELS,
            num_hidden=NUM_RESNET,
            lr=LR,
            device=device
    )
    model.load("../pretrained/RealNVP.torch")
    sample(model)
    return

    log().add_plot("loss", ["epoch", "nll_loss", "l2reg"])
    log().add_plot("test", ["epoch", "nll_loss"])

    epochs = 2
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs))
        sum_nll_loss, sum_l2reg, steps = 0., 0., 0
        for batch_idx, (image, _) in enumerate(dataloader):
            nll_loss, l2reg = model.train(torch.flatten(image, start_dim=1))

            steps += 1
            sum_nll_loss += nll_loss / D
            sum_l2reg += l2reg / D

            if batch_idx % 100 == 0:
                log().add_plot_point("loss", [epoch, sum_nll_loss / steps, sum_l2reg / steps])

        log().add_plot_point("test", [epoch, test(test_dataset, model)/D])
        model.save("../pretrained/RealNVP.torch")


def load_dataset(dataset):
    transform = RealNVPImageTransform(dataset)
    if dataset == "celeba":
        train_dataset = torchvision.datasets.CelebA("../datasets", split="train", download=True, transform=transform)
        test_dataset = torchvision.datasets.CelebA("../datasets", split="test", download=True, transform=transform)
    elif dataset == "mnist":
        train_dataset = torchvision.datasets.MNIST("../datasets", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST("../datasets", train=False, download=True, transform=transform)
    return train_dataset, test_dataset



if __name__ == "__main__":
    init_logger("../logdir", "tmplol")
    train_dataset, test_dataset = load_dataset("mnist")
    train(train_dataset, test_dataset)
