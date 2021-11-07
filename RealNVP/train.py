import sys
sys.path.append("..")

import torch
import torchvision
import matplotlib.pyplot as plt
import math

from RealNVP import RealNVP, RealNVPImageTransform
from utils import init_logger, log

BATCH_SIZE = 2
LR = 1e-3
NUM_RESNET = 4
HIDDEN_CHANNELS = 64
device = torch.device("cpu")

def sample(model):
    images = model.sample(10)
    images = ((torch.sigmoid(images) - 0.01) / 0.98).clip(0., 1.)
    grid_img = torchvision.utils.make_grid(images, nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
    plt.show()


def test(test_dataset, model):
    dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False, num_workers=4)
    sum = 0.
    for images, _ in dataloader:
        with torch.no_grad():
            sum += model.get_log_prob(images).mean().item()
    return sum / len(dataloader)
    

def train(train_dataset, test_dataset):
    dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    image_shape = train_dataset[0][0].shape
    model = RealNVP(
            image_shape=image_shape,
            hidden_channels=HIDDEN_CHANNELS,
            num_resnet=NUM_RESNET,
            lr=LR,
            device=device
    )
    #  model.load("../pretrained/RealNVP.torch")
    #  sample(model)
    #  return

    log().add_plot("loss", ["epoch", "nll_loss", "l2reg"])
    log().add_plot("test", ["epoch", "nll_loss"])

    epochs = 2
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs))
        sum_nll_loss, sum_l2reg, steps = 0., 0., 0
        D = torch.prod(torch.tensor(image_shape)).item()
        for batch_idx, (image, _) in enumerate(dataloader):
            nll_loss, l2reg = model.train(image)

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
