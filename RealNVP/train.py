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
NUM_COUPLING = 10
NUM_RESNET = 8
HIDDEN_CHANNELS = 64
device = torch.device("cpu")

def sample(model):
    images = model.sample(10)
    images = ((torch.sigmoid(images) - 0.01) / 0.98).clip(0., 1.)
    grid_img = torchvision.utils.make_grid(images, nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
    plt.show()


def train(dataset):
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    image_shape = dataset[0][0].shape
    model = RealNVP(
            image_shape=image_shape,
            hidden_channels=HIDDEN_CHANNELS,
            num_coupling=NUM_COUPLING,
            num_resnet=NUM_RESNET,
            lr=LR,
            device=device
    )

    epochs = 2
    for epoch in range(epochs):
        for batch_idx, (image, _) in enumerate(dataloader):
            loss = model.train(image)
            print("Epoch {}/{} Batch {}/{} Loss: {}"
                    .format(epoch, epochs, batch_idx, len(dataloader), loss))

            D = torch.prod(torch.tensor(image_shape))
            nbits = -loss[0] / (D * math.log(2)) - math.log(1 - 0.02) / math.log(2) + 8 \
                    + (torch.log(torch.sigmoid(image)) + torch.log(1 - torch.sigmoid(image))).sum(dim=(1,2,3)).mean() / (D * math.log(2))
            print("NBits {}".format(nbits))
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
    train(train_dataset)
