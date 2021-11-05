import sys
sys.path.append("..")

import torch
import torchvision
import matplotlib.pyplot as plt

from RealNVP import RealNVP, RealNVPImageTransform

BATCH_SIZE = 32
device = torch.device("cuda")

def sample(model):
    images = model.sample(10)
    images = torch.sigmoid(images)
    grid_img = torchvision.utils.make_grid(images, nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
    plt.show()


def train(dataset):
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model = RealNVP(
            image_shape=(3, 64, 64),
            hidden_channels=32,
            num_coupling=5,
            lr=1e-3,
            device=device
    )
    model.load("../pretrained/RealNVP.torch")
    sample(model)

    return

    for epoch in range(2):
        for image, _ in dataloader:
            loss = model.train(image)
            print(loss)
        model.save("../pretrained/RealNVP.torch")


def load_dataset(dataset):
    transform = RealNVPImageTransform(dataset)
    if dataset == "celeba":
        train_dataset = torchvision.datasets.CelebA("../datasets", split="train", download=True, transform=transform)
        test_dataset = torchvision.datasets.CelebA("../datasets", split="test", download=True, transform=transform)
    elif dataset == "mnist":
        train_dataset = torchvision.datasets.MNIST("../datasets", split="train", download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST("../datasets", split="test", download=True, transform=transform)
    return train_dataset, test_dataset



if __name__ == "__main__":
    train_dataset, test_dataset = load_dataset("celeba")
    train(train_dataset)
