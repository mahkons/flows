import sys
sys.path.append("..")

import torch
import torchvision
import matplotlib.pyplot as plt

from RealNVP import RealNVP, RealNVPImageTransform

BATCH_SIZE = 128
LR = 1e-3
NUM_COUPLING = 5
HIDDEN_CHANNELS = 32
device = torch.device("cuda")

def sample(model):
    images = model.sample(10)
    images = torch.sigmoid(images)
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
            lr=LR,
            device=device
    )

    epochs = 2
    for epoch in range(epochs):
        for batch_idx, (image, _) in enumerate(dataloader):
            loss = model.train(image)
            print("Epoch {}/{} Batch {}/{} Loss: {}"
                    .format(epoch, epochs, batch_idx, len(dataloader), loss))
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
    train_dataset, test_dataset = load_dataset("mnist")
    train(train_dataset)
