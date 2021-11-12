import sys
sys.path.append("..")

import torch
import torchvision

from utils import init_logger, log
from MAF import MAF, MAFImageTransform


BATCH_SIZE = 64 # increase to 64 to train
HIDDEN_DIM = 2048
NUM_BLOCKS = 2
LR = 1e-4
device = torch.device("cuda")

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
            sum += model.get_log_prob(images).mean().item()
    return -sum / len(dataloader)
    

def train(train_dataset, test_dataset):
    dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    image_shape = train_dataset[0][0].shape
    model = MAF(image_shape, HIDDEN_DIM, NUM_BLOCKS, LR, device)

    log().add_plot("loss", ["epoch", "nll_loss"])
    log().add_plot("test", ["epoch", "nll_loss"])

    epochs = 2
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs))
        sum_nll_loss, steps = 0., 0.
        D = torch.prod(torch.tensor(image_shape)).item()
        for batch_idx, (image, _) in enumerate(dataloader):
            nll_loss = model.train(image)

            steps += 1
            sum_nll_loss += nll_loss / D

            if batch_idx % 100 == 0:
                log().add_plot_point("loss", [epoch, sum_nll_loss / steps])

        log().add_plot_point("test", [epoch, test(test_dataset, model)/D])
        model.save("../pretrained/MAF.torch")


def load_dataset(dataset):
    transform = MAFImageTransform(dataset)
    if dataset == "mnist":
        train_dataset = torchvision.datasets.MNIST("../datasets", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST("../datasets", train=False, download=True, transform=transform)
    return train_dataset, test_dataset



if __name__ == "__main__":
    init_logger("../logdir", "tmplol")
    train_dataset, test_dataset = load_dataset("mnist")
    train(train_dataset, test_dataset)

