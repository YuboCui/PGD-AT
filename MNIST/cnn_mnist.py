import torch
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear
import torchvision.transforms as transforms


class CNN_MNIST(torch.nn.Module):
    def __init__(self, scale=16, normalize=False):
        super().__init__()

        self.scale = scale
        self.normalize = normalize
        # Use the default initialization in torch

        self.normalize_trans = transforms.Normalize(mean=(0.1307,), std=(0.3081,))

        self.net = Sequential(
            Conv2d(1, 2 * self.scale, 5, padding=2),
            ReLU(),
            MaxPool2d(2),
            Conv2d(2 * self.scale, 4 * self.scale, 5, padding=2),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(7 * 7 * 4 * self.scale, 64 * self.scale),
            ReLU(),
            Linear(64 * self.scale, 10),
        )

    def forward(self, x):
        if self.normalize:
            x = self.normalize_trans(x)
        x = self.net(x)
        return x

    def set_grad(self, value):
        for p in self.parameters():
            p.requires_grad = value


if __name__ == "__main__":
    model = CNN_MNIST()

    x = torch.rand((4, 1, 28, 28))
    y = model(x)
    print(y.argmax(dim=1).shape)
    # print(y.shape, y.dtype)
