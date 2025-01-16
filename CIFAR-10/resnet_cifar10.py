import torch
from torch.nn import (
    Sequential,
    Conv2d,
    BatchNorm2d,
    LeakyReLU,
    AvgPool2d,
    Flatten,
    Linear,
)
from torchvision.transforms import v2


class ResNet_CIFAR10(torch.nn.Module):
    def __init__(self):
        super().__init__()

        strides = [1, 2, 2]
        activate_before_res = [True, False, False]
        # filters = [16, 16, 32, 64]
        filters = [16, 160, 320, 640]

        self.net = Sequential(
            # self.PerImageStandardization(),
            # [, 3, 32, 32]
            Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            # [, 16, 32, 32]
            self.Residual(filters[0], filters[1], strides[0], activate_before_res[0]),
            # [, 160, 32, 32]
            self.Residual(filters[1], filters[1], 1, False),
            self.Residual(filters[1], filters[1], 1, False),
            self.Residual(filters[1], filters[1], 1, False),
            self.Residual(filters[1], filters[1], 1, False),
            # [, 160, 32, 32]
            self.Residual(filters[1], filters[2], strides[1], activate_before_res[1]),
            # [, 320, 16, 16]
            self.Residual(filters[2], filters[2], 1, False),
            self.Residual(filters[2], filters[2], 1, False),
            self.Residual(filters[2], filters[2], 1, False),
            self.Residual(filters[2], filters[2], 1, False),
            # [, 320, 16, 16]
            self.Residual(filters[2], filters[3], strides[2], activate_before_res[2]),
            # [, 640, 8, 8]
            self.Residual(filters[3], filters[3], 1, False),
            self.Residual(filters[3], filters[3], 1, False),
            self.Residual(filters[3], filters[3], 1, False),
            self.Residual(filters[3], filters[3], 1, False),
            # [, 640, 8, 8]
            BatchNorm2d(filters[-1]),
            LeakyReLU(0.1),
            AvgPool2d(8),
            # [, 640, 1, 1]
            Flatten(),
            # [, 640]
            Linear(filters[-1], 10),
        )

    class PerImageStandardization(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            mean = torch.mean(x, dim=(-3, -2, -1), keepdim=True)
            stddev = torch.std(x, dim=(-3, -2, -1), unbiased=False, keepdim=True)
            n = (x[0].numel()) ** -0.5
            adjusted_stddev = stddev.clone()
            adjusted_stddev[stddev < n] = n
            return (x - mean) / adjusted_stddev

    class Residual(torch.nn.Module):
        def __init__(self, in_filter, out_filter, stride, activate_before_residual):
            super().__init__()
            self.in_filter = in_filter
            self.out_filter = out_filter
            self.activate_before_residual = activate_before_residual
            self.net1 = Sequential(
                BatchNorm2d(in_filter),
                LeakyReLU(0.1),
            )
            self.net2 = Sequential(
                Conv2d(in_filter, out_filter, 3, stride, padding=1, bias=False),
                BatchNorm2d(out_filter),
                LeakyReLU(0.1),
                Conv2d(out_filter, out_filter, 3, stride=1, padding=1, bias=False),
            )
            if in_filter != out_filter:
                self.net3 = AvgPool2d(stride)

        def forward(self, x):
            if self.activate_before_residual:
                x = self.net1(x)
                orig_x = x
            else:
                orig_x = x
                x = self.net1(x)
            x = self.net2(x)
            if self.in_filter != self.out_filter:
                padding = (self.out_filter - self.in_filter) // 2
                orig_x = self.net3(orig_x)
                orig_x = torch.nn.functional.pad(orig_x, (0, 0, 0, 0, padding, padding))
            return x + orig_x

    def forward(self, x):
        x = self.normalize(x)
        return self.net(x)

    def set_grad(self, value):
        for p in self.parameters():
            p.requires_grad = value

    def process(self, x):
        trans = v2.Compose(
            [v2.RandomCrop(size=(32, 32), padding=2), v2.RandomHorizontalFlip()]
        )
        return trans(x)

    def normalize(self, x):
        trans = v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        return trans(x)


if __name__ == "__main__":
    x = torch.ones((7, 3, 32, 32))
    model = ResNet_CIFAR10()
    y = model(x)
    print(y.shape)
