import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from resnet_cifar10 import ResNet_CIFAR10
from tqdm import tqdm
from torchattacks import PGD, FGSM


def evaluate_cifar10(model, dataloader, attack=None):
    grad = next(model.parameters()).requires_grad
    if attack and grad:
        model.set_grad(False)
    device = next(model.parameters()).device
    loss = 0.0
    acc = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum").to(device)

    for _, (x, y) in enumerate(tqdm(dataloader)):
        if attack == None:
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss += loss_fn(y_hat, y)
                acc += (y_hat.argmax(dim=1) == y).sum()
        else:
            x_adv = attack(x, y)
            with torch.no_grad():
                y = y.to(device)
                y_hat = model(x_adv)
                loss += loss_fn(y_hat, y)
                acc += (y_hat.argmax(dim=1) == y).sum()
    loss /= len(dataloader.dataset)
    acc /= len(dataloader.dataset)
    if attack and grad:
        model.set_grad(True)
    return float(loss), float(acc)


if __name__ == "__main__":
    b_size = 128
    device = torch.device("cuda:0")
    model_path = "weights/cifar10.pth"

    train_data = torchvision.datasets.CIFAR10(
        root="./data", train=True, transform=torchvision.transforms.ToTensor()
    )
    train_loader = DataLoader(
        dataset=train_data, batch_size=b_size, shuffle=True, pin_memory=True
    )
    test_data = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=torchvision.transforms.ToTensor()
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=b_size, shuffle=True, pin_memory=True
    )

    model = ResNet_CIFAR10()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()

    att_method = None
    # att_method = FGSM(model, eps=8 / 255)
    # att_method = PGD(model, eps=8 / 255, alpha=2 / 255, steps=20)

    test_loss, test_acc = evaluate_cifar10(model, test_loader, attack=att_method)
    print(test_loss, test_acc)