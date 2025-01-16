import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from cnn_mnist import CNN_MNIST
from tqdm import tqdm
from torchattacks import PGD, FGSM


def evaluate_mnist(model, dataloader, attack=None):
    grad = next(model.parameters()).requires_grad
    if grad:
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
                # x_adv = (x_adv * 255).int()
                # x_adv = x_adv.float() / 255
                y = y.to(device)
                y_hat = model(x_adv)
                loss += loss_fn(y_hat, y)
                acc += (y_hat.argmax(dim=1) == y).sum()
    loss /= len(dataloader.dataset)
    acc /= len(dataloader.dataset)
    if grad:
        model.set_grad(True)
    return float(loss), float(acc)


if __name__ == "__main__":
    b_size = 128
    device = torch.device("cuda:0")
    model_path = "weights/mnist_adv.pth"

    test_data = MNIST(root="./data", train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=b_size, shuffle=True)

    model = CNN_MNIST()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()

    # att_method = None
    # att_method = FGSM(model, eps=0.3)
    att_method = PGD(model, eps=0.3, alpha=0.01, steps=100)

    test_loss, test_acc = evaluate_mnist(model, test_loader, attack=att_method)
    print(test_loss, test_acc)
