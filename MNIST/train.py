import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from cnn_mnist import CNN_MNIST
from tqdm import tqdm
from eval import evaluate_mnist
from copy import deepcopy
from torchattacks import PGD


def train(
    model,
    train_loader,
    val_loader,
    optim=None,
    scheduler=None,
    pretrain=None,
    max_epoch=200,
    early_stop=10,
    attack=None,
    save_path="./model.pth",
):
    device = next(model.parameters()).device
    if optim == None:
        optim = torch.optim.Adam(model.parameters())
    if pretrain:
        model.load_state_dict(torch.load(pretrain, weights_only=True))
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    best_weight = None
    best_val_loss = 1000.0
    num = 0
    for i in range(max_epoch):
        model.train()
        for _, (x, y) in enumerate(tqdm(train_loader)):
            optim.zero_grad()
            if attack:
                model.eval()
                model.set_grad(False)
                x_adv = attack(x, y)
                model.set_grad(True)
                model.train()
                y = y.to(device)
                y_hat = model(x_adv)
            else:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optim.step()
        if scheduler:
            scheduler.step()
        model.eval()
        val_loss, val_acc = evaluate_mnist(model, val_loader, attack)
        print("eopch " + str(i) + ":", val_loss, val_acc)
        if early_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weight = deepcopy(model.state_dict())
                num = 0
            else:
                num += 1
            if num >= early_stop:
                print("early stop!")
                model.load_state_dict(best_weight)
                val_loss, val_acc = evaluate_mnist(model, val_loader, attack)
                print(val_loss, val_acc)
                torch.save(model.cpu().state_dict(), save_path)
                return
    torch.save(model.cpu().state_dict(), save_path)


if __name__ == "__main__":
    b_size = 128
    device = torch.device("cuda:1")

    train_and_val_data = MNIST(
        root="./data", train=True, transform=transforms.ToTensor()
    )
    train_data, val_data = random_split(train_and_val_data, [50000, 10000])
    train_loader = DataLoader(
        dataset=train_data, batch_size=b_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_data, batch_size=b_size, shuffle=True, pin_memory=True
    )

    model = CNN_MNIST().to(device)
    # train a CNN with original MNIST dataset
    train(model, train_loader, val_loader, save_path="weights/mnist.pth")

    # adversarial training
    # attack = PGD(model, 0.3, 0.01, 40)
    # optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    # train(
    #     model,
    #     train_loader,
    #     val_loader,
    #     optim,
    #     save_path="weights/mnist_adv.pth",
    #     attack=attack,
    # )
