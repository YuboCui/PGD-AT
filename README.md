# PGD-AT

[中文README](https://github.com/YuboCui/PGD-AT/blob/main/README_zh.md)

PGD Adversarial training on MNIST and CIFAR-10. A PyTorch implementation of "[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)" ([Madry *et* *al*, ICLR’18](https://openreview.net/forum?id=rJzIBfZAb)). The author's code ([MNIST](https://github.com/MadryLab/mnist_challenge) and [CIFAR-10](https://github.com/MadryLab/cifar10_challenge)) is based on Tensorflow implementation, and we use PyTorch to basically restore the network structure, training methods, etc. in the original paper code. Meanwhile, we mainly focus on the adversarial training process and have utilized the PGD attack implementation in [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch).

The code includes two folders, MNIST and CIFAR10. You can enter the corresponding folder to execute the `.py` file, and the results are shown in the table below.

## MNIST

`data/`: MNIST dataset, default to empty, can be downloaded by setting the parameter `download=True`, using `torchvision.datasets.MNIST`

`weights/`: To save the model weights file, two options are provided: `mnist.pth` (normal training, accuracy 99.24%) and `mnist_adv.pth` (adversarial training)

`resnet_cifar10.py`: CNN model for MNIST classification

`train.py`: Train a normal MNIST classification model / Use PGD adversarial training to obtain a robust model

`eval.py`: Evalute the accuracy of the model under the original test set or against FGSM/PGD attack

Performance of the adversarially trained network against different adversaries for ε=0.3

| Method          | mnist_adv.pth | Madry _et_ _al_ ICLR’18 |
| --------------- | ------------- | ----------------------- |
| Natural         | **99.02%**    | 98.8%                   |
| FGSM            | **96.56%**    | 95.6%                   |
| PGD (40 steps)  | **94.56%**    | 93.2%                   |
| PGD (100 steps) | **99.02%**    | 91.8%                   |

Directly using PGD attack with step=40, ε=0.3 for adversarial training may result in non convergence. You can first train with smaller values of ε and step, and then gradually increase the values of ε and step. This is consistent with the **Warmup w.r.t. epsilon** mentioned in [Pang _et_ _al_ ICLR’21](https://openreview.net/forum?id=Xb8xvrtB8Ce).

## CIFAR-10

`data/`: CIFAR-10 dataset, default to empty, can be downloaded by setting the parameter `download=True`, using `torchvision.datasets.CIFAR10`

`weights/`: To save the model weights file, two options are provided: `cifar10.pth` (normal training, accuracy _90.86%_ but 95.2% in paper) and `cifar10_adv.pth` (adversarial training)

`cnn_mnist.py`: ResNet model for CIFAR-10 classification. The original code used `tf.image.per_image_standardization` to process images, while we use `torchvision.transforms.v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))`. We also implement a `PerImageStandardization` class, but not use it during the train/eval process.

`train.py`: Train a normal CIFAR-10 classification model

`train_adv.py`: Use PGD adversarial training to obtain a robust model

`eval.py`: Evalute the accuracy of the model under the original test set or against FGSM/PGD attack

Performance of the adversarially trained network against different adversaries for ε=8

| Method         | cifar10_adv.pth | Madry _et_ _al_ ICLR’18 |
| -------------- | --------------- | ----------------------- |
| Natural        | _82.87%_        | 87.3%                   |
| FGSM           | **56.25%**      | 56.1%                   |
| PGD (7 steps)  | **52.89%**      | 50.0%                   |
| PGD (20 steps) | **50.37%**      | 45.8%                   |

Our model has good robustness against adversarial samples. However, whether it is normal training or adversarial training, the accuracy of our model on the original test set is always low, which is an unresolved problem.
