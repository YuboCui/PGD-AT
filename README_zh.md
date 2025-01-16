# PGD-AT（PGD对抗训练）

[ICLR 2018]([Towards Deep Learning Models Resistant to Adversarial Attacks | OpenReview](https://openreview.net/forum?id=rJzIBfZAb))论文“[Towards Deep Learning Models Resistant to Adversarial Attacks]([[1706.06083\] Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083))”的简洁PyTorch实现，在MNIST和CIFAR-10数据集上通过PGD攻击方法进行对抗训练。原作者的代码（[MNIST]([MadryLab/mnist_challenge: A challenge to explore adversarial robustness of neural networks on MNIST.](https://github.com/MadryLab/mnist_challenge))和[CIFAR-10]([MadryLab/cifar10_challenge: A challenge to explore adversarial robustness of neural networks on CIFAR10.](https://github.com/MadryLab/cifar10_challenge))）基于Tensorflow实现，我们使用PyTorch框架，基本还原了原论文代码中不同分类器的网络结构、训练方法等。同时，我们主要关心对抗训练过程，调用了[torchattacks]([Harry24k/adversarial-attacks-pytorch: PyTorch implementation of adversarial attacks [torchattacks\]](https://github.com/Harry24k/adversarial-attacks-pytorch))库中的PGD对抗攻击实现。

代码包括两个文件夹，MNIST和CIFAR10。可进入对应文件夹内执行`.py`文件，复现效果见下方表格。

## MNIST

`data/`: 包含MNIST数据集，默认为空，可通过设置`torchvision.datasets.MNIST`的参数`download=True`下载数据集

`weights/`: 用于保存模型权重文件，这里提供了`mnist.pth`（正常训练，准确率99.24%）和`mnist_adv.pth`（对抗训练）两种

`cnn_mnist.py`: 用于MNIST分类的CNN模型定义

`train.py`: 训练正常的MNIST分类模型/使用PGD对抗训练得到鲁棒模型

`eval.py`: 验证模型在原始测试集/FGSM攻击/PGD攻击下的准确率

Performance of the adversarially trained network against different adversaries for ε=0.3

| Method          | mnist_adv.pth | Madry _et_ _al_ ICLR’18 |
| --------------- | ------------- | ----------------------- |
| Natural         | **99.02%**    | 98.8%                   |
| FGSM            | **96.56%**    | 95.6%                   |
| PGD (40 steps)  | **94.56%**    | 93.2%                   |
| PGD (100 steps) | **99.02%**    | 91.8%                   |

直接采用ε=0.3, step=40的PGD攻击进行对抗训练可能导致不收敛，可以先用较小的ε和step训练，然后逐步增加ε和step的值。这与[Pang _et_ _al_ ICLR’21](https://openreview.net/forum?id=Xb8xvrtB8Ce)提到的**Warmup w.r.t. epsilon**相符。

## CIFAR-10

`data/`: 包含CIFAR-10数据集，默认为空，可通过设置`torchvision.datasets.CIFAR10`的参数`download=True`下载数据集

`weights/`: 用于保存模型权重文件，这里提供了`cifar10.pth`（正常训练，准确率*90.86%*，原作者达到了95.2%）和`cifar10_adv.pth`（对抗训练）两种

`cnn_mnist.py`: 用于CIFAR-10分类的ResNet模型定义。原作者使用了`tf.image.per_image_standardization`处理图像，而我们采用了`torchvision.transforms.v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))`。我们也在代码中实现了`PerImageStandardization`类，但并没有在训练和测试中使用。

`train.py`: 训练正常的CIFAR-10分类模型/使用PGD对抗训练得到鲁棒模型

`eval.py`: 验证模型在原始测试集/FGSM攻击/PGD攻击下的准确率

Performance of the adversarially trained network against different adversaries for ε=8

| Method         | cifar10_adv.pth | Madry _et_ _al_ ICLR’18 |
| -------------- | --------------- | ----------------------- |
| Natural        | _82.87%_        | 87.3%                   |
| FGSM           | **56.25%**      | 56.1%                   |
| PGD (7 steps)  | **52.89%**      | 50.0%                   |
| PGD (20 steps) | **50.37%**      | 45.8%                   |

面对对抗样本时我们的模型鲁棒性较好。但无论是正常训练还是对抗训练，我们的模型在原始测试集样本上的准确率始终较低，这是一个待解决的问题。
