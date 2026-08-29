from torchvision.datasets import CIFAR10

trainset = CIFAR10(
    root="./data",
    train=True,
    download=True
)
