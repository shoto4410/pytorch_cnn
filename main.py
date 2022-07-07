import torch
from torch import nn, optim
from torch.utils.data import (Dataset,DataLoader,TensorDataset)
import tqdm
from torchvision.datasets import FashionMNIST
from torchvision import transforms

fashion_mnist_train = FashionMNIST("data/FashionMNIST",train=True, download=True, transform=transforms.ToTensor())
fashion_mnist_test = FashionMNIST("data/FashionMNIST",train=False, download=True, transform=transforms.ToTensor())

batch_size = 128
train_loader = DataLoader(fashion_mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(fashion_mnist_test, batch_size=batch_size, shuffle=True)

