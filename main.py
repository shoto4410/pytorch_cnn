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

#(N,C,H,W)形式のTensorを(N, C*H*W)に引伸ばす層
#畳み込み層の出力をMLPに渡す際に必要
class FlattenLayer(nn.Module):
    def forward(self, x):
        sizes = x.size()
        return x.view(sizes[0], -1)
    
#2dは画像形式用
conv_net = nn.Sequential(
    nn.Conv2d(1, 32, 5),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Dropout2d(0.25),
    nn.Conv2d(32, 64, 5),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.Dropout2d(0.25),
    FlattenLayer()
)