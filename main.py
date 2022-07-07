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
#畳み込みによって最終的にどのようなサイズになっているかを、
#実際にダミーデータを入れてみて確認する。
test_input = torch.ones(1, 1, 28, 28)
conv_output_size = conv_net(test_input).size()[-1]

#2層のMLP
mlp = nn.Sequential(
    nn.Linear(conv_output_size, 200),
    nn.ReLU(),
    nn.BatchNorm1d(200),
    nn.Dropout(0.25),
    nn.Linear(200, 10)
)

#最終的なCNN
net = nn.Sequential(
    conv_net,
    mlp
)

# 評価のヘルパー関数
def eval_net(net, data_loader, device="cpu"):
    # DropoutやBatchNormを無効化
    net.eval()
    ys = []
    ypreds = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        
        #確率が最大のクラスを予測
        #ここではforward(推論)の計算だけなので自動微分に
        #必要な処理はoffにして余計な計算を省く
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)
        
    #ミニバッチごとの予測結果などを1つにまとめる
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    #予測結果を計算
    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()

# 訓練のヘルパー関数
def train_net(net, train_loader, test_loader, optimizer_cls=optim.Adam, loss_fn=nn.CrossEntropyLoss(), n_iter=10, device="cpu"):
    train_losses = []
    train_acc = []
    val_acc = []
    optimizer = optimizer_cls(net.parameter())
    for epoch in range(n_iter):
        running_loss = 0.0
        #ネットワークを訓練モードにする
        net.train()
        n = 0
        n_acc = 0
        #非常に時間がかかるのでtqdmを使用してプログレスバーを出す
        for i , (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)
            h = net(xx)
            loss = loss_fn(h, yy)
            optimizer.zero_grad()
            loss.backward()
            ootimizer.step()
            running_loss += loss.item()
            n += len(xx)
            _, y_pred = h.max(1)
            n_acc += (yy == y_pred).float().sum().item()
        train_losses.append(running_loss / i)
        #訓練データの予測精度
        train_acc.append(n_acc / n)
        #検証データの予測精度
        val_acc.append(eval_net(net, test_loader, device))
        #このepochでの結果を表示
        print(epoch, train_losses[-1], train_acc[-1], val_acc[-1], flush=True)
