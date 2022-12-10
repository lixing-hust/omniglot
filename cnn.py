import main
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
N=5
s=5
q=5
train_data,test_data=main.data_init(N,s,q)

BATCH_SIZE=[1,2,5,10,15,25]
EPOCHS=40 # 总共训练批次
LEARNING_RATE=[0.1,0.05,0.02,0.01,0.005,0.002,0.001]
x_epoch=[5,10,15,20,25,30,35,40]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多



class MyDataset(Dataset):
    def __init__(self,data):
        self.df=data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df[idx]

train_ds = MyDataset(train_data)
test_ds = MyDataset(test_data)

transformer=transforms.Compose([transforms.ToTensor(), transforms.Resize((28,28))])
for i in range(len(train_ds)):
    train_ds[i][0]=transformer(train_ds[i][0])
    test_ds[i][0]=transformer(test_ds[i][0])



class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10, 5) # 输入通道数1，输出通道数10，核的大小5
        self.conv2 = nn.Conv2d(10, 20, 3) # 输入通道数10，输出通道数20，核的大小3
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(20*10*10, 500) # 输入通道数是2000，输出通道数是500
        self.fc2 = nn.Linear(500, N) # 输入通道数是500，输出通道数是10，即10分类
    def forward(self,x):
        in_size = x.size(0) # 在本例中in_size=是BATCH_SIZE的值。输入的x可以看成是512*1*28*28的张量。
        out = self.conv1(x) # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = F.relu(out) # batch*10*24*24（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2) # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = self.conv2(out) # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = F.relu(out) # batch*20*10*10
        out = out.view(in_size, -1) # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out = self.fc1(out) # batch*2000 -> batch*500
        out = F.relu(out) # batch*500
        out = self.fc2(out) # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1) # 计算log(softmax(x))
        return out

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if(batch_idx+1)%2 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)



def epoch_acc(EPOCHS):

    train_dl = DataLoader(train_ds, batch_size=5, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=5, shuffle=True, num_workers=0)
    Accuracy = []
    model = ConvNet().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_dl, optimizer, epoch)
        accuracy = test(model, DEVICE, test_dl)
        if((epoch+1)%5==0):
            Accuracy.append(accuracy)

    plt.plot(x_epoch,Accuracy,'bo--',alpha=0.5,linewidth=1,label='epochs')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()

def lr_acc(LEARNING_RATE):
    train_dl = DataLoader(train_ds, batch_size=5, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=5, shuffle=True, num_workers=0)
    Accuracy = []
    x_aixs=LEARNING_RATE

    for lr in LEARNING_RATE:
        model = ConvNet().to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        for epoch in range(1, 20 + 1):
            train(model, DEVICE, train_dl, optimizer, epoch)
            accuracy = test(model, DEVICE, test_dl)
            if ((epoch + 1) % 20 == 0):
                Accuracy.append(accuracy)
    plt.plot(x_aixs, Accuracy, 'bo--', alpha=0.5, linewidth=1, label='learning_rata')
    plt.legend()
    plt.xlabel('learning_rata')
    plt.ylabel('accuracy')
    plt.show()

def batchSize_acc(BATCH_SIZE):

    Accuracy = []
    x_aixs=BATCH_SIZE

    for bs in BATCH_SIZE:

        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
        test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True, num_workers=0)
        model = ConvNet().to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for epoch in range(1, 20 + 1):
            train(model, DEVICE, train_dl, optimizer, epoch)
            accuracy = test(model, DEVICE, test_dl)
            if ((epoch + 1) % 20 == 0):
                Accuracy.append(accuracy)
    plt.plot(x_aixs, Accuracy, 'bo--', alpha=0.5, linewidth=1, label='batch_size')
    plt.legend()
    plt.xlabel('batch_size')
    plt.ylabel('accuracy')
    plt.show()

# batchSize_acc(BATCH_SIZE)
# lr_acc(LEARNING_RATE)
epoch_acc(40)