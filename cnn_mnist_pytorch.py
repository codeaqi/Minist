'''
基于Pytorch的卷积神经网络MNIST手写数字识别
改进：BatchNorm + Dropout + StepLR + 数据增强
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import time

font = {'family': 'MicroSoft YaHei', 'weight': 'bold', 'size': '10'}
matplotlib.rc("font", **font)

# 超参数
learning_rate = 0.001
batch_size = 100
epochs_num = 20
download = True
use_gpu = 1
is_train = 1
show_pic = 0

# 数据增强
train_transform = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
])

train_dataset = datasets.MNIST(root='.', train=True,
                               transform=train_transform, download=download)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

# 改进的卷积神经网络：BatchNorm + Dropout
class MNIST_Network(nn.Module):
    def __init__(self):
        super(MNIST_Network, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)              # 批归一化
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)              # 批归一化
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.fc3 = nn.Linear(7*7*64, 1024)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)             # Dropout防过拟合

        self.fc4 = nn.Linear(1024, 10)

    def forward(self, input1):
        x = self.pool1(self.relu1(self.bn1(self.conv1(input1))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size()[0], -1)
        x = self.dropout(self.relu3(self.fc3(x)))
        x = self.fc4(x)
        return x

net = MNIST_Network()
if use_gpu:
    net = net.cuda()

if is_train:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 学习率调度

    counter = []
    loss_history = []
    correct_history = []
    correct_cnt = 0
    counter_temp = 0
    record_interval = 100

    for epoch in range(0, epochs_num):
        for i, data in enumerate(train_loader, 0):
            img, label = data
            if use_gpu:
                img, label = img.cuda(), label.cuda()

            optimizer.zero_grad()
            output = net(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            _, predict = torch.max(output, 1)
            correct_cnt += (predict == label).sum()

            if i % record_interval == record_interval - 1:
                counter_temp += record_interval * batch_size
                counter.append(counter_temp)
                loss_history.append(loss.item())
                correct_history.append(correct_cnt.float().item() / (record_interval * batch_size))
                correct_cnt = 0

        scheduler.step()  # 更新学习率
        print("迭代次数 {} 当前损失函数值 {} 学习率 {}".format(
            epoch, loss.item(), scheduler.get_last_lr()[0]))

    torch.save(net.state_dict(), '.\modelpara.pth')

if use_gpu:
    net.load_state_dict(torch.load('.\modelpara.pth'))
else:
    net.load_state_dict(torch.load('.\modelpara.pth', map_location='cpu'))

test_dataset = datasets.MNIST(root='.', train=False,
                              transform=transforms.ToTensor(), download=download)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

start = time.time()
correct = 0
for i, data in enumerate(test_loader, 0):
    img, label = data
    if use_gpu:
        img, label = img.cuda(), label.cuda()
    output = net(img)
    _, predict = torch.max(output, 1)
    correct += (predict == label).sum()
end = time.time()

print('MNIST测试集识别准确率= {:.2f}'.format(correct.cpu().numpy() / len(test_dataset) * 100) + '%')
print('10000张识别时间= {:.3f}'.format(end - start) + ' s')
