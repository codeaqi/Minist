import torch.nn as nn


class MNISTNetwork(nn.Module):
    def __init__(self, in_chans=1, mid_chans=32, num_blocks=2, num_classes=10, drop_rate=0.0):
        super(MNISTNetwork, self).__init__()
        if num_blocks != 2:
            raise ValueError("MNISTNetwork currently supports num_blocks=2.")

        self.conv1 = nn.Conv2d(in_chans, mid_chans, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(mid_chans, mid_chans * 2, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.fc3 = nn.Linear(7 * 7 * mid_chans * 2, 1024)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.fc4 = nn.Linear(1024, num_classes)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x
