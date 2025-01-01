# -*- coding: utf-8 -*-
# @Time         : 2024/12/25 02:41
# @Author       : Song Zilong
# @Software     : PyCharm
# @Description  : LeNet实现
from torch import nn
import torch.nn.functional as f

class LeNet(nn.Module):
    def __init__(self, input_channels=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.batch_norm = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2)
        x = self.batch_norm(x)
        feature = x.view(x.size(0), -1)
        x = f.relu(self.fc1(feature))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x, feature
