# -*- coding: utf-8 -*-
# @Time         : 2024/12/26 02:18
# @Author       : Song Zilong
# @Software     : PyCharm
# @Description  : 测试相关
import torch
from torch.utils import data


def global_test(net, test_dataset, args):
    net.eval()
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs, _ = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.eq(predicted, labels).sum().item()

    acc = (correct / total) * 100
    return acc
