# -*- coding: utf-8 -*-
# @Time         : 2024/12/25 02:36
# @Author       : Song Zilong
# @Software     : PyCharm
# @Description  : 为实验初始化模型

from system.model import ResNet
from system.model.LeNet import LeNet


def build_model(args):
    if args.model == 'lenet':
        if args.dataset == 'mnist':
            net = LeNet(1).to(args.device)
        else:
            net = LeNet().to(args.device)
    elif args.model == 'resnet18':
        net = ResNet.resnet18(args.num_classes)
        net.to(args.device)

    elif args.model == 'resnet34':
        net = ResNet.resnet18(args.num_classes)
        net.to(args.device)

    elif args.model == 'resnet50':
        net = ResNet.resnet50(args.num_classes)
        net.to(args.device)
    else:
        exit('Error: unrecognized model')

    return net

