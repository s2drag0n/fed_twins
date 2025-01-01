# -*- coding: utf-8 -*-
# @Time         : 2024/12/26 16:41
# @Author       : Song Zilong
# @Software     : PyCharm
# @Description  : 模型预测
import numpy as np
import torch
import torch.nn.functional as f


def get_output(loader, net, args, latent=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if not latent:
                outputs, _ = net(images)
                outputs = f.softmax(outputs, dim=1)
            else:
                outputs, _ = net(images, True)
            loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole
