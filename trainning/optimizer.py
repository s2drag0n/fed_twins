# -*- coding: utf-8 -*-
# @Time         : 2024/12/25 03:18
# @Author       : Song Zilong
# @Software     : PyCharm
# @Description  : optimizer实现

import copy

import numpy as np
import torch
import torch.nn.functional as f
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Optimizer


class MyOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(MyOptimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        weight_update = copy.deepcopy(local_weight_updated)
        group = self.param_groups[0]
        for p, localweight in zip(group['params'], weight_update):
            p.data = p.data - group['lr'] * (
                    p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)
        return group['params'], loss

    def update_param(self, local_weight_updated):
        weight_update = copy.deepcopy(local_weight_updated)
        group = self.param_groups[0]
        for p, localweight in zip(group['params'], weight_update):
            p.data = localweight.data
        return group['params']


def filter_noisy_data(input: Tensor, target: Tensor):
    loss = f.cross_entropy(input, target, reduction='none')
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)  # number of batch
    loss_v = np.zeros(num_batch)  # selected tag
    loss_ = -torch.log(f.softmax(input, dim=1) + 1e-8)
    # sel metric
    loss_sel = loss - torch.mean(loss_, 1)
    loss_div_numpy = loss_sel.data.cpu().numpy()
    for i in range(len(loss_numpy)):
        if loss_div_numpy[i] <= 0:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)

    return Variable(torch.from_numpy(loss_v)).bool()


def f_beta(epoch, args):
    beta1 = np.linspace(0.0, 0.0, num=args.local_ep * 2)
    beta2 = np.linspace(0.0, args.max_beta, num=int(args.local_ep * args.begin_sel))
    beta3 = np.linspace(args.max_beta, args.max_beta, num=args.rounds2 * args.local_ep)
    beta = np.concatenate((beta1, beta2, beta3), axis=0)
    return beta[epoch]


def adjust_learning_rate(epoch, args, optimizer=None):
    # 需要后期再确认一次是否是args.begin_sel，之前是10
    # if args.dataset == 'cifar10':
    alpha_plan = [[args.plr] * int(args.local_ep * args.rounds2 / 2) + [args.plr] * args.local_ep * args.rounds2,
                  [args.lr] * int(args.local_ep * args.rounds2 / 2) + [args.lr] * args.local_ep * args.rounds2]
    if optimizer == 'plr':
        lr = alpha_plan[0][epoch] / (1 + f_beta(epoch, args))
        return lr
    elif optimizer == 'lr':
        plr = alpha_plan[1][epoch] / (1 + f_beta(epoch, args))
        return plr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = alpha_plan[0][epoch] / (1 + f_beta(epoch, args))
