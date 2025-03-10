# -*- coding: utf-8 -*-
# @Time         : 2024/12/25 03:16
# @Author       : Song Zilong
# @Software     : PyCharm
# @Description  : loss计算方法

from typing import Optional
from torch import Tensor
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as f
from . optimizer import filter_noisy_data, f_beta
from torch.nn import CrossEntropyLoss


class CORESLoss(CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduction: str = 'mean', label_smoothing: float = 0.0):
        super().__init__(weight, size_average, reduction=reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, input, target, beta=0, noise_prior=None):
        loss = f.cross_entropy(input, target, reduction=self.reduction)
        loss_ = -torch.log(f.softmax(input, dim=1) + 1e-8)
        noise_prior = noise_prior
        if noise_prior is None:
            loss = loss - beta * torch.mean(loss_, 1)  # CORESLoss
        else:
            loss = loss - beta * torch.sum(torch.mul(noise_prior, loss_), 1)
        loss_ = loss
        return loss_


class FedTwinCRLoss(CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduction=reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction


    def forward(self, target, input_p=None, input_g=None, rounds=0, epoch=0, args=None, noise_prior=None):
        cores_loss = CORESLoss(reduction='none')

        beta = f_beta(rounds * args.local_ep + epoch, args)
        ind_noise_p = []
        ind_noise_g = []
        if rounds <= args.begin_sel:  # 如果在前30epoch集中式，对应联邦应该是30/local_epoch
            loss_p_update = cores_loss(input_p, target, beta, noise_prior)
            loss_g_update = cores_loss(input_g, target, beta, noise_prior)

            ind_g_update = Variable(torch.from_numpy(np.ones(len(loss_p_update)))).bool()
        else:
            ind_p_update = filter_noisy_data(input_p, target)
            ind_g_update = filter_noisy_data(input_g, target)

            ind_noise_p = torch.ByteTensor([not x for x in ind_p_update]).bool()
            ind_noise_g = torch.ByteTensor([not x for x in ind_g_update]).bool()

            # 根据阈值筛选出比较可靠的噪声样本并使用预测值作为样本标签，使用label——smoothing进行loss计算
            target_g = []
            target_p = []
            for noise_output_g in input_p[ind_noise_g]:
                # if torch.max(noise_output_g) > 0.5:
                target_g.append(torch.argmax(noise_output_g))
            for noise_output_p in input_g[ind_noise_p]:
                target_p.append(torch.argmax(noise_output_p))

            target_g = torch.LongTensor(target_g).to(args.device)
            target_p = torch.LongTensor(target_p).to(args.device)

            loss_g_update_noise = cores_loss(input_g[ind_noise_p], target_p, beta, noise_prior)
            loss_p_update_noise = cores_loss(input_p[ind_noise_g], target_g, beta, noise_prior)

            # 关键
            loss_p_update = cores_loss(input_p[ind_g_update], target[ind_g_update], beta, noise_prior)
            loss_g_update = cores_loss(input_g[ind_p_update], target[ind_p_update], beta, noise_prior)

        loss_batch_p = len(loss_p_update.data.cpu().numpy())  # number of batch loss1
        loss_batch_g = len(loss_g_update.data.cpu().numpy())  # number of batch loss1

        if loss_batch_p == 0:
            loss_p = cores_loss(input_p, target, beta, noise_prior)
            loss_p = torch.mean(loss_p) / 100000000
        else:
            loss_p = torch.sum(loss_p_update) / loss_batch_p
            # if(len(ind_noise_p)) != 0.0:
            #     loss_p = loss_p + torch.sum(loss_p_update_noise) / (len(ind_noise_p))
        if loss_batch_g == 0:
            loss_g = cores_loss(input_g, target, beta, noise_prior)
            loss_g = torch.mean(loss_g) / 100000000
        else:
            loss_g = torch.sum(loss_g_update) / loss_batch_g
            # if (len(ind_noise_g)) != 0.0:
            #     loss_g = loss_g + torch.sum(loss_g_update_noise) / (len(ind_noise_g))
        return loss_p, loss_g, loss_batch_p, loss_batch_g, ind_g_update, ind_noise_p, ind_noise_g
