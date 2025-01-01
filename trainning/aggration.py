# -*- coding: utf-8 -*-
# @Time         : 2024/12/26 02:16
# @Author       : Song Zilong
# @Software     : PyCharm
# @Description  : 用于全局聚合
import copy

import torch


def personalized_aggregation(net_glob, w, n_bar, gamma):
    w_agg = copy.deepcopy(w[0])
    for k in w_agg.keys():
        w_agg[k] = w_agg[k] * n_bar[0]
        for i in range(1, len(w)):
            w_agg[k] += w[i][k] * n_bar[i]
        if sum(n_bar) == 0:
            w_agg[k] = gamma * torch.div(w_agg[k], sum(n_bar) + 100000000) + (1 - gamma) * net_glob[k]
        else:
            w_agg[k] = gamma * torch.div(w_agg[k], sum(n_bar)) + (1 - gamma) * net_glob[k]
    return w_agg
