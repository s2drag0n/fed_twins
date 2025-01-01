# -*- coding: utf-8 -*-
# @Time         : 2024/12/25 02:08
# @Author       : Song Zilong
# @Software     : PyCharm
# @Description  : main文件
import random

import numpy as np
import torch

from utils.arg_paser import args_parser
from algorithms.fed_twins import fed_twins


def run(arguments):
    # seed
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    np.random.seed(arguments.seed)
    random.seed(arguments.seed)

    if torch.cuda.is_available() and arguments.gpu != -1:
        # 如果是 NVIDIA GPU，使用 CUDA
        arguments.device = torch.device('cuda:{}'.format(arguments.gpu))
        print("using nvidia GPU......")
    elif torch.backends.mps.is_available():
        # 如果是苹果 M 系列芯片，使用 MPS
        arguments.device = torch.device('mps')
        print("using apple M silicon......")
    else:
        # 如果没有可用的 GPU，使用 CPU
        arguments.device = torch.device('cpu')
        print("using cpu......")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if arguments.dataset == "mnist":
        arguments.plr = arguments.lr / 2
    if arguments.dataset == 'cifar100':
        arguments.max_beta = 2.0
    elif arguments.dataset == 'cifar10' or 'mnist':
        arguments.max_beta = 2.0
    elif arguments.dataset == 'clothing1m':
        arguments.max_beta = 2.8
    for x in vars(arguments).items():
        print(x)
    # run Algorithm
    eval(arguments.algorithm)(arguments)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # parse args
    args = args_parser()
    # float for some parameters
    args.lr = float(args.lr)
    args.plr = float(args.plr)
    args.frac2 = float(args.frac2)
    args.begin_sel = int(args.begin_sel)
    run(args)
