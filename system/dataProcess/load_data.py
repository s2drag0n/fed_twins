# -*- coding: utf-8 -*-
# @Time         : 2024/12/25 02:08
# @Author       : Song Zilong
# @Software     : PyCharm
# @Description  : 获取分配完毕并添加过噪声的数据

import numpy as np

from system.dataProcess.add_noise import add_noise
from system.dataProcess.dataset import get_dataset


def load_data_with_noisy_label(args, dataset_train, dataset_test, dict_users):
    # ---------------------Add Noise ---------------------------
    if args.dataset == 'clothing1m':
        y_train_noisy = np.array(dataset_train.targets)
        dataset_train.targets = y_train_noisy
        gamma_s = None
        noisy_sample_idx = None
        y_train = y_train_noisy
        print(f"len(dataset_train)= {len(dataset_train)}, len(dataset_test) = {len(dataset_test)}")
    else:
        y_train = np.array(dataset_train.targets)
        y_train_noisy, gamma_s, real_noise_level, noisy_sample_idx = add_noise(args, y_train, dict_users)
        dataset_train.targets = y_train_noisy
    # dataset_train : 训练数据集
    # dataset_test  : 测试数据集
    # dict_users : 分配每个客户端上的样本索引列表
    # y_train : 训练集的真实样本标签
    # gamma_s : 客户端是否有噪声
    # noisy_sample_idx : 有噪声的样本索引
    return dataset_train, dataset_test, dict_users, y_train, gamma_s, noisy_sample_idx
