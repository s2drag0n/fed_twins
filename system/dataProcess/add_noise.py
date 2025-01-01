# -*- coding: utf-8 -*-
# @Time         : 2024/12/25 02:08
# @Author       : Song Zilong
# @Software     : PyCharm
# @Description  : 为样本添加噪声

import copy

import numpy as np


def add_noise(args, y_train, dict_users):
    np.random.seed(args.seed)

    gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial

    y_train_noisy = copy.deepcopy(y_train)

    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, 10, len(noisy_idx))
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio.item()))
        real_noise_level[i] = noise_ratio
    # By comparing the labels of the dataset before and after adding noise
    # we can determine which samples have been affected by the noise
    noisy_samples_idx = np.where(y_train != y_train_noisy)[0]
    return y_train_noisy, gamma_s, real_noise_level, noisy_samples_idx
