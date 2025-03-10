# -*- coding: utf-8 -*-
# @Time         : 2024/12/25 02:08
# @Author       : Song Zilong
# @Software     : PyCharm
# @Description  : 对样本进行采样，根据不同的模式将样本分配给不同客户端

from typing import Any

import numpy as np


def iid_sampling(n_train, num_users, seed):
    np.random.seed(seed)
    num_items = int(n_train / num_users)
    dict_users, all_idx = {}, [i for i in range(n_train)]  # initial user and index for whole dataset
    for i in range(num_users):
        dict_users[i] = set(
            np.random.choice(all_idx, num_items, replace=False))  # 'replace=False' make sure that there is no repeat
        all_idx = list(set(all_idx) - dict_users[i])
    return dict_users


def non_iid_dirichlet_sampling(y_train, num_classes, prob, num_users, seed, alpha_dirichlet=100):
    np.random.seed(seed)
    phi = np.random.binomial(1, prob, size=(num_users, num_classes))  # indicate the classes chosen by each client
    n_classes_per_client = np.sum(phi, axis=1)
    while np.min(n_classes_per_client) == 0:
        invalid_idx = np.where(n_classes_per_client == 0)[0]
        phi[invalid_idx] = np.random.binomial(1, prob, size=(len(invalid_idx), num_classes))
        n_classes_per_client = np.sum(phi, axis=1)
    psi = [list(np.where(phi[:, j] == 1)[0]) for j in range(num_classes)]  # indicate the clients that choose each class

    # 确保每个类别至少被一个客户端选择
    n_clients_per_class = np.sum(phi, axis=0)
    while np.min(n_clients_per_class) == 0:
        invalid_classes = np.where(n_clients_per_class == 0)[0]
        for class_j in invalid_classes:
            client_k = np.random.randint(num_users)
            phi[client_k, class_j] = 1
        n_clients_per_class = np.sum(phi, axis=0)

    psi = [list(np.where(phi[:, j] == 1)[0]) for j in range(num_classes)]  # 每个类别对应的客户端列表

    num_clients_per_class = np.array([len(x) for x in psi])
    dict_users: dict[Any, set[Any]] = {}
    for class_i in range(num_classes):
        all_idx = np.where(y_train == class_i)[0]
        p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])
        assignment = np.random.choice(psi[class_i], size=len(all_idx), p=p_dirichlet.tolist())

        for client_k in psi[class_i]:
            if client_k in dict_users:
                dict_users[client_k] = set(dict_users[client_k] | set(all_idx[(assignment == client_k)]))
            else:
                dict_users[client_k] = set(all_idx[(assignment == client_k)])
    return dict_users
