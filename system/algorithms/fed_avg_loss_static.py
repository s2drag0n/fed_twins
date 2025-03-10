import copy
import time

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.utils.data import Subset

from system.correction.correction import lid_term
from system.dataProcess.dataset import get_dataset
from system.dataProcess.load_data import load_data_with_noisy_label
from system.model.bulid_model import build_model
from system.trainning.aggration import personalized_aggregation, aggregation
from system.trainning.fscore import cal_f_score
from system.trainning.get_output import get_output
from system.trainning.local_training import FedTwinLocalUpdate, FedAVGLocalUpdate
from system.trainning.test import global_test


def fed_avg_loss_static(args):
    dataset_train, dataset_test, dict_users = get_dataset(args)
    dataset_train, dataset_test, dict_users, y_train, gamma_s, noisy_sample_idx = load_data_with_noisy_label(args, dataset_train, dataset_test, dict_users)
    start = time.time()
    net_glob = build_model(args)

    m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    prob = [1 / args.num_users] * args.num_users


    print(f'noisy ratio = {len(noisy_sample_idx) / len(dataset_train)}')

    for rnd in range(args.rounds2):
        w_locals, p_models, loss_locals = [], [], []

        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in idxs_users:
            local = FedAVGLocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local= local.update_weights(net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(loss_local)

        loss_round = sum(loss_locals) / len(loss_locals)
        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob_fl = aggregation(w_locals, dict_len)
        net_glob.load_state_dict(w_glob_fl)

        noisy_loss, clean_loss, _ = compute_split_loss(net_glob, dataset_train, noisy_sample_idx, device=args.device)

        print(f'round {rnd + 1} clean sample avg loss: {clean_loss}')
        print(f'round {rnd + 1} noise sample avg loss: {noisy_loss}')

        acc_s2 = global_test(net_glob.to(args.device), dataset_test, args)

        show_info_loss = "Round %d train loss  %.4f" % (rnd, loss_round)
        show_info_test_acc = "Round %d global test acc  %.4f \n" % (rnd, acc_s2)
        print(show_info_loss)
        print(show_info_test_acc)

        f_scores = []
        all_idxs_users = [i for i in range(args.num_users)]
        if rnd == args.rounds2 - 1:
            for idx in all_idxs_users:
                f_scores.append(cal_f_score(args, net_glob.to(args.device), dataset_train, y_train, dict_users[idx]))
            show_info_f_score = "Round %d f_score \n" % rnd
            print(show_info_f_score)
            print(str(f_scores))

    show_time_info = f"time: {time.time() - start}"
    print(show_time_info)


import torch
from torch.utils.data import Subset, DataLoader


def compute_split_loss(
        model,
        dataset,
        noisy_idx,
        criterion=torch.nn.CrossEntropyLoss(),
        batch_mode=True,
        batch_size=64,
        device=None
):
    """
    计算噪声样本和干净样本的平均损失

    参数：
    model: 已加载权重的模型
    dataset: 包含所有样本的Dataset对象
    noisy_idx: 噪声样本索引列表/集合
    criterion: 损失函数（默认交叉熵）
    batch_mode: 是否使用批量计算（默认True）
    batch_size: 批量大小（仅batch_mode=True时有效）
    device: 指定计算设备（默认自动获取模型所在设备）

    返回：
    (noisy_avg_loss, clean_avg_loss)
    """

    # 设备处理
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    # 创建索引集合
    noisy_set = set(noisy_idx)
    all_idx = set(range(len(dataset)))
    clean_set = all_idx - noisy_set

    num_samples_of_loss = [0 for _ in range(120)]

    if batch_mode:
        # 批量计算模式
        def _batch_compute(indices):
            subset = Subset(dataset, list(indices)) if indices else []
            if not subset: return 0.0
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

            total_loss = 0.0
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs,_ = model(inputs)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * inputs.size(0)
            return total_loss / len(subset) if subset else 0.0

        noisy_loss = _batch_compute(noisy_set)
        clean_loss = _batch_compute(clean_set)

    else:
        # 逐个样本计算模式
        noisy_loss, clean_loss = 0.0, 0.0
        noisy_count, clean_count = 0, 0

        with torch.no_grad():
            for i in range(len(dataset)):
                data, label = dataset[i]
                data = data.unsqueeze(0).to(device)
                label = torch.tensor([label], device=device) if not isinstance(label, torch.Tensor) else label.to(
                    device)

                output, _ = model(data)
                loss = criterion(output, label).item()

                num_samples_of_loss[loss/1+60] += 1

                if i in noisy_set:
                    noisy_loss += loss
                    noisy_count += 1
                else:
                    clean_loss += loss
                    clean_count += 1

        noisy_loss = noisy_loss / noisy_count if noisy_count > 0 else 0.0
        clean_loss = clean_loss / clean_count if clean_count > 0 else 0.0

    return noisy_loss, clean_loss, num_samples_of_loss