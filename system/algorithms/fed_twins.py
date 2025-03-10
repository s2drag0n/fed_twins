# -*- coding: utf-8 -*-
# @Time         : 2024/12/25 02:15
# @Author       : Song Zilong
# @Software     : PyCharm
# @Description  : 本实验主要方法
import copy
import math
import time

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.utils.data import Subset, DataLoader

from system.correction.correction import lid_term
from system.dataProcess.dataset import get_dataset
from system.dataProcess.load_data import load_data_with_noisy_label
from system.model.bulid_model import build_model
from system.trainning.aggration import personalized_aggregation
from system.trainning.fscore import cal_f_score
from system.trainning.get_output import get_output
from system.trainning.local_training import FedTwinLocalUpdate
from system.trainning.test import global_test


def fed_twins(args):
    dataset_train, dataset_test, dict_users = get_dataset(args)
    dataset_train, dataset_test, dict_users, y_train, gamma_s, noisy_sample_idx = load_data_with_noisy_label(args,
                                                                                                             dataset_train,
                                                                                                             dataset_test,
                                                                                                             dict_users)
    start = time.time()
    net_glob = build_model(args)

    m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    prob = [1 / args.num_users] * args.num_users

    lid_accumulative_client = np.zeros(args.num_users)

    for rnd in range(args.rounds2):
        if rnd <= args.begin_sel:
            print("\rRounds {:d} early training:".format(rnd), end='\n', flush=True)
        else:
            print("\rRounds {:d} filter noisy data:".format(rnd), end='\n', flush=True)

        w_locals, p_models, loss_locals, n_bar = [], [], [], []

        # lid_whole = np.zeros(len(y_train))
        loss_whole = np.zeros(len(y_train))
        lid_client = np.zeros(args.num_users)
        loss_accumulative_whole = np.zeros(len(y_train))

        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in idxs_users:
            local = FedTwinLocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], client_idx=idx)
            p_model, w_local, loss_local, n_bar_k, local_ind_noise_p, local_ind_noise_g = local.update_weights(
                net_p=copy.deepcopy(net_glob).to(args.device),
                net_glob=copy.deepcopy(net_glob).to(args.device), rounds=rnd)

            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            p_models.append(p_model)
            loss_locals.append(loss_local)
            n_bar.append(n_bar_k)

            # label correction part
            if args.correction and rnd >= args.pre_correction_begin_round:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                net_correction = build_model(args)
                net_correction.load_state_dict(w_local)
                local_output, loss = get_output(
                    loader,
                    net_correction,
                    args,
                    latent=False,
                    criterion=nn.CrossEntropyLoss(reduction='none'))
                lid_local = list(lid_term(local_output, local_output))
                sample_idx = np.array(list(dict_users[idx]))
                # lid_whole[sample_idx] = lid_local
                loss_whole[sample_idx] = loss
                lid_client[idx] = np.mean(lid_local)

        if rnd == 0 or rnd == 20 or rnd ==30 or rnd ==40:
            noisy_loss, clean_loss, num_samples_of_loss = compute_split_loss(net_glob, dataset_train, noisy_sample_idx,
                                                                             batch_mode=False, device=args.device)
            print(f'round {rnd + 1} clean sample avg loss: {clean_loss}')
            print(f'round {rnd + 1} noise sample avg loss: {noisy_loss}')
            print(f'round {rnd + 1} num_samples_of_loss: {num_samples_of_loss}')
        else:
            noisy_loss, clean_loss, num_samples_of_loss = compute_split_loss(net_glob, dataset_train, noisy_sample_idx, device=args.device)
            print(f'round {rnd + 1} clean sample avg loss: {clean_loss}')
            print(f'round {rnd + 1} noise sample avg loss: {noisy_loss}')


        loss_round = sum(loss_locals) / len(loss_locals)
        w_glob_fl = personalized_aggregation(net_glob.state_dict(), w_locals, n_bar, args.gamma)
        net_glob.load_state_dict(w_glob_fl)

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

        # label correction part
        if args.correction and rnd >= args.correction_begin_round:
            lid_accumulative_client = lid_accumulative_client + np.array(lid_client)
            loss_accumulative_whole = loss_accumulative_whole + np.array(loss_whole)

            # Apply Gaussian Mixture Model to LID
            gmm_lid_accumulative = GaussianMixture(n_components=2, random_state=args.seed, init_params='k-means++', max_iter=200).fit(
                np.array(lid_accumulative_client).reshape(-1, 1))
            labels_lid_accumulative = gmm_lid_accumulative.predict(np.array(lid_accumulative_client).reshape(-1, 1))
            clean_label = np.argsort(gmm_lid_accumulative.means_[:, 0])[0]

            noisy_set = np.where(labels_lid_accumulative != clean_label)[0]
            estimated_noisy_level = np.zeros(args.num_users)

            for client_id in noisy_set:
                sample_idx = np.array(list(dict_users[client_id]))
                loss = np.array(loss_accumulative_whole[sample_idx])
                gmm_loss = GaussianMixture(n_components=2, random_state=args.seed, init_params='k-means++', max_iter=200).fit(np.array(loss).reshape(-1, 1))
                labels_loss = gmm_loss.predict(np.array(loss).reshape(-1, 1))
                gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]

                pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
                estimated_noisy_level[client_id] = len(pred_n) / len(sample_idx)
                # print("client {} noisy level: {}".format(client_id, estimated_noisy_level[client_id]))

            for idx in noisy_set:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                loss = np.array(loss_accumulative_whole[sample_idx])
                local_output, _ = get_output(
                    loader,
                    net_glob.to(args.device),
                    args, False,
                    criterion=nn.CrossEntropyLoss(reduction='none'))
                relabel_idx = (-loss).argsort()[
                              :int(len(sample_idx) * estimated_noisy_level[idx] * args.relabel_ratio)]
                relabel_idx = list(
                    set(np.where(np.max(local_output, axis=1) > args.confidence_thres)[0]) & set(relabel_idx))
                # print("client {} relabel sample num: {}".format(idx, len(relabel_idx)))

                y_train_noisy_new = np.array(dataset_train.targets)
                y_train_noisy_new[sample_idx[relabel_idx]] = np.argmax(local_output, axis=1)[relabel_idx]
                dataset_train.targets = y_train_noisy_new

    show_time_info = f"time: {time.time() - start}"
    print(show_time_info)



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

                output,_ = model(data)
                loss = criterion(output, label).item()

                num_samples_of_loss[math.floor(loss)+60] += 1

                if i in noisy_set:
                    noisy_loss += loss
                    noisy_count += 1
                else:
                    clean_loss += loss
                    clean_count += 1

        noisy_loss = noisy_loss / noisy_count if noisy_count > 0 else 0.0
        clean_loss = clean_loss / clean_count if clean_count > 0 else 0.0

    return noisy_loss, clean_loss, num_samples_of_loss
