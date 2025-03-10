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
from system.trainning.local_training import FedTwinLocalUpdate, FedAVGLocalUpdate, FedAVGLocalUpdate_withCR
from system.trainning.test import global_test


def fed_avg(args):
    dataset_train, dataset_test, dict_users = get_dataset(args)
    dataset_train, dataset_test, dict_users, y_train, gamma_s, noisy_sample_idx = load_data_with_noisy_label(args, dataset_train, dataset_test, dict_users)
    start = time.time()
    net_glob = build_model(args)

    m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    prob = [1 / args.num_users] * args.num_users


    for rnd in range(args.rounds2):
        w_locals, p_models, loss_locals = [], [], []

        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in idxs_users:

            local = FedAVGLocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            if args.fedavgcr:
                local = FedAVGLocalUpdate_withCR(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local= local.update_weights(net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(loss_local)

        loss_round = sum(loss_locals) / len(loss_locals)
        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob_fl = aggregation(w_locals, dict_len)
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

    show_time_info = f"time: {time.time() - start}"
    print(show_time_info)