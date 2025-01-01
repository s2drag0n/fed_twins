# -*- coding: utf-8 -*-
# @Time         : 2024/12/25 03:14
# @Author       : Song Zilong
# @Software     : PyCharm
# @Description  : 各种算法的本地训练阶段
import torch
from torch.utils.data import DataLoader, Dataset

from system.trainning.loss import FedTwinCRLoss, CORESLoss
from system.trainning.optimizer import adjust_learning_rate, MyOptimizer, f_beta


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]
        return image, label, self.idxs[item]


class FedTwinLocalUpdate:
    def __init__(self, args, dataset, idxs, client_idx):
        self.persionalized_model_bar = None
        self.args = args
        self.loss_func = FedTwinCRLoss()  # loss function -- cross entropy
        self.cores_loss_fun = CORESLoss(reduction='none')
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))
        self.client_idx = client_idx

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net_p, net_glob, rounds):
        # 设置本地模型和全局模型为训练模式
        net_p.train()
        net_glob.train()

        # 创建优化器，用于更新本地模型和全局模型的参数
        optimizer_theta = MyOptimizer(net_p.parameters(), lr=self.args.plr, lamda=self.args.lamda)
        # 使用 SGD 优化器更新全局模型的参数
        optimizer_w = torch.optim.SGD(net_glob.parameters(), lr=self.args.lr)

        # 初始化用于存储每个epoch损失的列表
        epoch_loss = []
        # 初始化用于存储每个epoch噪声样本数量的列表
        n_bar_k = []

        # 初始化噪声数据的索引字典
        local_ind_noise_p = {}
        local_ind_noise_g = {}

        # 遍历每个本地训练周期
        for iter in range(self.args.local_ep):
            batch_loss = []
            b_bar_p = []
            # 动态调整优化器的学习率
            adjust_learning_rate(rounds * self.args.local_ep + iter, self.args, optimizer_theta)
            adjust_learning_rate(rounds * self.args.local_ep + iter, self.args, optimizer_w)
            # 获取当前的学习率
            # plr = adjust_learning_rate(rounds * self.args.local_ep + iter, self.args, 'plr')
            lr = adjust_learning_rate(rounds * self.args.local_ep + iter, self.args, 'lr')

            # 遍历训练数据集
            for batch_idx, (images, labels, indexes) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                labels = labels.long()

                # 获取两个模型的对数概率输出
                log_probs_p, _ = net_p(images)
                log_probs_g, _ = net_glob(images)

                # 计算损失
                loss_p, loss_g, len_loss_g, len_loss_g, ind_g, ind_noise_p, ind_noise_g \
                    = self.loss_func(labels, log_probs_p, log_probs_g, rounds, iter, self.args)

                # 如果是最后一个训练周期，保存噪声数据的索引
                if iter == self.args.local_ep - 1:
                    if self.args.unsupervised:
                        local_ind_noise_p[indexes] = ind_noise_p
                        local_ind_noise_g[indexes] = ind_noise_g

                for i in range(self.args.K):
                    net_p.zero_grad()
                    if i == 0:
                        loss_p.backward()
                        self.persionalized_model_bar, _ = optimizer_theta.step(list(net_glob.parameters()))
                    else:
                        log_probs_p, _ = net_p(images)
                        beta = f_beta(rounds * self.args.local_ep + iter, self.args)
                        loss_p = self.cores_loss_fun(log_probs_p, labels, beta)
                        loss_p = torch.sum(loss_p[ind_g]) / len(loss_p[ind_g])
                        loss_p.backward()
                        self.persionalized_model_bar, _ = optimizer_theta.step(list(net_glob.parameters()))

                for new_param, localweight in zip(self.persionalized_model_bar, net_glob.parameters()):
                    localweight.data = localweight.data - self.args.lamda * lr * (
                            localweight.data - new_param.data)

                net_glob.zero_grad()
                loss_g.backward()
                optimizer_w.step()
                batch_loss.append(loss_g.item())
                b_bar_p.append(len_loss_g)

                # 如果数据集是 clothing1m，只使用前 100 个批次作为一个小 epoch
                if self.args.dataset == 'clothing1m':
                    if batch_idx >= 100:
                        break

            # 记录每个 epoch 的损失和噪声数据的数量
            n_bar_k.append(sum(b_bar_p))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # 计算平均损失和噪声数据的平均数量
        n_bar_k = sum(n_bar_k) / len(n_bar_k)
        return net_p, net_glob.state_dict(), sum(epoch_loss) / len(
            epoch_loss), n_bar_k, local_ind_noise_p, local_ind_noise_g
