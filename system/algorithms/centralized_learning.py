import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from system.dataProcess.dataset import get_dataset
from system.dataProcess.load_data import load_data_with_noisy_label
from system.model.bulid_model import build_model
from system.algorithms.fed_avg_loss_static import compute_split_loss


def centralized_learning(args):
    device = args.device
    dataset_train, dataset_test, dict_users = get_dataset(args)
    if args.cl:
        dict_users = [np.arange(0, len(dataset_train))]
    dataset_train, dataset_test, dict_users, y_train, gamma_s, noisy_sample_idx = load_data_with_noisy_label(args,
                                                                                                             dataset_train,
                                                                                                             dataset_test,
                                                                                                             dict_users)

    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)

    start = time.time()
    model = build_model(args)

    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 使用 SGD 优化器

    # 定义一个简单的全连接神经网络
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 128)  # 输入层 (28x28) -> 隐藏层 (128)
            self.fc2 = nn.Linear(128, 64)  # 隐藏层 (128) -> 隐藏层 (64)
            self.fc3 = nn.Linear(64, 10)  # 隐藏层 (64) -> 输出层 (10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)  # 将图像展平为一维向量
            x = torch.relu(self.fc1(x))  # 激活函数 ReLU
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # 运行训练和测试
    train(model, device, dataset_train, train_loader, optimizer, criterion, noisy_sample_idx, epochs=args.rounds2, )
    test(model, device, test_loader, criterion)

    show_time_info = f"time: {time.time() - start}"
    print(show_time_info)


# 训练模型
def train(model, device, dataset_train, train_loader, optimizer, criterion, noisy_sample_idx, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # 清空梯度
            output, _ = model(data)
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        noisy_loss, clean_loss,_ = compute_split_loss(model, dataset_train, noisy_sample_idx, device=device)
        print(f'round {epoch} clean sample avg loss: {clean_loss}')
        print(f'round {epoch} noise sample avg loss: {noisy_loss}')

# 测试模型
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += criterion(output, target).item()  # 累加损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({accuracy:.0f}%)")
