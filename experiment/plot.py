import re
import matplotlib.pyplot as plt
import os  # 导入os模块用于文件操作

# 初始化两个数组来存储train loss和test accuracy的值
train_loss_values = []
test_accuracy_values = []

file_path = 'result/cifar10_hete_level_2_iid_False_sample_ratio_0.7_correction_False'

# 打开文件并读取每一行
with open(file_path, 'r') as file:
    for line in file:
        # 使用正则表达式匹配包含"train loss"的行
        train_loss_match = re.search(r'Round \d+ train loss\s+([-\d.]+)', line)
        if train_loss_match:
            train_loss_values.append(float(train_loss_match.group(1)))

        # 使用正则表达式匹配包含"global test acc"的行
        test_acc_match = re.search(r'Round \d+ global test acc\s+([-\d.]+)', line)
        if test_acc_match:
            test_accuracy_values.append(float(test_acc_match.group(1)))

# 检查是否有提取到数据
if train_loss_values and test_accuracy_values:
    # 找到较短的长度
    min_length = min(len(train_loss_values), len(test_accuracy_values))

    # 截断较长的数组
    train_loss_values = train_loss_values[:min_length]
    test_accuracy_values = test_accuracy_values[:min_length]

    # 计算最大准确度及其对应的轮次
    max_acc = max(test_accuracy_values)
    max_acc_round = test_accuracy_values.index(max_acc)

    print("max acc: {:.4f}".format(max_acc))
    print("最大值于第{}轮取得".format(max_acc_round))

    # 生成轮次数组，假设轮次是从0开始的
    rounds = list(range(min_length))

    # 绘制train loss和test accuracy在同一张图上
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, train_loss_values, label='Train Loss', color='blue')
    plt.plot(rounds, test_accuracy_values, label='Test Accuracy', color='green')

    # 在最大准确度处添加突出标记和文本标注
    plt.scatter(max_acc_round, max_acc, color='red', zorder=5)  # 突出显示最大准确度的点
    plt.text(max_acc_round, max_acc, f"Max Acc: {max_acc:.4f}\nRound: {max_acc_round}",
             fontsize=10, verticalalignment='bottom', color='red', weight='bold')

    plt.title('Training Loss and Test Accuracy per Round')
    plt.xlabel('Round')
    plt.ylabel('Value')
    plt.legend()

    # 确保保存路径存在
    graph_folder = "graph"
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)  # 如果文件夹不存在，创建文件夹

    # 提取原始文件名并构造保存路径
    original_file_name = os.path.basename(file_path).replace('.', '-')  # 获取原始文件名（不含路径）
    save_path = os.path.join(graph_folder, f"{os.path.splitext(original_file_name)[0]}.png")  # 构造保存路径

    # 保存图表
    plt.savefig(save_path)
    print(f"图表已保存到: {save_path}")

    # 显示图表
    plt.tight_layout()
    plt.show()
else:
    print("No data found.")