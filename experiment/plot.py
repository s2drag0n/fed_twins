import re
import matplotlib.pyplot as plt

# 初始化两个数组来存储train loss和test accuracy的值
train_loss_values = []
test_accuracy_values = []

# 打开文件并读取每一行
with open('cifar_10_1_05.txt', 'r') as file:
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

    print("max acc: {}".format(max(test_accuracy_values)))
    print("最大值于第{}轮取得".format(test_accuracy_values.index(max(test_accuracy_values))))

    # 生成轮次数组，假设轮次是从0开始的
    rounds = list(range(min_length))

    # 绘制train loss图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rounds, train_loss_values, label='Train Loss')
    plt.title('Training Loss per Round')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制test accuracy图
    plt.subplot(1, 2, 2)
    plt.plot(rounds, test_accuracy_values, label='Test Accuracy', color='green')
    plt.title('Test Accuracy per Round')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()

    # 显示图表
    plt.tight_layout()
    plt.show()
else:
    print("No data found.")