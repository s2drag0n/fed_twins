import matplotlib.pyplot as plt
import os
import numpy as np

# ======================
# 可视化样式配置（核心复用部分）
# ======================
# 字体配置（中文宋体，英文/数字Times New Roman）
plt.rcParams.update({
    'font.family': 'sans-serif',  # 主字体族
    'font.sans-serif': ['SimSun', 'Times New Roman'],  # 中文宋体，英文自动匹配Times New Roman
    'axes.unicode_minus': False  # 解决负号显示问题
})

def smooth_curve(scalars, alpha=0.2):
    """指数移动平均平滑"""
    smoothed = [scalars[0]]
    for value in scalars[1:]:
        smoothed.append(smoothed[-1] * (1 - alpha) + value * alpha)
    return np.array(smoothed)


def extract_losses(file_path):
    clean_losses = []
    noise_losses = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 过滤出包含损失值的行
    loss_lines = [line.strip() for line in lines if "clean sample avg loss:" in line]
    loss_lines_noise = [line.strip() for line in lines if "noise sample avg loss:" in line]

    if len(loss_lines_noise) == 0:
        # 每两行处理一次
        for i in range(0, len(loss_lines), 2):
            if i + 1 >= len(loss_lines):
                break
            clean_loss = float(loss_lines[i].split(":")[1].strip())
            noise_loss = float(loss_lines[i + 1].split(":")[1].strip())
            clean_losses.append(clean_loss)
            noise_losses.append(noise_loss)
    else:
        for line in loss_lines:
            clean_losses.append(float(line.split(":")[1].strip()))
        for line in loss_lines_noise:
            noise_losses.append(float(line.split(":")[1].strip()))

    noise_ratio_lines = [line.strip() for line in lines if "noisy ratio =" in line]

    noise_ratio = 0
    for line in noise_ratio_lines:
        noise_ratio = float(line.split("=")[1].strip())

    return clean_losses, noise_losses, noise_ratio


def process_folder(folder_path):
    # 获取所有txt文件并按文件名排序
    files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path)
                    if f.endswith('.txt')])[:9]  # 只处理前9个文件

    # 收集全局范围数据
    all_clean, all_noise = [], []
    max_rounds = 0
    noise_ratio_all = []

    # 预读所有文件获取全局范围
    for file in files:
        clean, noise, noise_ratio = extract_losses(file)
        # clean = smooth_curve(clean, alpha=0.8)
        # noise = smooth_curve(noise, alpha=0.8)
        all_clean.extend(clean)
        all_noise.extend(noise)
        max_rounds = max(max_rounds, len(clean))

    # 计算全局坐标范围
    global_min = min(min(all_clean), min(all_noise))
    # global_max = max(max(all_clean), max(all_noise))
    global_max = 7
    x_max = max_rounds - 1  # 轮数从0开始

    # 创建3x3子图
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    plt.subplots_adjust(
        left=0.1,  # 左侧留白
        right=0.85,  # 右侧留白给行标题
        top=0.9,  # 顶部留白给列标题
        bottom=0.1,  # 底部留白
        wspace=0.05,  # 列间距
        hspace=0.05  # 行间距
    )
    column_labels = (f'({chr(963)}= 0.5,{chr(950)}= 0)\nNoise ratio = 0.15', f'({chr(963)}= 1,{chr(950)}= 0.3)\nNoise ratio = 0.5', f'({chr(963)}= 1,{chr(950)}= 0.8)\nNoise ratio = 0.8')
    row_labels = ('IID', 'Non-IID\n(p=0.3)', 'Non-IID\n(p=0.15)')
    for col_idx, title in enumerate(column_labels):
        x_pos = 0.22 + col_idx * 0.26  # 均匀分布三列位置
        fig.text(x_pos, 0.91, title,
                 ha='center', va='bottom',
                 fontweight='bold',fontsize=20)

        # 添加行标题（在右侧）
    for row_idx, title in enumerate(row_labels):
        y_pos = 0.77 - row_idx * 0.27  # 均匀分布三行位置
        fig.text(0.86, y_pos, title,
                 ha='left', va='center',
                 rotation=0,
                 fontweight='bold',fontsize=20)

    plt.rcParams.update({
        'axes.titlesize': 25,  # 标题字号
        'axes.labelsize': 20,  # 坐标轴标签字号
        'xtick.labelsize': 20,  # x轴刻度字号
        'ytick.labelsize': 20,  # y轴刻度字号
        'legend.fontsize': 20  # 图例字号
    })

    # 为所有子图保持一致的坐标范围
    padding = (global_max - global_min) * 0.05  # 5%的边距
    y_min = global_min - padding
    y_max = global_max + padding


    # 绘制每个子图
    for idx, (file, ax) in enumerate(zip(files, axes.flatten())):
        ax.grid(False)

        clean, noise, noise_ratio = extract_losses(file)
        rounds = np.arange(len(clean))

        # 绘制散点图
        ax.scatter(rounds, clean, color='blue', marker='o', s=20, label='Clean Loss')
        ax.scatter(rounds, noise, color='red', marker='x', s=20, label='Noise Loss')

        # 坐标轴控制逻辑
        row, col = idx // 3, idx % 3
        # 纵坐标设置（仅第一列显示）
        if col != 0:
            ax.set_yticklabels([])
            ax.set_yticks([0,2,4,6,8])
        else:
            ax.set_yticks([0,2,4,6,8])
        # 横坐标设置（仅最后一行显示）
        if row != 2:
            ax.set_xticklabels([])
            ax.set_xticks([0,100,200,300,400])
        else:
            ax.set_xticks([0, 100, 200, 300, 400])

        # 设置统一坐标范围
        ax.set_xlim(0, x_max + 0.5)
        ax.set_ylim(0, y_max)

        # 设置标签和标题
        if idx == 7:  # 最后一行显示x轴标签
            ax.set_xlabel('轮次', fontsize=20, labelpad=20)
        if idx == 3:  # 第一列显示y轴标签
            ax.set_ylabel('损失值', fontsize=20, labelpad=20)

        if idx == 2:
            handles = [
                plt.Line2D([], [], color='blue', linewidth=3, linestyle='dotted', label='clean'),
                plt.Line2D([], [], color='red', linewidth=3, linestyle='dotted', label='noise')
            ]
            ax.legend(handles=handles, loc='upper right', fontsize=20, frameon=True)


        # 从文件名提取标题
        # title = f'总噪声率：{noise_ratio}'
        # ax.set_title(title, fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', labelsize=20)



    # # 添加全局图例
    # handles, labels = axes[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=25)

    # 保存结果
    output_path = os.path.join(folder_path, "loss_combined_plots.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"组合图表已保存至：{output_path}")


if __name__ == "__main__":
    folder_path = "result"  # 修改为你的文件夹路径
    process_folder(folder_path)