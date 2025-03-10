import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LinearLocator
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sympy.logic.boolalg import Boolean

# ======================
# 环境配置
# ======================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.chdir("../../")  # 根据实际项目结构调整路径



# ======================
# 可视化样式配置（核心复用部分）
# ======================
# 字体配置（中文宋体，英文/数字Times New Roman）
plt.rcParams.update({
    'font.family': 'sans-serif',  # 主字体族
    'font.sans-serif': ['SimSun', 'Times New Roman'],  # 中文宋体，英文自动匹配Times New Roman
    'axes.unicode_minus': False  # 解决负号显示问题
})

# 尺寸与排版配置
FIG_SIZE = (8, 4)  # 画布尺寸
FONT_CONFIG = {
    'title': 24,  # 标题字号
    'axis_label': 14,  # 坐标轴标签
    'tick_label': 12,  # 刻度标签
    'legend': 14  # 图例字号
}

# 颜色配置
COLOR_MAP = LinearSegmentedColormap.from_list(
    "noise_cmap",
    [(0.0, "#9ECAE1"), (0.3, "#ff7f0e"), (1.0, "#FCAE91")]  # 蓝-橙-红渐变
)
MEAN_LINE_COLOR = '#2ca02c'  # 均值线颜色
THRESHOLD_COLOR = '#d62728'  # 阈值线颜色
DENSITY_COLOR = '#9467bd'  # 密度图颜色

from matplotlib import ticker

def format_z_axis(value, pos):
    """
    自定义 z 轴刻度格式化函数
    将原始值除以 1000 后添加 'k' 单位
    自动处理小数显示逻辑
    """
    formatted_value = value / 1000
    # 整数处理逻辑
    if formatted_value.is_integer():
        return f'{int(formatted_value)}k'
    # 小数处理逻辑（保留1位小数）
    else:
        return f'{formatted_value:.1f}k'

def plot_client_class_distribution(dataset_train, dict_users, y_train, noisy_sample_idx, save_prefix1,save_prefix2, save_prefix3, swap_axes=False):
    """
    绘制三维堆叠柱状图展示客户端-类别样本分布
    :param swap_axes: 控制是否交换坐标轴，True时x轴为类别，y轴为客户端
    """
    # ======================
    # 数据预处理
    # ======================
    num_clients = len(dict_users)
    num_classes = len(np.unique(y_train))

    # 初始化计数矩阵
    clean_counts = np.zeros((num_clients + 1, num_classes), dtype=int)
    noisy_counts = np.zeros((num_clients + 1, num_classes), dtype=int)

    # 统计每个客户端每个类别的样本分布
    for client_id in range(1, num_clients + 1):
        for sample_idx in dict_users[client_id-1]:
            true_label = y_train[sample_idx]
            if sample_idx in noisy_sample_idx:
                noisy_counts[client_id][true_label] += 1
            else:
                clean_counts[client_id][true_label] += 1

    # 坐标轴控制逻辑
    if swap_axes:
        clean_counts = clean_counts.T
        noisy_counts = noisy_counts.T
        x_label, y_label = '类别', '客户端'
        x_ticks = np.arange(num_classes)
        y_ticks = np.arange(num_clients)
    else:
        x_label, y_label = '客户端', '类别'
        x_ticks = np.arange(num_clients) + 1
        y_ticks = np.arange(num_classes)

    # ======================
    # 可视化配置
    # ======================
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['SimSun', 'Times New Roman'],
        'axes.unicode_minus': False
    })

    # 颜色配置
    clean_color = COLOR_MAP(0.0)  # 蓝色表示干净样本
    noisy_color = COLOR_MAP(1.0)  # 红色表示噪声样本

    # ======================
    # 三维绘图核心逻辑
    # ======================
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111, projection='3d')

    # 生成网格坐标
    x_pos, y_pos = np.meshgrid(x_ticks, y_ticks, indexing='ij')
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()

    # 柱体参数配置
    bar_width = 0.4  # 柱体宽度
    bar_depth = 0.4  # 柱体深度
    gap = 0.1  # 柱体间距

    # 新增：有效数据点标记
    has_data_mask = (clean_counts + noisy_counts) > 0

    # 遍历所有可能的组合（仅绘制有数据的柱子）
    for x, y in zip(x_pos, y_pos):
        if not has_data_mask[x, y]:  # 新增条件判断
            continue  # 跳过无数据的柱子

        clean = clean_counts[x][y]
        noisy = noisy_counts[x][y]

        # 绘制逻辑优化
        if clean > 0:
            ax.bar3d(
                x - bar_width / 2 + gap / 2,
                y - bar_depth / 2 + gap / 2,
                0,
                bar_width - gap,
                bar_depth - gap,
                clean,
                color=clean_color,
                edgecolor='k' if clean > 0 else 'none',  # 无数据时隐藏边框
                linewidth=0.5
            )

        if noisy > 0:
            ax.bar3d(
                x - bar_width / 2 + gap / 2,
                y - bar_depth / 2 + gap / 2,
                clean,  # 注意z轴起始位置
                bar_width - gap,
                bar_depth - gap,
                noisy,
                color=noisy_color,
                edgecolor='k' if noisy > 0 else 'none',
                linewidth=0.5
            )

    # ======================
    # 图表装饰
    # ======================
    ax.set_xlabel(x_label, labelpad=5, fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel(y_label, labelpad=7, fontsize=FONT_CONFIG['axis_label'])
    ax.set_zlabel('样本数量', labelpad=13, fontsize=FONT_CONFIG['axis_label'])
    ax.zaxis.set_major_formatter(ticker.FuncFormatter(format_z_axis))

    max_z = np.max(clean_counts + noisy_counts)
    min_z = 0  # 默认从0开始

    # 设置刻度定位器（强制3个主刻度）
    ax.zaxis.set_major_locator(LinearLocator(numticks=3))
    # 智能格式化函数
    def smart_z_formatter(x, pos):
        if max_z < 1000:  # 小数据范围显示实际值
            return f'{int(x)}'
        else:  # 大数据范围用k单位
            value_k = x / 1000
            if value_k.is_integer():
                return f'{int(value_k)}k'
            else:
                return f'{value_k:.1f}k'

    ax.zaxis.set_major_formatter(FuncFormatter(smart_z_formatter))
    # 动态调整轴范围（保证刻度合理分布）
    if max_z > 0:
        ax.set_zlim(min_z, max_z * 1.05)  # 扩展5%避免顶格
    else:
        ax.set_zlim(0, 1000)  # 全零数据的兜底
    # # 三维图标签位置优化
    # ax.zaxis.set_tick_params(pad=5, rotation=15)
    # ax.zaxis._axinfo['juggled'] = (1, 0, 2)

    # 新增纵横比控制逻辑
    axis_ratio = 1.75  # x轴与y轴的长度比例系数
    if swap_axes:
        # 当坐标轴交换时，y轴对应原始客户端数量，需要更短的轴
        ax.set_box_aspect((axis_ratio, 1, 0.5))  # (x_axis, y_axis, z_axis)
    else:
        # 默认情况，x轴对应客户端需要更短的轴
        ax.set_box_aspect((1, axis_ratio, 0.5))  # (y_axis, x_axis, z_axis)

    # 设置刻度范围
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    # client_labels = [str(i + 1) for i in range(num_clients)]
    # ax.set_xticklabels(client_labels)
    ax.set_xlim(0, 6)
    ax.set_zlim(0, np.max(clean_counts + noisy_counts) * 1.2)



    # 设置刻度标签样式
    ax.tick_params(axis='x', labelsize=FONT_CONFIG['tick_label'],pad=1)
    ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label'],pad=1)
    ax.tick_params(axis='z', labelsize=FONT_CONFIG['tick_label'],pad=1)

    # 创建图例
    legend_elements = [
        Patch(facecolor=clean_color, label='干净样本'),
        Patch(facecolor=noisy_color, label='噪声样本')
    ]
    # ax.legend(
    #     handles=legend_elements,
    #     fontsize=FONT_CONFIG['legend'],
    #     loc='upper right',
    #     bbox_to_anchor=(1.45, 0.9)
    # )

    # 调整视角
    ax.view_init(elev=30, azim=-30)

    plt.tight_layout()

    plt.gcf().subplots_adjust(
        left=0.2,  # 左边距
        right=0.85,  # 右边距
        bottom=0.1,  # 下边距
        top=0.9  # 上边距
    )

    # # 可选：启用自动旋转标签
    # plt.xticks(rotation=30, ha='right')  # x轴标签旋转30度
    # plt.yticks(rotation=-30, va='top')  # y轴标签反向旋转
    fig = plt.gcf()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())

    # 扩展右侧空间（单位：英寸）
    adjusted_bbox = Bbox.from_extents(
        bbox.x0 - 0.5,  # 左边界扩展0.2英寸
        bbox.y0 + 0.2,  # 下边界扩展0.1英寸
        # bbox.x1 + 0.2,  # 右边界扩展0.5英寸（为图例留空间）
        bbox.x1 + 1.54,  # 右边界扩展0.5英寸（为图例留空间）
        bbox.y1 - 0.3  # 上边界扩展0.1英寸
    )

    # plt.savefig(f'{save_prefix1}_{save_prefix2}_{save_prefix3}_iid_and_noise.pdf', bbox_inches='tight')
    plt.savefig(f'{save_prefix1}_{save_prefix2}_{save_prefix3}_iid_and_noise.pdf', bbox_inches=adjusted_bbox)
    plt.close()



if __name__ == "__main__":
    # ======================
    # 数据准备（示例代码，需根据实际情况调整）
    # ======================
    from system.dataProcess.dataset import get_dataset
    from system.dataProcess.load_data import load_data_with_noisy_label
    from system.utils.arg_paser import args_parser

    args = args_parser()
    args.dataset = "cifar10"
    args.iid = False
    args.level_n_system = 0.5
    args.level_n_lowerb = 0
    args.num_users = 10
    args.non_iid_prob_class = 0.15

    # 加载数据
    dataset_train, dataset_test, dict_users = get_dataset(args)
    dataset_train, dataset_test, dict_users, y_train, gamma_s, noisy_sample_idx = load_data_with_noisy_label(
        args, dataset_train, dataset_test, dict_users)

    for i in range(10):
        labels_users = set([])
        for idx in dict_users[i]:
            labels_users.add(y_train[idx])
        print(f'client {i} has {len(labels_users)} labels')



    print(f'noisy ratio = {len(noisy_sample_idx) / len(dataset_train)}')

#     os.chdir("experiment/iid_and_noise/")
#     plot_client_class_distribution(dataset_train, dict_users, y_train, noisy_sample_idx, args.non_iid_prob_class,args.level_n_system,args.level_n_lowerb,swap_axes=False)
# # 使用示例
# # plot_client_class_distribution(dataset_train, dict_users, y_train, noisy_sample_idx, swap_axes=False)