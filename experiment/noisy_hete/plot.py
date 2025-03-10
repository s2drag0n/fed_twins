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
# 数据准备（示例代码，需根据实际情况调整）
# ======================
from system.dataProcess.dataset import get_dataset
from system.dataProcess.load_data import load_data_with_noisy_label
from system.utils.arg_paser import args_parser

args = args_parser()
args.dataset = "cifar10"
args.iid = True
args.level_n_system = 0.9
args.level_n_lowerb = 0.0
args.num_users = 10

# 加载数据
dataset_train, dataset_test, dict_users = get_dataset(args)
dataset_train, dataset_test, dict_users, y_train, gamma_s, noisy_sample_idx = load_data_with_noisy_label(
    args, dataset_train, dataset_test, dict_users)

os.chdir("experiment/noisy_hete/")

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
FIG_SIZE = (15, 5)  # 画布尺寸
FONT_CONFIG = {
    'title': 24,  # 标题字号
    'axis_label': 21,  # 坐标轴标签
    'tick_label': 18,  # 刻度标签
    'legend': 21  # 图例字号
}

# 颜色配置
COLOR_MAP = LinearSegmentedColormap.from_list(
    "noise_cmap",
    [(0.0, "#1f77b4"), (0.3, "#ff7f0e"), (1.0, "#d62728")]  # 蓝-橙-红渐变
)
MEAN_LINE_COLOR = '#2ca02c'  # 均值线颜色
THRESHOLD_COLOR = '#d62728'  # 阈值线颜色
DENSITY_COLOR = '#9467bd'  # 密度图颜色


# ======================
# 核心逻辑函数
# ======================
def calculate_client_noise_rates(dict_users, noisy_sample_idx):
    """计算客户端噪声比例"""
    # print(dict_users)
    global_noisy_set = set(noisy_sample_idx)
    return np.array([len(set(dict_users[i]) & global_noisy_set) / len(dict_users[i]) * 100
                     if (len(dict_users[i]) > 0) else 0 for i in range(len(dict_users))])


# ======================
# 可视化逻辑
# ======================
def plot_noise_distribution(epsilon, client_ids, save_prefix):
    # 初始化画布
    fig, ax1 = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.1, right=0.85)

    # 保持原始顺序（Client 1到Client 10）
    raw_order_idx = np.arange(len(epsilon))
    sorted_clients = [client_ids[i] for i in raw_order_idx]
    sorted_epsilon = epsilon[raw_order_idx]

    # 主图：横向条形图
    bar_colors = [COLOR_MAP(min(e / 100, 1)) for e in sorted_epsilon]
    bars = ax1.barh(sorted_clients, sorted_epsilon,
                    color=bar_colors, edgecolor='whitesmoke', linewidth=0.5)

    # 强制横坐标从0开始
    ax1.set_xlim(left=0)  # 关键修改点

    # 辅助线标注
    mean_val = np.mean(sorted_epsilon)
    ax1.axvline(mean_val, color=MEAN_LINE_COLOR, ls='--', lw=1.5, alpha=0.8)
    ax1.axvline(50, color=THRESHOLD_COLOR, ls=':', lw=1.5)

    # 密度图（限制范围）
    ax2 = ax1.twinx()
    sns.kdeplot(sorted_epsilon, ax=ax2, color=DENSITY_COLOR,
                fill=True, alpha=0.2, linewidth=1.5, bw_adjust=0.5,
                clip=(0, None))  # 关键修改点

    # 样式配置
    ax1.set_xlabel('噪声率 (%)', fontsize=FONT_CONFIG['axis_label'],
                   fontweight='bold', labelpad=10)
    ax1.set_ylabel('客户端ID', fontsize=FONT_CONFIG['axis_label'],
                   fontweight='bold', labelpad=10)
    ax1.tick_params(axis='both', labelsize=FONT_CONFIG['tick_label'])
    ax1.grid(axis='x', linestyle=':', alpha=0.6)
    ax1.set_facecolor('#f8f9fa')

    ax2.set_ylabel('概率密度', fontsize=FONT_CONFIG['axis_label'],
                   color=DENSITY_COLOR, labelpad=10)
    ax2.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label'],
                    colors=DENSITY_COLOR)
    ax2.spines['right'].set_color(DENSITY_COLOR)

    # 图例配置
    legend_elements = [
        Line2D([0], [0], color=MEAN_LINE_COLOR, ls='--', lw=1.5,
               label=f'平均噪声率 ({mean_val:.1f}%)'),
        Line2D([0], [0], color=THRESHOLD_COLOR, ls=':', lw=1.5,
               label='高噪声阈值'),
        Line2D([0], [0], color=DENSITY_COLOR, alpha=0.5, lw=3,
               label='噪声率分布')
    ]
    ax2.legend(handles=legend_elements, loc='upper right',
               fontsize=FONT_CONFIG['legend'], framealpha=0.9)

    # 保存输出
    plt.savefig(f'{save_prefix}_noise_heterogeneity.pdf', bbox_inches='tight')
    plt.savefig(f'{save_prefix}_noise_heterogeneity.png', dpi=300)
    plt.close()


# ======================
# 执行流程
# ======================
if __name__ == "__main__":
    # 计算噪声比例
    noise_rates = calculate_client_noise_rates(dict_users, noisy_sample_idx)
    client_labels = [f"Client {i + 1}" for i in range(len(noise_rates))]

    # 生成可视化
    plot_noise_distribution(noise_rates, client_labels, args.dataset)