import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ======================
# 可视化样式配置（与之前保持一致）
# ======================
# 字体配置（中文宋体，英文/数字Times New Roman）
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimSun', 'Times New Roman'],
    'axes.unicode_minus': False
})

# 尺寸与排版配置
FIG_SIZE = (15, 5)  # 与条形图相同画幅
FONT_CONFIG = {
    'title': 24,  # 标题字号（未使用）
    'axis_label': 21,  # 坐标轴标签
    'tick_label': 18,  # 刻度标签
    'legend': 21  # 图例字号
}

# 颜色配置（与条形图配色体系一致）
CL_COLOR = '#1f77b4'  # 蓝色（来自渐变色起点）
FL_COLOR = '#ff7f0e'  # 橙色（来自渐变色中点）

# ======================
# 数据准备
# ======================
noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 噪声比例
cl_acc = [92.94, 71.34, 68.29, 57.18, 52.14, 44.01, 34.84, 25.90, 20.45, 14.37]  # 集中式
fl_acc = [87.50, 53.90, 47.04, 43.62, 34.43, 28.53, 25.97, 20.09, 17.19, 13.41]  # 联邦


# ======================
# 可视化逻辑
# ======================
def plot_accuracy_curve():
    # 初始化画布
    plt.figure(figsize=FIG_SIZE)

    # 设置坐标轴样式
    ax = plt.gca()
    ax.spines['bottom'].set_color('#2e2e2e')  # 深灰坐标轴
    ax.spines['left'].set_color('#2e2e2e')
    ax.spines['bottom'].set_linewidth(1.5)  # 轴宽统一
    ax.spines['left'].set_linewidth(1.5)

    # 绘制折线
    plt.plot(noise_levels, cl_acc,
             marker='o', markersize=8,
             color=CL_COLOR, linewidth=2.5,
             label='集中式学习 (CL)')

    plt.plot(noise_levels, fl_acc,
             marker='s', markersize=8,
             color=FL_COLOR, linewidth=2.5,
             label='联邦学习 (FL)')

    # 坐标轴设置
    plt.xlabel('噪声比例',
               fontsize=FONT_CONFIG['axis_label'],
               fontweight='bold',
               labelpad=10)

    plt.ylabel('准确度 (%)',
               fontsize=FONT_CONFIG['axis_label'],
               fontweight='bold',
               labelpad=10)

    # 刻度设置
    plt.xticks(noise_levels,
               fontsize=FONT_CONFIG['tick_label'])

    plt.yticks(np.arange(0, 101, 20),
               fontsize=FONT_CONFIG['tick_label'])

    plt.xlim(-0.05, 0.95)  # 留白优化
    plt.ylim(0, 100)

    # 网格样式
    plt.grid(True,
             linestyle=':',
             linewidth=1,
             color='gray',
             alpha=0.6)

    # 图例设置
    plt.legend(loc='upper right',
               fontsize=FONT_CONFIG['legend'],
               frameon=True,
               framealpha=0.9,
               edgecolor='#2e2e2e')

    # 输出保存
    plt.tight_layout()
    plt.savefig('accuracy_comparison.pdf',
                dpi=300,
                bbox_inches='tight',
                format='pdf')
    plt.close()


# ======================
# 执行流程
# ======================
if __name__ == "__main__":
    plot_accuracy_curve()