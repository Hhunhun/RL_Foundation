import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# ---------------------------------------------------------
# 0. 全局样式设置 (符合学术期刊标准)
# ---------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["SimSun", "Times New Roman"],  # 支持中文宋体
    "mathtext.fontset": "stix",  # 确保数学公式字体与 Times New Roman 协调
    "axes.titlesize": 18,
    "axes.labelsize": 12,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.2,
    "axes.unicode_minus": False,  # 解决中文宋体下负号显示为方块的问题
})

# ---------------------------------------------------------
# 0.5 图表局部细节调节接口 (Style Configuration)
# ---------------------------------------------------------
PLOT_STYLE = {
    "label_fontsize": 16,    # 横轴“动作 a”与纵轴“概率密度”的字号大小
    "title_pad": 12          # 图标题与上方坐标轴之间的距离 (留白间距)
}


def plot_multimodal_comparison():
    # ---------------------------------------------------------
    # 1. 数据模拟 (Data Generation)
    # ---------------------------------------------------------
    x = np.linspace(-5, 5, 500)
    
    # 专家真实分布 (Ground Truth)：完美的双峰意图 (减速让行 vs 加速超车)
    y_gt = 0.5 * norm.pdf(x, loc=-2.5, scale=0.6) + 0.5 * norm.pdf(x, loc=2.5, scale=0.6)
    
    # 左图：标准的单模态高斯分布 (Baseline SAC)
    y_left = norm.pdf(x, loc=0, scale=1.2)
    
    # 右图：高斯混合模型 (Diff-SAC 多模态先验)
    # 与专家分布近乎完美重合，证明扩散模型的多模态表达能力
    y_right = 0.5 * norm.pdf(x, loc=-2.5, scale=0.65) + 0.5 * norm.pdf(x, loc=2.5, scale=0.65)
               
    # 封装单个子图的绘制逻辑，以便生成拼接图和单图
    def draw_left(ax):
        ax.plot(x, y_gt, color='gray', linestyle='--', linewidth=2.0, label='真实意图')
        ax.plot(x, y_left, color='#4DBBD5', linewidth=2.0, label='SAC 策略')
        ax.fill_between(x, y_left, color='#4DBBD5', alpha=0.15)
        ax.set_title("基线 SAC 算法（单模态策略）", pad=PLOT_STYLE["title_pad"])
        ax.set_xlabel("动作 $a$", fontsize=PLOT_STYLE["label_fontsize"])
        ax.set_ylabel("概率密度 $\pi(a|s)$", fontsize=PLOT_STYLE["label_fontsize"])
        ax.set_xlim([-5, 5])
        ax.set_ylim([0, 0.45])  # 稍微抬高顶部，为注释留出呼吸空间
        ax.set_yticks([])  
        ax.set_xticks([-2.5, 2.5])
        ax.set_xticklabels(["减速让行", "加速超车"], fontsize=12)
        
        # 添加均值塌陷(Mode Averaging)的红色高亮错误标注
        peak_y = np.max(y_left)
        ax.annotate("居中危险动作\n(无效指令)", xy=(0, peak_y), xytext=(0, peak_y + 0.05),
                    arrowprops=dict(arrowstyle="->", color="#E64B35", lw=1.5),
                    color="#E64B35", ha="center", va="bottom", fontweight="bold", fontsize=12)
        ax.legend(loc="upper left", frameon=False, fontsize=11)

    def draw_right(ax):
        ax.plot(x, y_gt, color='gray', linestyle='--', linewidth=2.0, label='真实意图')
        ax.plot(x, y_right, color="#8491B4", linewidth=2.0, label='Diff-SAC 策略')
        ax.fill_between(x, y_right, color="#8491B4", alpha=0.15)
        ax.set_title("Diff-SAC 算法（多模态策略）", pad=PLOT_STYLE["title_pad"])
        ax.set_xlabel("动作 $a$", fontsize=PLOT_STYLE["label_fontsize"])
        ax.set_ylabel("概率密度 $\pi(a|s)$", fontsize=PLOT_STYLE["label_fontsize"])
        ax.set_xlim([-5, 5])
        ax.set_ylim([0, 0.45])
        ax.set_yticks([])
        ax.set_xticks([-2.5, 2.5])
        ax.set_xticklabels(["减速让行", "加速超车"], fontsize=12)
        ax.legend(loc="upper left", frameon=False, fontsize=11)

    # ---------------------------------------------------------
    # 2. 图形绘制与排版 (Styling & Layout)
    # ---------------------------------------------------------
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plot", "plot_2-5_multimodal")
    os.makedirs(save_dir, exist_ok=True)
    
    # === 绘制并保存 1x2 拼接图 ===
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    draw_left(axes[0])
    draw_right(axes[1])
    plt.tight_layout()
    
    output_combined = os.path.join(save_dir, "multimodal_policy_comparison_combined.png")
    plt.savefig(output_combined, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()
    
    # === 单独输出左侧单模态图 ===
    fig_left, ax_left = plt.subplots(figsize=(5, 4))
    draw_left(ax_left)
    plt.tight_layout()
    output_left = os.path.join(save_dir, "multimodal_policy_single.png")
    plt.savefig(output_left, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

    # === 单独输出右侧多模态图 ===
    fig_right, ax_right = plt.subplots(figsize=(5, 4))
    draw_right(ax_right)
    plt.tight_layout()
    output_right = os.path.join(save_dir, "multimodal_policy_multi.png")
    plt.savefig(output_right, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

    print(f"✅ 组合对比图已生成: {output_combined}")
    print(f"✅ 单模态独立图已生成: {output_left}")
    print(f"✅ 多模态独立图已生成: {output_right}")

if __name__ == "__main__":
    plot_multimodal_comparison()