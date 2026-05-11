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
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
    "axes.unicode_minus": False,  # 解决中文宋体下负号显示为方块的问题
})

def plot_multimodal_comparison():
    # ---------------------------------------------------------
    # 1. 数据模拟 (Data Generation)
    # ---------------------------------------------------------
    x = np.linspace(-5, 5, 500)
    
    # 左图：标准的单模态高斯分布 (Baseline SAC)
    y_left = norm.pdf(x, loc=0, scale=1.2)
    
    # 右图：高斯混合模型 (Diff-SAC 多模态先验)
    y_right = (0.50 * norm.pdf(x, loc=-2.5, scale=0.6) + 
               0.15 * norm.pdf(x, loc=0.5, scale=1.0) + 
               0.35 * norm.pdf(x, loc=3.0, scale=0.5))
               
    # 封装单个子图的绘制逻辑，以便生成拼接图和单图
    def draw_left(ax):
        ax.plot(x, y_left, color='#1f77b4', linewidth=2.0)
        ax.fill_between(x, y_left, color='#1f77b4', alpha=0.15)
        ax.set_title("基线 SAC 算法\n(单模态高斯策略)", pad=15)
        ax.set_xlabel("动作 $a$")
        ax.set_ylabel("概率密度 $\pi(a|s)$")
        ax.set_xlim([-5, 5])
        ax.set_ylim([0, 0.4])
        ax.set_yticks([])  
        ax.set_xticks([0])
        ax.set_xticklabels(['']) 

    def draw_right(ax):
        ax.plot(x, y_right, color='#2ca02c', linewidth=2.0)
        ax.fill_between(x, y_right, color='#2ca02c', alpha=0.15)
        ax.set_title("Diff-SAC 算法\n(多模态策略先验)", pad=15)
        ax.set_xlabel("动作 $a$")
        ax.set_ylabel("概率密度 $\pi(a|s)$")
        ax.set_xlim([-5, 5])
        ax.set_ylim([0, 0.4])
        ax.set_yticks([])
        ax.set_xticks([-2.5, 3.0])
        ax.set_xticklabels(["低速区间", "高速区间"])

    # ---------------------------------------------------------
    # 2. 图形绘制与排版 (Styling & Layout)
    # ---------------------------------------------------------
    save_dir = os.path.dirname(os.path.abspath(__file__))
    
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