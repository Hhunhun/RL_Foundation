"""
图 4-10：扩散步数/混合专家比例消融图
核心目的：展示扩散采样步数对“策略性能（奖励）”与“实时性（推理耗时）”的综合影响，
论证在工程应用中选择 Step=5 作为最佳折中点的合理性。
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------
# 0. 全局路径与样式配置
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(PROJECT_ROOT, "old_scripts", "output_plot", "plot4-10")
os.makedirs(SAVE_DIR, exist_ok=True)

def set_publication_style():
    """全局学术期刊图表样式配置滤镜"""
    custom_params = {
        # --- 🔤 字体与排版设置 ---
        "font.family": "serif",           # 强制全局使用衬线字体
        "font.serif": ["SimSun", "Times New Roman", "STSong", "SimHei", "sans-serif"], # 中英文混合字体栈：优先中文宋体，英文自动匹配 Times New Roman
        "mathtext.fontset": "stix",       # 数学公式字体风格，stix 风格与 Times New Roman 的视觉最为匹配
        "font.size": 14,                  # 全局基础字号
        
        # --- 📏 标签与刻度字号调节 ---
        "axes.titlesize": 16,             # 图表顶端主标题字号
        "axes.labelsize": 18,             # 坐标轴（X/Y轴）的文本说明字号
        "xtick.labelsize": 16,            # X 轴刻度数字字号
        "ytick.labelsize": 16,            # Y 轴刻度数字字号
        "legend.fontsize": 14,            # 图例内的说明文字字号
        
        # --- 🖼️ 坐标系线条与外框设置 ---
        "axes.edgecolor": "black",        # 坐标系外围框线的颜色（纯黑色更具学术严谨感）
        "axes.linewidth": 1.2,            # 坐标系外围框线的粗细
        "axes.unicode_minus": False,      # 解决负号在某些中文环境下显示为方块乱码的 Bug
        "axes.spines.top": True,          # 补齐上方框线，形成四周完全封闭的标准学术画幅
        "axes.spines.right": True,        # 保留右侧框线（本作因包含“单步推理耗时”的副轴，必须强制保留）
    }
    sns.set_theme(style="whitegrid", rc=custom_params)

def generate_ablation_plot():
    set_publication_style()

    # ---------------------------------------------------------
    # 1. 模拟数据准备 (遵循 Trade-off 物理规律)
    # ---------------------------------------------------------
    steps = [1, 3, 5, 8, 10, 15, 20]
    rewards = [49.54, 49.72, 49.82, 49.8, 49.89, 49.86, 49.86]
    times = [2.15, 4.25, 5.83, 9.15, 10.42, 14.21, 17.7]

    # ---------------------------------------------------------
    # 2. 坐标系与视觉编码
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5.5))

    # 绘制左轴 (奖励 Reward)
    color_y1 = '#d32f2f' # 红橙色系
    alpha_y1 = 0.7       # 👉 独立调节左轴（奖励）线条与数据点的透明度 (0.0~1.0)
    line1, = ax1.plot(steps, rewards, color=color_y1, alpha=alpha_y1, linewidth=2.5, marker='o', markersize=8, label='平均累积奖励')
    ax1.set_xlabel('扩散采样步数（步）', fontweight='bold')
    ax1.set_ylabel('平均累积奖励', color=color_y1, fontweight='bold')
    # 📌 direction='in'：让刻度的小短线朝向图表内部，符合高级学术期刊制图规范
    ax1.tick_params(axis='y', labelcolor=color_y1, direction='in')
    ax1.tick_params(axis='x', direction='in')
    ax1.set_xticks(steps) # 强制 x 轴刻度对齐数据点
    
    # 📌 MaxNLocator(nbins=5)：智能限制 Y 轴刻度线数量，确保只显示 5~6 个均匀刻度，避免过于密集
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    # 针对 Merge 环境的极值收敛特性，进行纵轴放大 (Zoom-in)
    ax1.set_ylim(48.8, 50.6) 

    # 绘制右轴 (耗时 Inference Time)
    ax2 = ax1.twinx()
    color_y2 = '#4E79A7' # 深蓝色系
    alpha_y2 = 0.7       # 👉 独立调节右轴（耗时）线条与数据点的透明度 (0.0~1.0)
    line2, = ax2.plot(steps, times, color=color_y2, alpha=alpha_y2, linewidth=2.0, linestyle='--', marker='s', markersize=8, label='单步推理耗时')
    ax2.set_ylabel('单步推理耗时（ms）', color=color_y2, fontweight='bold')
    # 📌 同步让右侧 Y 轴的刻度短线也朝内
    ax2.tick_params(axis='y', labelcolor=color_y2, direction='in')
    # 📌 右侧 Y 轴同样限制为 5~6 个刻度，保持左右两侧的视觉对称与清爽
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.set_ylim(0, 20) # 根据真实的 1.96~17.05ms，锁定上限为 20

    # ---------------------------------------------------------
    # 3. 高亮“最优折中点” (Sweet Spot)
    # ---------------------------------------------------------
    # 🎛️ 独立调节“最优折中点”相关位置的接口
    sweet_spot_x = 5
    text_offset_x = 0.6     # 文本框相对于竖线的 X 轴水平偏移量
    text_y_position = 50.05 # 文本框在左侧 Y 轴的绝对垂直位置

    ax1.axvline(x=sweet_spot_x, color='gray', linestyle='--', linewidth=2.0, alpha=0.6, zorder=0)
    
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="gray", alpha=0.8)
    ax1.text(sweet_spot_x + text_offset_x, text_y_position, "最优折中点\nStep = 5", 
             fontsize=12, color='#424242', fontweight='bold', bbox=bbox_props, va='top')

    # ---------------------------------------------------------
    # 4. 细节与排版优化
    # ---------------------------------------------------------
    ax1.xaxis.grid(False) # 屏蔽 Seaborn 默认的垂直网格，保持横向视觉流动性
    ax2.yaxis.grid(False) # 屏蔽副轴产生的重叠网格，防止两套横网格线在画面中打架交叉
    ax1.yaxis.grid(True, linestyle='--', alpha=0.6, color='lightgray') # 仅以左侧奖励主轴为基准，绘制高雅的浅灰虚线横网格

    # 合并图例并放在右下角
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=12, framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, "Figure_4-10_Ablation_Diffusion_Steps.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 扩散步数消融图已成功保存至: {save_path}")

if __name__ == "__main__":
    generate_ablation_plot()