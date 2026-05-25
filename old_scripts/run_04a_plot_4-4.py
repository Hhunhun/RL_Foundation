"""
图 4-4：全局变异系数 (CV) 对比柱状图
核心目的：展示 Diff-SAC 在高动态交互场景(Merge/Racetrack)的稳定性突破，
以及在简单纵向跟驰场景(Highway)中由于马尔可夫采样延迟导致的过参数化倒挂。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# ---------------------------------------------------------
# 0. 全局路径与样式配置
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(PROJECT_ROOT, "old_scripts", "output_plot", "plot4-4")
os.makedirs(SAVE_DIR, exist_ok=True)

def set_publication_style():
    """全局学术期刊图表样式配置滤镜"""
    custom_params = {
        "font.family": "serif",
        "font.serif": ["SimSun", "Times New Roman", "STSong", "SimHei", "sans-serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "axes.unicode_minus": False,
        "axes.spines.top": True,          # 强制显示上方边框线
        "axes.spines.right": True,        # 强制显示右侧边框线
    }
    sns.set_theme(style="whitegrid", rc=custom_params) # 对齐 run_03，使用高学术感的白底灰网格

# ---------------------------------------------------------
# 1. 数据来源与视觉映射配置
# ---------------------------------------------------------
# [重构数据结构]：支持在同一个环境下录入 CV 和 Reward 两种指标
DATA = {
    "merge-v0": {
        "cv": {"sac": 0.030, "diff": 0.025},
        "reward": {"sac": 48.7, "diff": 49.5}, # 奖励值演示数据，请根据实际情况修改
        "env_name": "匝道汇入 (merge-v0)"
    },
    "racetrack-v0": {
        "cv": {"sac": 1.186, "diff": 1.020},
        "reward": {"sac": 78.1, "diff": 83.2},   # 奖励值演示数据，请根据实际情况修改
        "env_name": "极限赛道 (racetrack-v0)"
    },
    #"highway-v0": {
    #    "cv": {"sac": 0.131, "diff": 0.293},
    #    "reward": {"sac": 25.0, "diff": 22.1},
    #    "env_name": "高速巡航 (Highway-v0)"
    #}
}

COLOR_SAC = '#8491B4'  # 莫灰紫 (Slate Purple) - 完全对齐 run_03 配色
COLOR_DIFF = '#E64B35' # 胭脂红 (Carmine Red) - 完全对齐 run_03 配色

def draw_metric_bar_chart(ax, env_key, metric='cv', is_single=False):
    """底层绘制函数：支持根据 metric 动态绘制 CV 或 Reward 柱状图"""
    d = DATA[env_key]
    val_sac = d[metric]["sac"]
    val_diff = d[metric]["diff"]
    
    x = np.array([0, 1])
    width = 0.45
    
    # 1. 绘制柱体
    bars = ax.bar(x, [val_sac, val_diff], width=width, color=[COLOR_SAC, COLOR_DIFF], edgecolor=[COLOR_SAC, COLOR_DIFF], linewidth=1.0, alpha=0.40) # 对齐 run_03 透明度与边框方案
    
    # 2. 坐标系标签与清理
    ax.set_xticks(x)
    # 组合图为了节省空间，可将名称写在底部；单图可直接写全名
    ax.set_xticklabels(['SAC 基线', 'Diff-SAC'] if is_single else ['SAC', 'Diff-SAC'], fontsize=12, fontweight='bold')
    
    # 智能解析指标相关的标题和文字格式
    if metric == 'cv':
        ax.set_ylabel('变异系数 (CV)', fontweight='bold')
        text_fmt = f"{{yval:.3f}}"
    else:
        ax.set_ylabel('平均累计奖励', fontweight='bold')
        text_fmt = f"{{yval:.1f}}"
        
    ax.set_xlabel(d["env_name"], fontweight='bold', labelpad=10)
    
    # 动态适应 Y 轴范围以容纳箭头
    max_y = max(val_sac, val_diff)
    ax.set_ylim(0, max_y * 1.50) # 调大顶部预留空间，防止抬高后的箭头文字被截断
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5)) # 智能限制 Y 轴只显示 5~6 个刻度
    
    # 3. 柱顶数值标注
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + max_y * 0.015, text_fmt.format(yval=yval), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    # 4. 绘制相对变化率连接箭头 (智能判断颜色涨跌优劣)
    pct_change = (val_diff - val_sac) / val_sac * 100
    
    if metric == 'cv':
        # CV 越小越好：降低为绿，升高为红
        color_text = '#2e7d32' if pct_change < 0 else '#c62828' 
    else:
        # 奖励 越大越好：升高为绿，降低为红
        color_text = '#2e7d32' if pct_change > 0 else '#c62828'
    
    # 将相对变化箭头的起点和终点整体向上大幅平移，彻底避开柱体
    y_start = val_sac + max_y * 0.25
    y_end = val_diff + max_y * 0.25
    
    ax.annotate('', xy=(1, y_end), xytext=(0, y_start),
                arrowprops=dict(arrowstyle="->", color='dimgray', lw=2.0, shrinkA=5, shrinkB=5))
                
    mid_y = (y_start + y_end) / 2
    ax.text(0.5, mid_y + max_y * 0.08, f"{pct_change:+.1f}%", 
            ha='center', va='bottom', color=color_text, fontweight='bold', fontsize=13)

def generate_plots():
    set_publication_style()
    
    num_envs = len(DATA)
    if num_envs == 0:
        print("没有可绘制的数据！")
        return
        
    metrics = ['cv', 'reward']
    metric_names = {'cv': 'CV', 'reward': 'Reward'}
    
    for metric in metrics:
        metric_name = metric_names[metric]
        print(f"==================================================")
        print(f"📊 正在生成 {metric_name} 对比图表...")
        
        # 步骤一：独立输出单图
        for env_key in DATA.keys():
            fig, ax = plt.subplots(figsize=(4.5, 5))
            draw_metric_bar_chart(ax, env_key, metric=metric, is_single=True)
            plt.tight_layout()
            save_path = os.path.join(SAVE_DIR, f"Figure_4-4_{metric_name}_Bar_{env_key}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 生成单环境图: {save_path}")

        # 步骤二：输出动态 1xN 组合对比图
        fig, axes = plt.subplots(1, num_envs, figsize=(4 * num_envs, 5)) 
        if num_envs == 1:
            axes = [axes]
            
        for ax, env_key in zip(axes, DATA.keys()):
            draw_metric_bar_chart(ax, env_key, metric=metric, is_single=False)
            
        # 全局图例顶置居中
        handles = [Patch(facecolor=COLOR_SAC, edgecolor=COLOR_SAC, linewidth=1.0, alpha=0.40, label='SAC 基线'), 
                   Patch(facecolor=COLOR_DIFF, edgecolor=COLOR_DIFF, linewidth=1.0, alpha=0.40, label='Diff-SAC')]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, prop={'weight': 'bold', 'size': 14})
        
        plt.tight_layout()
        combined_path = os.path.join(SAVE_DIR, f"Figure_4-4_{metric_name}_Bar_Combined_1x{num_envs}.png")
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 生成全场景组合图: {combined_path}")
        print("==================================================")

if __name__ == "__main__":
    generate_plots()