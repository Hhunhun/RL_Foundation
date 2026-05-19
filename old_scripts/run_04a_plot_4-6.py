"""
图 4-6：单帧状态下输出动作分布对比图 (2D Action Manifold)
核心目的：从概率密度的几何视角，直观揭示传统 SAC 的“均值塌陷”顽疾，
以及 Diff-SAC 完美捕捉“多模态意图 (Multi-modal)”的底层数学优势。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------
# 0. 全局路径与样式配置
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(PROJECT_ROOT, "old_scripts", "output_plot", "plot4-6")
os.makedirs(SAVE_DIR, exist_ok=True)

def set_publication_style():
    """全局学术期刊图表样式配置滤镜"""
    custom_params = {
        "font.family": "serif",
        "font.serif": ["SimSun", "Times New Roman", "STSong", "SimHei", "sans-serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.titlesize": 15,             # 标题稍大，字重加粗
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.5,
        "axes.unicode_minus": False,
    }
    sns.set_theme(style="whitegrid", rc=custom_params)

# ---------------------------------------------------------
# 1. 数据来源与模型配置
# ---------------------------------------------------------
# 定义 Racetrack 环境的 PKL 路径
HARDCODED_PKL_PATH_RACETRACK = r"E:\Autol_Lab\RL_Foundation\outputs\racetrack-v0\eval_results\[R01_R02_R03_R04_R05_R06_R07_R08_DR01_DR02_DR03_DR04_DR05_DR06_DR07_DR08]_20260516_125952\data\all_results.pkl"

# 定义 Racetrack 环境的混合专家数据路径
# 这是 DR06 模型训练时所用的专家数据，它本身是多模态的
MIXED_EXPERT_DATA_PATH_RACETRACK = os.path.join(PROJECT_ROOT, "data", "expert_data", "racetrack-v0", "dataset_mixed_0.8R05_0.2R01_20260506_142446", "expert_transitions_mixed_0.8R05_0.2R01.npz")

def load_real_data(pkl_path, sac_model_id, diff_sac_model_id, expert_data_path):
    """
    从真实的 .pkl 和 .npz 文件中加载动作数据。
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"评估结果 PKL 文件未找到: {pkl_path}")
    if not os.path.exists(expert_data_path):
        raise FileNotFoundError(f"专家数据 NPZ 文件未找到: {expert_data_path}")

    with open(pkl_path, 'rb') as f:
        all_results = pickle.load(f)
    
    # 提取 SAC 和 Diff-SAC 的动作数据
    sac_actions = all_results.get(sac_model_id, {}).get('actions')
    diff_sac_actions = all_results.get(diff_sac_model_id, {}).get('actions')

    if sac_actions is None or len(sac_actions) == 0:
        raise ValueError(f"模型 {sac_model_id} 在 PKL 文件中未找到或动作数据为空。")
    if diff_sac_actions is None or len(diff_sac_actions) == 0:
        raise ValueError(f"模型 {diff_sac_model_id} 在 PKL 文件中未找到或动作数据为空。")

    # 加载混合专家数据 (Ground Truth)
    expert_raw_data = np.load(expert_data_path)
    expert_actions = expert_raw_data['actions']

    # 动作数据的裁剪与降采样 (防止 KDE 渲染过慢或过载)
    # 原始动作数据可能非常庞大，这里进行合理截取
    max_sample_size = 50000 # 可以根据需要调整
    
    # 确保动作数据是二维的 (Steering, Acceleration)，如果有多余维度需要裁剪
    sac_actions = sac_actions[:, :2] if sac_actions.ndim > 1 else sac_actions
    diff_sac_actions = diff_sac_actions[:, :2] if diff_sac_actions.ndim > 1 else diff_sac_actions
    expert_actions = expert_actions[:, :2] if expert_actions.ndim > 1 else expert_actions

    # 均匀降采样，保证数据量一致且足够代表整体分布
    if len(sac_actions) > max_sample_size:
        sac_actions = sac_actions[np.random.choice(len(sac_actions), max_sample_size, replace=False)]
    if len(diff_sac_actions) > max_sample_size:
        diff_sac_actions = diff_sac_actions[np.random.choice(len(diff_sac_actions), max_sample_size, replace=False)]
    if len(expert_actions) > max_sample_size:
        expert_actions = expert_actions[np.random.choice(len(expert_actions), max_sample_size, replace=False)]

    print(f"数据加载完成: SAC 动作点数 {len(sac_actions)}, Diff-SAC 动作点数 {len(diff_sac_actions)}, 专家动作点数 {len(expert_actions)}")
    return expert_actions, sac_actions, diff_sac_actions

def plot_action_manifold():
    set_publication_style()
    
    # 加载真实数据 (指定 Racetrack 环境和模型 ID)
    expert_data, sac_data, diff_sac_data = load_real_data(
        HARDCODED_PKL_PATH_RACETRACK, "R05", "DR06", MIXED_EXPERT_DATA_PATH_RACETRACK
    )
    
    print("==================================================")
    print("📊 正在渲染图 4-6：动作输出流形对比图...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap_choice = "magma"  # magma / mako 皆可，magma 热力对比更强，能突显金色十字

    # ==========================================
    # 左图：传统 SAC (均值塌陷)
    # ==========================================
    ax = axes[0]
    # 底层：KDE 核密度等高线图
    sns.kdeplot(x=sac_data[:, 0], y=sac_data[:, 1], ax=ax, fill=True, cmap=cmap_choice, levels=12, thresh=0.05, alpha=0.9)
    # 顶层：专家真实动作散点
    ax.scatter(expert_data[:, 0], expert_data[:, 1], marker='+', color='gold', s=80, linewidths=1.5, alpha=0.85, label='专家真实数据 (GT)')
    
    ax.set_title("传统 SAC (M04): 单峰均值塌陷 (Mode Averaging)", fontweight='bold', pad=15)
    
    # 关键特征标注：致命居中错误
    ax.annotate("致命居中错误", xy=(0, 0), xytext=(-0.5, -0.2),
                arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle="wedge,tail_width=0.6", alpha=0.8),
                color='red', fontweight='bold', fontsize=14, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1.5, alpha=0.8))

    # ==========================================
    # 右图：Diff-SAC (多模态捕捉)
    # ==========================================
    ax = axes[1]
    # 底层：KDE 核密度等高线图
    sns.kdeplot(x=diff_sac_data[:, 0], y=diff_sac_data[:, 1], ax=ax, fill=True, cmap=cmap_choice, levels=12, thresh=0.05, alpha=0.9)
    # 顶层：专家真实动作散点
    ax.scatter(expert_data[:, 0], expert_data[:, 1], marker='+', color='gold', s=80, linewidths=1.5, alpha=0.85, label='专家真实数据 (GT)')
    
    ax.set_title("混合专家 Diff-SAC (DM06): 双峰流形捕捉 (Multi-modal)", fontweight='bold', pad=15)
    
    # 关键特征标注：双意图识别
    ax.text(-0.5, 0.82, "意图1：左侧超车", color='white', fontweight='bold', fontsize=13, ha='center', bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.4))
    ax.text(0.5, -0.82, "意图2：右侧减速", color='white', fontweight='bold', fontsize=13, ha='center', bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.4))

    # ==========================================
    # 全局美化与保存
    # ==========================================
    for ax in axes:
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("方向盘转角 (Steering)", fontweight='bold')
        ax.set_ylabel("加速度 (Acceleration)", fontweight='bold')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.legend(loc="upper left", framealpha=0.9, edgecolor='black')
        ax.grid(linestyle='--', alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "Figure_4-6_Action_Manifold.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 图 4-6 (动作流形对比图) 已成功生成并保存至:\n📁 {save_path}")
    print("==================================================")

if __name__ == "__main__":
    plot_action_manifold() # 运行绘制函数