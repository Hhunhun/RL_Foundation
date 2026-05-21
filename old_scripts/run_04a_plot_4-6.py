"""
图 4-6：冲突状态下的单步动作流形与分布意图对比图

核心目的：
通过在高度博弈的冲突状态下“冻结”时间，连续采样 500 次动作，
展示传统 SAC 模型由于高斯策略局限导致的“意图塌陷(均值平庸)”，
以及 Diff-SAC 由于扩散网络特性展现出的完美“多模态流形(同时保留让行与抢道两个专家意图)”。
"""

import os
import sys
import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from datetime import datetime
import warnings
import pickle

# ----------------------------------------------------
# 屏蔽底层第三方库的警告，保持控制台纯净
# ----------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 精准屏蔽 Gymnasium 的录像覆盖警告
warnings.filterwarnings("ignore", message=".*Overwriting existing videos.*")

# --- [核心修复] 动态追加根目录环境变量 ---
# 解决将脚本移动到子文件夹后，找不到 algorithms, core, envs 模块的问题
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# 导入底层组件
from algorithms.sac.sac_agent import SACAgent
from algorithms.diffusion_sac.diff_sac_agent import DiffSACAgent
from core.offline_buffer import MixedReplayBuffer
from envs import create_environment

def set_publication_style():
    """
    全局学术期刊图表样式配置滤镜，支持高度个性化排版定制。
    """
    custom_params = {
        # --- 字体配置 (注意中文兼容性) ---
        "font.family": "sans-serif",      # 默认字体族 (若需纯英文论文可改为 "serif")
        "font.sans-serif": ["SimSun", "SimHei", "PingFang SC", "Microsoft YaHei", "sans-serif"], # 中文后备字体栈，优先使用宋体(SimSun)
        "font.serif": ["SimSun", "Times New Roman"],# 衬线字体也加入宋体防乱码
        "font.size": 14,                  # 全局基础字号
        
        # --- 轴与标签字号配置 ---
        "axes.titlesize": 18,             # 图表标题字号
        "axes.labelsize": 16,             # 坐标轴标签字号
        "xtick.labelsize": 16,            # X轴刻度字号
        "ytick.labelsize": 16,            # Y轴刻度字号
        "legend.fontsize": 12,            # 图例字号 (如需使用图例)
        
        # --- 线条与外框配置 ---
        "lines.linewidth": 2.0,           # 全局线宽
        "axes.edgecolor": "black",        # 强制显示坐标轴外框
        "axes.linewidth": 1.2,            # 外框线宽 (略微加粗，更符合双栏排版)
        "axes.unicode_minus": False,      # 解决负号显示为方块的乱码问题
        
        # --- 输出保存配置 ---
        "figure.figsize": (8.0, 6.0),     # 默认画幅：宽 8.0 英寸，高 6.0 英寸 (4:3 比例，更适合 12 号字体的展示)
        "figure.dpi": 300,                # 默认分辨率 (学术期刊通常要求 300 或 600)
        "savefig.bbox": "tight"           # 保存时自动裁剪多余空白边距
    }
    
    sns.set_theme(style="whitegrid", rc=custom_params)

# ----------------------------------------------------
# 🔒 定义全局评估基础种子，并预先生成极度离散的测试种子库
# ----------------------------------------------------
EVAL_BASE_SEED = 52  # 👉 锁定表现最好的黄金种子
random.seed(EVAL_BASE_SEED)
np.random.seed(EVAL_BASE_SEED)
torch.manual_seed(EVAL_BASE_SEED)

def get_all_conflict_states(env_name, max_steps=40):
    """
    生成绝对公平的统一冲突状态序列。
    通过控制一辆基准车驶入交互博弈区，截取从开始到撞车前的所有物理快照。
    """
    print(f"\n🔍 正在生成公平测试用的物理快照序列 - {env_name}...")
    env = create_environment(env_name, is_eval=True, algo="sac")
    state, _ = env.reset(seed=EVAL_BASE_SEED)
    
    states = [state]
    
    for step in range(1, max_steps + 1):
        action = np.array([0.3, 0.0]) # 默认微加速直线行驶
        if env_name == "racetrack-v0":
            action = np.array([0.1, 0.0]) # 赛道更危险，速度放缓
            
        state, _, done, _, _ = env.step(action)
        states.append(state)
        if done:
            print(f"⚠️ 寻找冲突状态时在第 {step} 帧发生提前碰撞或结束。")
            break
            
    env.close()
    print(f"✅ 成功提取 0 到 {len(states)-1} 帧的物理快照序列！")
    return states

def capture_model_manifold_sequence(model_id, model_path, display_label, env_name, expert_data_path, conflict_states, num_samples=500):
    """
    核心逻辑：针对传入的所有状态序列，执行独立的大量采样，
    从而逼出模型内心深处的动作分布意图。
    """
    print(f"\n" + "=" * 60)
    print(f" 🎯 开始采集单步流形: [{display_label}] (ID: {model_id}) | 采样数: {num_samples}")
    print(f"📁 权重路径: {os.path.abspath(model_path)}")
    print("=" * 60)

    is_diff = "Diff" in model_id or "diff" in model_id or "DM" in model_id or "DH" in model_id or "DR" in model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    algo_type = "diff" if is_diff else "sac"

    dummy_env = create_environment(env_name, is_eval=True, algo=algo_type)
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.shape[0]
    max_action = float(dummy_env.action_space.high[0])
    dummy_env.close()

    if is_diff:
        print(f"🧠 检测到 Diffusion 架构 ({model_id})，正在挂载 DiffSACAgent 与数据归一化器...")
        buffer = MixedReplayBuffer(expert_data_path=expert_data_path, max_online_size=10, device=device)
        agent = DiffSACAgent(state_dim, action_dim, device=device)
        agent.load_pretrained_actor(model_path)
        agent.ema_actor.model.load_state_dict(agent.actor.state_dict())
    else: # SAC 架构
        print(f"🧠 检测到 SAC 架构 ({model_id})，正在挂载经典 SACAgent...")
        agent = SACAgent(state_dim, action_dim, action_scale=max_action)
        try:
            agent.load_model(model_path) # SACAgent 的 load_model 接受路径
        except Exception as e:
            print(f"❌ 加载模型 {display_label} 失败: {e}")
            return None

    all_steps_actions = []
    for step_idx, state in enumerate(conflict_states):
        actions = []
        for i in range(num_samples):
            if is_diff:
                norm_state = buffer.state_normalizer.normalize(state)
                norm_action = agent.select_action(norm_state, sample_steps=10, explore=True)
                action = buffer.action_normalizer.unnormalize(norm_action)
                action = np.clip(action, -1.0, 1.0)
            else:
                action = agent.select_action(state, evaluate=False)
                
            actions.append(action)
        all_steps_actions.append(np.array(actions))
        print(f"\r  └─ 正在生成多模态动作 (第 {step_idx:02d} 帧) ... 完成!", end="")
        
    print("\n✅ 序列动作流形意图采样完成！")
    return all_steps_actions

def plot_action_manifold(all_actions, models_to_evaluate, save_dir, env_name, step_idx=0):
    """
    生成高密度的动作核密度分布图 (Action KDE Grid)。
    展示各模型面对同一冲突状态时的策略多模态思维差异。
    """
    # 激活全局学术样式滤镜
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)
    model_ids = list(all_actions.keys())
    display_labels = [models_to_evaluate[mid]["display_name"] for mid in model_ids]
    
    # --- 采用《Nature》顶级期刊高对比度色系 (NPG Academic Palette) ---
    # 保持与训练曲线 (run_04) 绝对统一的视觉风格
    academic_colors = [
        "#8491B4", # 莫灰紫 (Slate Purple)
        "#E64B35", # 胭脂红 (Carmine Red)
        "#91D1C2", # 薄荷青 (Mint)
        "#4DBBD5", # 蔚蓝色 (Cerulean Blue)
        "#00A087", # 翠绿色 (Teal Green)
        "#3C5488", # 午夜蓝 (Midnight Blue)
        "#F39B7F", # 珊瑚粉 (Salmon Pink)
        "#DC0000", # 深红色 (Crimson)
        "#7E6148", # 咖啡褐 (Coffee Brown)
        "#B09C85", # 浅卡其 (Light Khaki)
        "#4E79A7", # 稳重蓝 (Muted Blue)
        "#A73030", # 暗红色 (Dark Red)
    ]
    colors = sns.color_palette(academic_colors, n_colors=len(model_ids))

    n_models = len(model_ids)
    # 无论多少个模型，此特定图表强制使用 1xN 布局横向对比
    fig, axes = plt.subplots(1, n_models, figsize=(4.0 * n_models, 4.5), sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        if i < n_models:
            mid = model_ids[i]
            color = colors[i]
            label = display_labels[i]
            
            actions = all_actions[mid]
            if len(actions) > 0:
                if actions.shape[1] >= 2:
                    # X轴：横向控制(转向) | Y轴：纵向控制(加减速)
                    x_plot, y_plot = actions[:, 1], actions[:, 0]
                        
                    try:
                        # 将渐变的终点颜色加深为极深墨绿色 (#00332A)，使高密度区域的蜂窝在视觉上更加厚重饱满
                        custom_cmap = sns.light_palette("#00A087", as_cmap=True)
                        
                        # gridsize 调大至 50 让蜂窝网格变得更密，展现更细腻的动作流形
                        # 设定统一的 extent 保证不同子图的六边形网格在空间上绝对对齐
                        ax.hexbin(x_plot, y_plot, gridsize=50, cmap=custom_cmap, mincnt=1, alpha=0.9, extent=(-1.1, 1.1, -1.1, 1.1))
                    except Exception:
                        pass
                    
            ax.set_title(label, color='black', fontweight='bold', fontsize=18)
            if i == 0:
                ax.set_ylabel('纵向控制 / 加减速', fontweight='bold')
            ax.set_xlabel('横向控制 / 转向', fontweight='bold')
                
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            
            # 绘制极具科技感的原点十字准星 (调低透明度与颜色，置于底层，让其退居背景)
            ax.axhline(0, color='lightgray', linestyle='--', linewidth=1.0, alpha=0.4, zorder=0)
            ax.axvline(0, color='lightgray', linestyle='--', linewidth=1.0, alpha=0.4, zorder=0)
            ax.grid(False) # 屏蔽常规网格
            
            # [视觉升级] 针对 52 号种子的第 15 帧黄金画面，自动为多模态网络添加引导注释
            if step_idx == 15:
                if "Diff" in label or "混合" in label:
                    # 🎛️ 右图：“抢道意图”与“让行意图”的坐标调节接口 (X, Y)
                    qiangdao_arrow_target = (0.2, 1.0) # 抢道箭头指向的坐标
                    qiangdao_text_pos = (-0.6, 0.7)     # 抢道文本所在坐标
                    rangxing_arrow_target = (0.0, -1.0)# 让行箭头指向的坐标
                    rangxing_text_pos = (0.6, -0.8)    # 让行文本所在坐标
                    
                    ax.annotate("让行意图", xy=rangxing_arrow_target, xytext=rangxing_text_pos,
                                arrowprops=dict(arrowstyle="->", color='#E64B35', lw=1.5, connectionstyle="arc3,rad=0.2"),
                                fontsize=13, fontweight='bold', color='#E64B35', ha='center', zorder=10)
                                
                    ax.annotate("抢道意图", xy=qiangdao_arrow_target, xytext=qiangdao_text_pos, 
                                arrowprops=dict(arrowstyle="->", color='#E64B35', lw=1.5, connectionstyle="arc3,rad=-0.2"),
                                fontsize=13, fontweight='bold', color='#E64B35', ha='center', zorder=10)
                else:
                    # 🎛️ 左图：“均值塌陷”的坐标调节接口 (X, Y)
                    collapse_arrow_target = (-0.2, 0.0) # 箭头指向的绿团中心坐标 (请根据图表现实情况微调)
                    collapse_text_pos = (0.6, -0.1)     # 文本所在坐标
                    
                    ax.annotate("均值塌陷\n（居中跟随）", xy=collapse_arrow_target, xytext=collapse_text_pos,
                                arrowprops=dict(arrowstyle="->", color='#E64B35', lw=1.5, connectionstyle="arc3,rad=0.2"),
                                fontsize=13, fontweight='bold', color='#E64B35', ha='center', zorder=10)

            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(direction='in')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    # [核心修复] 文件名附加当前帧编号，支持批量检视
    save_path = os.path.join(save_dir, f"Figure_4-6_Action_Manifold_{env_name}_Step_{step_idx:02d}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 第 {step_idx:02d} 帧分布对比图已保存至: {save_path}")


if __name__ == "__main__":
    # ==========================================
    # 终端交互配置终端
    # ==========================================
    print("🤖 单步动作流形扫描器 (Action Manifold Scanner)")
    print("==========================================")
    
    # 1. 选择运行模式
    print("👉 请选择运行模式:")
    print("  [1] 实时扫描 (重新寻找物理帧并提取动作意图)")
    print("  [2] 快速重绘 (跳过仿真，读取最新已有数据直接出图)")
    mode_choice_input = input("请输入 1 或 2 (默认 2): ").strip()
    mode_choice = mode_choice_input if mode_choice_input else '2' # Default to '2' if input is empty
    PLOT_ONLY = (mode_choice == '2')
    
    # 2. 选择环境
    print("==========================================")
    print("👉 请选择评估环境:")
    print("  [H] Highway 环境 (highway-v0)")
    print("  [M] Merge 环境 (merge-v0)")
    print("  [R] Racetrack 环境 (racetrack-v0)")
    env_choice = input("请输入 H, M 或 R (默认 H): ").strip().upper()
    if env_choice == 'M':
        TARGET_ENV = "merge-v0"
    elif env_choice == 'R':
        TARGET_ENV = "racetrack-v0"
    else:
        TARGET_ENV = "highway-v0"
    print(f"✅ 已锁定评估环境: {TARGET_ENV}")

    # 3. 自动寻找 pkl (如果开启重绘)
    LOAD_PKL_PATH = None
    if PLOT_ONLY:
        data_dir = os.path.join(PROJECT_ROOT, "old_scripts", "output_plot", "plot4-6", TARGET_ENV, "data")
        pkl_file = os.path.join(data_dir, "action_manifold.pkl")
        
        if os.path.exists(pkl_file):
            LOAD_PKL_PATH = pkl_file
            print(f"✅ 已应用指定的数据文件: {LOAD_PKL_PATH}")
        else:
            print(f"❌ 找不到数据文件 {pkl_file}，请先运行 [1] 实时扫描 以生成图表！")
            sys.exit(1)

    # 4. 选择图表标签风格
    print("==========================================")
    print("👉 请选择图表标签风格:")
    print("  [1] 原始工程调试标签 (例如: DM01 纯 BC 克隆)")
    print("  [2] 学术中文规范标签 (例如: Diff-SAC (纯行为克隆))")
    label_choice = input("请输入 1 或 2 (默认 1): ").strip()
    USE_ACADEMIC_LABELS = (label_choice == '2')
    print("==========================================")

    # 5. 选择绘制帧范围
    print("👉 请选择要绘制的帧范围:")
    print("  [A] 批量绘制全序列 (0 到 N 帧)")
    print("  [S] 仅绘制特定单帧 (例如黄金帧 15)")
    frame_choice = input("请输入 A 或 S (默认 A): ").strip().upper()
    PLOT_ALL_FRAMES = (frame_choice != 'S')
    TARGET_FRAME_IDX = 15
    if not PLOT_ALL_FRAMES:
        idx_input = input("👉 请输入指定的帧编号 (默认 15): ").strip()
        TARGET_FRAME_IDX = int(idx_input) if idx_input.isdigit() else 15
    print("==========================================")

    # ==========================================
    # 实验配置区：模型演进与消融实验
    # ==========================================
    if TARGET_ENV == "merge-v0":
        # 定义该环境下的两大流形数据集路径
        SINGLE_DATA_PATH = "data/expert_data/merge-v0/dataset_M04_mode1_20260423_154904/expert_transitions.npz"
        MIXED_DATA_PATH = "data/expert_data/merge-v0/dataset_mixed_0.8M04_0.2M03_20260513_161828/expert_transitions_mixed_0.8M04_0.2M03.npz"
        
        models_to_evaluate = {
            # === 第一期 SAC 消融矩阵 ===
            #"M01": {"path": "outputs/merge-v0/models/SAC_M01_Base_Merge_20260511_042953/sac_merge_final.pth", "raw_name": "M01 基础生存", "acad_name": "SAC-标准基线"},
            #"M02": {"path": "outputs/merge-v0/models/SAC_M02_Efficient_Smooth_20260420_154007/sac_merge_final.pth", "raw_name": "M02 高效平滑", "acad_name": "SAC-平顺偏好"},
            #"M03": {"path": "outputs/merge-v0/models/SAC_M03_Aggressive_Gap_Finding_20260420_162217/sac_merge_final.pth", "raw_name": "M03 激进寻隙", "acad_name": "SAC-效率导向"},
            "M04": {"path": "outputs/merge-v0/models/SAC_M04_Safety_First_20260420_170911/sac_merge_final.pth", "raw_name": "M04 安全至上", "acad_name": "SAC 基线"},
            #"M05": {"path": "outputs/merge-v0/models/SAC_M05_Patient_Merger_20260420_220108/sac_merge_final.pth", "raw_name": "M05 耐心等待", "acad_name": "SAC-保守适应"},
            #"M06": {"path": "outputs/merge-v0/models/SAC_M06_Extreme_Penalty_20260420_232207/sac_merge_final.pth", "raw_name": "M06 极限死刑", "acad_name": "SAC-强安全约束"},
            #"M07": {"path": "outputs/merge-v0/models/SAC_M07_Smooth_Marathon_20260421_003822/sac_merge_final.pth", "raw_name": "M07 平滑马拉松", "acad_name": "SAC-长视界平顺"},
            #"M08": {"path": "outputs/merge-v0/models/SAC_M08_Ultimate_Merge_20260421_023258/sac_merge_final.pth", "raw_name": "M08 终极汇入", "acad_name": "SAC-综合强约束"},

            # === 第一期 diff-SAC 单专家实验 ===
            #"DM01": {"path": "outputs/merge-v0/models/DiffSAC_DM01_Pure_BC_20260511_135709/online_finetune/diff_sac_final.pth", "raw_name": "DM01 纯 BC 克隆", "acad_name": "单专家 Diff-SAC-纯BC", "data_path": SINGLE_DATA_PATH},
            #"DM02": {"path": "outputs/merge-v0/models/DiffSAC_DM02_Micro_Q_20260511_152002/online_finetune/diff_sac_final.pth", "raw_name": "DM02 微引导", "acad_name": "单专家 Diff-SAC-微引导", "data_path": SINGLE_DATA_PATH},
            #"DM03": {"path": "outputs/merge-v0/models/DiffSAC_DM03_Standard_Q_20260511_170511/online_finetune/diff_sac_final.pth", "raw_name": "DM03 标准引导", "acad_name": "单专家 Diff-SAC-标准引导", "data_path": SINGLE_DATA_PATH},
            #"DM04": {"path": "outputs/merge-v0/models/DiffSAC_DM04_Strong_Q_20260511_183810/online_finetune/diff_sac_final.pth", "raw_name": "DM04 强力干预", "acad_name": "单专家 Diff-SAC-强引导", "data_path": SINGLE_DATA_PATH},

            # === 第二期 diff-SAC 混合专家实验 ===
            #"DM05": {"path": "outputs/merge-v0/models/DiffSAC_DM05_Mixed_BC_20260513_163546/online_finetune/diff_sac_final.pth", "raw_name": "DM05 混合纯BC", "acad_name": "混合专家 Diff-SAC-纯BC", "data_path": MIXED_DATA_PATH},
            "DM06": {"path": "outputs/merge-v0/models/DiffSAC_DM06_Mixed_Micro_Q_20260513_175844/online_finetune/diff_sac_final.pth", "raw_name": "DM06 混合微引导", "acad_name": "Diff-SAC", "data_path": MIXED_DATA_PATH},
            #"DM07": {"path": "outputs/merge-v0/models/DiffSAC_DM07_Mixed_Standard_Q_20260513_194923/online_finetune/diff_sac_final.pth", "raw_name": "DM07 混合标引导", "acad_name": "混合专家 Diff-SAC-标准引导", "data_path": MIXED_DATA_PATH},
            #"DM08": {"path": "outputs/merge-v0/models/DiffSAC_DM08_Mixed_Strong_Q_20260513_211948/online_finetune/diff_sac_final.pth", "raw_name": "DM08 混合强干预", "acad_name": "混合专家 Diff-SAC-强引导", "data_path": MIXED_DATA_PATH},
        }

    elif TARGET_ENV == "racetrack-v0":
        SINGLE_DATA_PATH = "data/expert_data/racetrack-v0/dataset_R05_mode1_20260506_011817/expert_transitions.npz"
        MIXED_DATA_PATH = "data/expert_data/racetrack-v0/dataset_mixed_0.8R05_0.2R01_20260506_142446/expert_transitions_mixed_0.8R05_0.2R01.npz"
        
        models_to_evaluate = {
            # === 第一期 SAC 消融矩阵 ===
            #"R01": {"path": "outputs/racetrack-v0/models/SAC_R01_SAC_Baseline_20260505_033212/sac_racetrack_final.pth", "raw_name": "R01 基础 SAC", "acad_name": "SAC-标准基线"},
            #"R02": {"path": "outputs/racetrack-v0/models/SAC_R02_SAC_Speed_Priority_20260505_060152/sac_racetrack_final.pth", "raw_name": "R02 速度优先", "acad_name": "SAC-效率导向"},
            #"R03": {"path": "outputs/racetrack-v0/models/SAC_R03_SAC_Safety_Priority_20260505_083207/sac_racetrack_final.pth", "raw_name": "R03 安全优先", "acad_name": "SAC-安全约束"},
            #"R04": {"path": "outputs/racetrack-v0/models/SAC_R04_SAC_Extreme_Drift_20260505_110254/sac_racetrack_final.pth", "raw_name": "R04 极限漂移", "acad_name": "SAC-无约束探索"},
            #"R05": {"path": "outputs/racetrack-v0/models/SAC_R05_SAC_Smooth_Racing_20260505_131614/sac_racetrack_final.pth", "raw_name": "R05 单专家", "acad_name": "SAC-平顺专家"},
            #"R06": {"path": "outputs/racetrack-v0/models/SAC_R06_SAC_Wide_Dynamic_20260505_152958/sac_racetrack_final.pth", "raw_name": "R06 宽域动态", "acad_name": "SAC-宽域动态"},
            #"R07": {"path": "outputs/racetrack-v0/models/SAC_R07_SAC_Zero_Tolerance_20260505_173235/sac_racetrack_final.pth", "raw_name": "R07 零容忍", "acad_name": "SAC-强安全约束"},
            #"R08": {"path": "outputs/racetrack-v0/models/SAC_R08_SAC_Expert_Pro_20260505_184949/sac_racetrack_final.pth", "raw_name": "R08 专家底座", "acad_name": "SAC-专家基准"},
    
            # === 第一期 diff-SAC 单专家实验 ===
            #"DR01": {"path": "outputs/racetrack-v0/models/DiffSAC_DR01_Pure_BC_20260510_025310/online_finetune/diff_sac_final.pth", "raw_name": "DR01 纯 BC 克隆", "acad_name": "单专家 Diff-SAC-纯BC", "data_path": SINGLE_DATA_PATH},
            #"DR02": {"path": "outputs/racetrack-v0/models/DiffSAC_DR02_Micro_Q_20260510_060657/online_finetune/diff_sac_final.pth", "raw_name": "DR02 微引导", "acad_name": "单专家 Diff-SAC-微引导", "data_path": SINGLE_DATA_PATH},
            #"DR03": {"path": "outputs/racetrack-v0/models/DiffSAC_DR03_Standard_Q_20260510_092222/online_finetune/diff_sac_final.pth", "raw_name": "DR03 标准引导", "acad_name": "单专家 Diff-SAC-标准引导", "data_path": SINGLE_DATA_PATH},
            #"DR04": {"path": "outputs/racetrack-v0/models/DiffSAC_DR04_Strong_Q_20260510_123826/online_finetune/diff_sac_final.pth", "raw_name": "DR04 强力干预", "acad_name": "单专家 Diff-SAC-强引导", "data_path": SINGLE_DATA_PATH},

            # === 第二期 Diff-SAC 混合专家实验 ===
            #"DR05": {"path": "outputs/racetrack-v0/models/DiffSAC_DR05_Mixed_BC_20260510_155536/online_finetune/diff_sac_final.pth", "raw_name": "DR05 混合纯BC", "acad_name": "混合专家 Diff-SAC-纯BC", "data_path": MIXED_DATA_PATH},
            #"DR06": {"path": "outputs/racetrack-v0/models/DiffSAC_DR06_Mixed_Micro_Q_20260510_191112/online_finetune/diff_sac_final.pth", "raw_name": "DR06 混合微引导", "acad_name": "混合专家 Diff-SAC-微引导", "data_path": MIXED_DATA_PATH},
            #"DR07": {"path": "outputs/racetrack-v0/models/DiffSAC_DR07_Mixed_Standard_Q_20260510_223055/online_finetune/diff_sac_final.pth", "raw_name": "DR07 混合标引导", "acad_name": "混合专家 Diff-SAC-标准引导", "data_path": MIXED_DATA_PATH},
            #"DR08": {"path": "outputs/racetrack-v0/models/DiffSAC_DR08_Mixed_Strong_Q_20260511_015410/online_finetune/diff_sac_final.pth", "raw_name": "DR08 混合强干预", "acad_name": "混合专家 Diff-SAC-强引导", "data_path": MIXED_DATA_PATH},
                    
            "R06": {"path": "outputs/racetrack-v0/models/SAC_R06_SAC_Wide_Dynamic_20260505_152958/sac_racetrack_final.pth", "raw_name": "R06 宽域动态", "acad_name": "SAC 基线"},
            "DR04": {"path": "outputs/racetrack-v0/models/DiffSAC_DR04_Strong_Q_20260510_123826/online_finetune/diff_sac_final.pth", "raw_name": "DR04 强力干预", "acad_name": "Diff-SAC", "data_path": SINGLE_DATA_PATH},
        }

    else: # highway-v0
        SINGLE_DATA_PATH = "data/expert_data/highway-v0/dataset_H02_mode1_20260513_161932/expert_transitions.npz"
        MIXED_DATA_PATH = "data/expert_data/highway-v0/dataset_mixed_0.8H02_0.2H01_20260513_204225/expert_transitions_mixed_0.8H02_0.2H01.npz"
        
        models_to_evaluate = {
            # === 第一期 SAC 消融矩阵 ===
            "H01": {"path": "outputs/highway-v0/models/SAC_H01_Base_Highway_20260511_225245/sac_highway_final.pth", "raw_name": "H01 基础高速", "acad_name": "SAC-标准基线"},
            "H02": {"path": "outputs/highway-v0/models/SAC_H02_Safety_Priority_20260512_040012/sac_highway_final.pth", "raw_name": "H02 安全优先", "acad_name": "SAC-安全约束"},
            "H03": {"path": "outputs/highway-v0/models/SAC_H03_Speed_Priority_20260512_100655/sac_highway_final.pth", "raw_name": "H03 速度优先", "acad_name": "SAC-效率导向"},
            #"H04": {"path": "outputs/highway-v0/models/SAC_H04_Traffic_Jam_20260512_154634/sac_highway_final.pth", "raw_name": "H04 拥堵路况", "acad_name": "SAC-拥堵适应"},

            # === 第一期 diff-SAC 单专家实验 ===
            "DH01": {"path": "outputs/highway-v0/models/DiffSAC_DH01_Pure_BC_20260514_023858/online_finetune/diff_sac_final.pth", "raw_name": "DH01 纯 BC 克隆", "acad_name": "单专家 Diff-SAC-纯BC", "data_path": SINGLE_DATA_PATH},
            #"DH02": {"path": "outputs/highway-v0/models/DiffSAC_DH02_Micro_Q_20260514_075013/online_finetune/diff_sac_final.pth", "raw_name": "DH02 微引导", "acad_name": "单专家 Diff-SAC-微引导", "data_path": SINGLE_DATA_PATH},
            #"DH03": {"path": "outputs/highway-v0/models/DiffSAC_DH03_Standard_Q_20260514_130251/online_finetune/diff_sac_final.pth", "raw_name": "DH03 标准引导", "acad_name": "单专家 Diff-SAC-标准引导", "data_path": SINGLE_DATA_PATH},
            #"DH04": {"path": "outputs/highway-v0/models/DiffSAC_DH04_Strong_Q_20260514_181545/online_finetune/diff_sac_final.pth", "raw_name": "DH04 强力干预", "acad_name": "单专家 Diff-SAC-强引导", "data_path": SINGLE_DATA_PATH},
            
            # === 第二期 Diff-SAC 混合专家实验 ===
            #"DH05": {"path": "outputs/highway-v0/models/DiffSAC_DH05_Mixed_BC_20260515_050420/online_finetune/diff_sac_final.pth", "raw_name": "DH05 混合纯BC", "acad_name": "混合专家 Diff-SAC-纯BC", "data_path": MIXED_DATA_PATH},
            "DH06": {"path": "outputs/highway-v0/models/DiffSAC_DH06_Mixed_Micro_Q_20260515_100237/online_finetune/diff_sac_final.pth", "raw_name": "DH06 混合微引导", "acad_name": "混合专家 Diff-SAC-微引导", "data_path": MIXED_DATA_PATH},
            #"DH07": {"path": "outputs/highway-v0/models/DiffSAC_DH07_Mixed_Standard_Q_20260515_180112/online_finetune/diff_sac_final.pth", "raw_name": "DH07 混合标引导", "acad_name": "混合专家 Diff-SAC-标准引导", "data_path": MIXED_DATA_PATH},
            #"DH08": {"path": "outputs/highway-v0/models/DiffSAC_DH08_Mixed_Strong_Q_20260516_015209/online_finetune/diff_sac_final.pth", "raw_name": "DH08 混合强干预", "acad_name": "混合专家 Diff-SAC-强引导", "data_path": MIXED_DATA_PATH},
        }

    # 动态应用所选名称
    for k, v in models_to_evaluate.items():
        v["display_name"] = v["acad_name"] if USE_ACADEMIC_LABELS else v["raw_name"]

    target_plot_folder = "plot4-6"
    plot_save_dir = os.path.join(PROJECT_ROOT, "old_scripts", "output_plot", target_plot_folder, TARGET_ENV)
    data_save_dir = os.path.join(plot_save_dir, "data")
    os.makedirs(plot_save_dir, exist_ok=True)

    if not PLOT_ONLY:
        # 1. 提取连续的冲突时间序列
        conflict_states = get_all_conflict_states(TARGET_ENV)
        
        # 2. 对每个模型进行序列全采样
        NUM_SAMPLES = 500
        all_models_sequence_actions = {}
        for model_id, model_config in models_to_evaluate.items():
            model_path = model_config["path"]
            display_label = model_config["display_name"]
            expert_data_path = model_config.get("data_path", None)
            
            if os.path.exists(model_path):
                actions_seq = capture_model_manifold_sequence(
                    model_id=model_id, model_path=model_path, display_label=display_label, 
                    env_name=TARGET_ENV, expert_data_path=expert_data_path, 
                    conflict_states=conflict_states, num_samples=NUM_SAMPLES
                )
                if actions_seq is not None:
                    all_models_sequence_actions[model_id] = actions_seq
            else:
                print(f"⚠️ 找不到权重文件，跳过评估: {model_path}")

        # 3. 落盘与批量出图
        if len(all_models_sequence_actions) > 0:
            os.makedirs(data_save_dir, exist_ok=True)
            pkl_path = os.path.join(data_save_dir, 'action_manifold.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(all_models_sequence_actions, f)
            print(f"\n💾 全序列原始评估数据已备份至: {os.path.abspath(pkl_path)}")
            
            num_frames = len(conflict_states)
            frames_to_plot = range(num_frames) if PLOT_ALL_FRAMES else [TARGET_FRAME_IDX]
            
            for step_idx in frames_to_plot:
                if step_idx < 0 or step_idx >= num_frames:
                    print(f"⚠️ 指定的帧 {step_idx} 超出当前序列范围 (0~{num_frames-1})，跳过绘制！")
                    continue
                frame_actions = {mid: all_models_sequence_actions[mid][step_idx] for mid in all_models_sequence_actions}
                plot_action_manifold(frame_actions, models_to_evaluate, save_dir=plot_save_dir, env_name=TARGET_ENV, step_idx=step_idx)
    else:
        print(f"\n⏩ [极速出图模式] 直接加载本地快照数据...")
        print(f"📦 读取路径: {LOAD_PKL_PATH}")
        if os.path.exists(LOAD_PKL_PATH):
            with open(LOAD_PKL_PATH, 'rb') as f:
                all_models_sequence_actions = pickle.load(f)
            
            all_models_sequence_actions = {k: v for k, v in all_models_sequence_actions.items() if k in models_to_evaluate}
            
            if len(all_models_sequence_actions) > 0:
                num_frames = len(list(all_models_sequence_actions.values())[0])
                frames_to_plot = range(num_frames) if PLOT_ALL_FRAMES else [TARGET_FRAME_IDX]
                
                print(f"✅ 数据过滤成功！正在绘制图表...")
                for step_idx in frames_to_plot:
                    if step_idx < 0 or step_idx >= num_frames:
                        print(f"⚠️ 指定的帧 {step_idx} 超出当前序列范围 (0~{num_frames-1})，跳过绘制！")
                        continue
                    frame_actions = {mid: all_models_sequence_actions[mid][step_idx] for mid in all_models_sequence_actions}
                    plot_action_manifold(frame_actions, models_to_evaluate, save_dir=plot_save_dir, env_name=TARGET_ENV, step_idx=step_idx)
                print(f"\n🎨 所选帧图表已全部渲染并保存至: {plot_save_dir}")
            else:
                print("❌ 过滤后没有任何模型数据！请检查您解除注释的模型 ID 是否存在于该 .pkl 文件中。")
        else:
            print(f"❌ 找不到指定的 .pkl 数据文件，请检查 LOAD_PKL_PATH 路径！")