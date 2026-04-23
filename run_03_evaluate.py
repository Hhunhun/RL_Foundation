"""
阶段四：统一模型评估与可视化流水线 (Unified Model Evaluation & Visualization)

此模块是整个自动驾驶项目的“最高裁判法庭”。
为确保公平（控制变量法），所有不同阶段、不同架构（SAC 与 Diff-SAC）的模型，
都将在完全一致的“纯净物理环境（is_eval=True）”中进行测试。
所有训练期的辅助惩罚（如转向限制、舒适度惩罚等）都被关闭，仅考核两项最硬核的指标：
1. 能不能活下来（不撞车）。
2. 能不能开得快（均速高）。
"""

import os
import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# ----------------------------------------------------
# 屏蔽底层第三方库的警告，保持控制台纯净
# ----------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 精准屏蔽 Gymnasium 的录像覆盖警告
warnings.filterwarnings("ignore", message=".*Overwriting existing videos.*")

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# 导入底层组件
from algorithms.sac.sac_agent import SACAgent
from algorithms.diffusion_sac.diff_sac_agent import DiffSACAgent
from core.offline_buffer import MixedReplayBuffer
from envs import create_environment

# 统一设置中文字体，防止 matplotlib 画图时出现乱码 (针对 Windows 系统)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------------------------------
# 🔒 定义全局评估基础种子，并预先生成极度离散的测试种子库
# ----------------------------------------------------
EVAL_BASE_SEED = 52
random.seed(EVAL_BASE_SEED)
np.random.seed(EVAL_BASE_SEED)
torch.manual_seed(EVAL_BASE_SEED)

# 预生成一个庞大的、极度离散的种子库，彻底打破连续整数种子可能带来的环境分布同质化
MASTER_SEED_BANK = [random.randint(0, 9999999) for _ in range(10000)]

def evaluate_single_model(model_id, model_path, display_label, env_name, eval_run_dir, num_episodes=100, record_video=True, expert_data_path=None, max_steps_per_episode=1000):
    """
    对单个模型进行双线程评估：定性录像 (防崩溃) + 定量统计 (出数据)。
    包含智能路由逻辑：根据模型名称自动选择加载 SAC 还是 Diffusion 网络。
    """
    print(f"\n" + "=" * 60)
    print(f" 开始公平评估模型: [{display_label}] (ID: {model_id}) | 测试回合数: {num_episodes}")
    print(f"📁 权重路径: {os.path.abspath(model_path)}")
    print("=" * 60)

    # 智能路由判定：根据 model_id 自动识别算法 (兼容老名字 Diff/diff 和新名字 DM / DH)
    is_diff = "Diff" in model_id or "diff" in model_id or "DM" in model_id or "DH" in model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🚨 核心改造：根据模型类型和评估模式，动态选择算法包装器
    algo_type = "diff" if is_diff else "sac"

    # 1. 探针环境：开一个临时的环境，仅仅是为了读取状态和动作的维度
    dummy_env = create_environment(env_name, is_eval=True, algo=algo_type)
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.shape[0]
    max_action = float(dummy_env.action_space.high[0])
    dummy_env.close()

    # 2. 根据模型类型加载大脑
    if is_diff:
        print(f"🧠 检测到 Diffusion 架构 ({model_id})，正在挂载 DiffSACAgent 与数据归一化器...")
        # Diffusion 模型极其依赖数据归一化。我们通过传入专家数据，建立统一的归一化基准
        buffer = MixedReplayBuffer(expert_data_path=expert_data_path, max_online_size=10, device=device)
        agent = DiffSACAgent(state_dim, action_dim, device=device)
        agent.load_pretrained_actor(model_path)

        # 🚨 [极其核心的修复] 🚨
        # 强行将主网络的权重 100% 覆盖给 EMA 影子网络！
        # 否则因为 EMA 的 decay=0.995，影子网络将保留 99.5% 的初始随机垃圾权重，导致出门就撞车。
        agent.ema_actor.model.load_state_dict(agent.actor.state_dict())
        print("🔧 EMA 权重 100% 同步修复完成，解除随机驾驶锁定！")

    else: # SAC 架构
        print(f"🧠 检测到 SAC 架构 ({model_id})，正在挂载经典 SACAgent...")
        agent = SACAgent(state_dim, action_dim, action_scale=max_action)
        try:
            agent.load_model(model_path) # SACAgent 的 load_model 接受路径
        except Exception as e:
            print(f"❌ 加载模型 {display_label} 失败: {e}")
            return None

    # ==========================================
    # 阶段一：纯粹的视频录制定性环节 (仅录制前 2 局)
    # ==========================================
    if record_video:
        print(f"🎬 [阶段 1] 正在为 [{display_label}] 录制实战视频 (前 2 局)...")
        env_video = create_environment(env_name, is_eval=True, algo=algo_type)
        video_dir = os.path.join(eval_run_dir, "videos") # Simplified: model_name will be in the video file name
        os.makedirs(video_dir, exist_ok=True)
        env_video = RecordVideo(env_video, video_folder=video_dir, name_prefix=f"{display_label.replace(' ', '_')}_eval") # 使用 display_label 作为视频前缀

        for ep in range(2):
            # 🔒 从预生成的离散种子库中抽取种子，保证每次录像面对的路况极具多样性且完全公平
            eval_seed = MASTER_SEED_BANK[ep]
            state, _ = env_video.reset(seed=eval_seed)
            ep_steps = 0
            while True:
                # 动作生成的路由逻辑：Diffusion 需要来回切换数据尺度，SAC 直接输出
                if is_diff:
                    norm_state = buffer.state_normalizer.normalize(state)
                    # explore=False 表示彻底关掉随机探索，拿出最高水平
                    norm_action = agent.select_action(norm_state, sample_steps=5, explore=False)
                    action = buffer.action_normalizer.unnormalize(norm_action)
                    action = np.clip(action, -1.0, 1.0) # 物理保护锁
                else:
                    action = agent.select_action(state, evaluate=True)

                state, reward, terminated, truncated, info = env_video.step(action)
                ep_steps += 1
                ego_speed = info.get("ego_speed_vx", 0.0)

                # 🚨 [核心修复] 为视频录制阶段同步添加 100 步强制截断保护
                # 防止高 Q 权重下的模型 (如 DM6) 发癫加速到 40m/s 冲入虚空导致无限死循环
                if env_name == "merge-v0" and ep_steps >= 100:
                    truncated = True

                print(f"\r├─ 录制 Ep {ep + 1}/2 | Step {ep_steps:3d} | 车速 vx: {ego_speed:5.2f} m/s", end="")
                if terminated or truncated:
                    # 🚨 物理探针：区分出界完赛与真实撞车
                    is_crashed = terminated
                    if env_name == "merge-v0":
                        try:
                            actual_crash = getattr(env_video.unwrapped.vehicle, "crashed", False)
                            is_sideways = abs(env_video.unwrapped.vehicle.heading) > 0.4
                            is_not_on_road = not getattr(env_video.unwrapped.vehicle, "on_road", True)
                            is_reverse = getattr(env_video.unwrapped.vehicle, "speed", 0) < -1.0
                            is_crashed = actual_crash or is_sideways or is_not_on_road or is_reverse
                        except Exception: pass
                    print(f"\n└─ 录像完成: {'💥 撞车/越野' if is_crashed else '🏁 完赛'}")
                    break
        env_video.close()

    # ==========================================
    # 阶段二：最高速的大样本定量评估环节
    # ==========================================
    print(f"\n⚡ [阶段 2] 执行 {num_episodes} 局大样本闭门测试...")
    env_eval = create_environment(env_name, is_eval=True, algo=algo_type) # 再次确认开启纯净评估模式
    metrics = {'rewards': [], 'lengths': [], 'speeds': [], 'crashes': 0}

    for ep in range(num_episodes):
        # 🔒 抽取离散测试种子，通过引入剧烈的初始态波动，彻底粉碎单一种子池导致的过拟合陷阱
        eval_seed = MASTER_SEED_BANK[ep + 10] # 偏移 10，避开上面录像用过的种子
        state, _ = env_eval.reset(seed=eval_seed)
        ep_reward, ep_steps, ep_speeds = 0, 0, []

        while True:
            if is_diff:
                norm_state = buffer.state_normalizer.normalize(state)
                norm_action = agent.select_action(norm_state, sample_steps=5, explore=False)
                action = buffer.action_normalizer.unnormalize(norm_action)
                action = np.clip(action, -1.0, 1.0)
            else:
                action = agent.select_action(state, evaluate=True)

            state, reward, terminated, truncated, info = env_eval.step(action)
            ep_reward += reward
            ep_steps += 1
            ep_speeds.append(info.get("ego_speed_vx", 0.0))
            
            # [核心修复] 强制截断，与训练脚本 (main_merge.py) 的 100 步“完赛”标准保持一致
            # 解决了评估时没有“完赛”出口，导致最终必然“被判负”的问题。 
            # 仅对 merge-v0 生效，防止影响 highway-v0 等其他环境的评估。
            if env_name == "merge-v0" and ep_steps >= 100:
                truncated = True

            # [新增] 实时终端可视化，对齐训练体验
            ego_speed = info.get("ego_speed_vx", 0.0)
            print(f"\r├─ 定量测试 Ep {ep + 1}/{num_episodes} | Step {ep_steps:3d} | 车速 vx: {ego_speed:5.2f} m/s | 单步奖励: {reward:.3f}", end="")
            
            if terminated or truncated or ep_steps >= max_steps_per_episode: # 当任何终止条件满足时，结束本局并记录数据
                metrics['rewards'].append(ep_reward)
                metrics['lengths'].append(ep_steps)
                metrics['speeds'].append(np.mean(ep_speeds))
                
                # 🚨 [核心修复] 物理探针直达底层，精准剔除“极速完赛”带来的 Fake Terminated
                is_crashed = terminated
                if env_name == "merge-v0":
                    try:
                        actual_crash = getattr(env_eval.unwrapped.vehicle, "crashed", False)
                        is_sideways = abs(env_eval.unwrapped.vehicle.heading) > 0.4
                        is_not_on_road = not getattr(env_eval.unwrapped.vehicle, "on_road", True)
                        is_reverse = getattr(env_eval.unwrapped.vehicle, "speed", 0) < -1.0
                        is_crashed = actual_crash or is_sideways or is_not_on_road or is_reverse
                    except Exception: pass

                if is_crashed:
                    metrics['crashes'] += 1 # 统计事故率
                
                # [优化] 打印更详细的局末总结
                status = "💥 撞车/越野" if is_crashed else "🏁 完赛"
                print(f"\n└─ Episode {ep + 1} 结束: 存活 {ep_steps} 步 | 总回报: {ep_reward:.2f} | 结局: {status}")
                break
    env_eval.close()

    # 3. 结算统计学指标
    results = {
        'mean_reward': np.mean(metrics['rewards']),
        'std_reward': np.std(metrics['rewards']),
        'survival_rate': (num_episodes - metrics['crashes']) / num_episodes * 100,
        'mean_speed': np.mean(metrics['speeds']),
        'raw_rewards': metrics['rewards']
    }

    print(f"\n📊 [{display_label}] 评估报告:")
    print(f"   平均累计奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"   存活率(完赛率): {results['survival_rate']:.1f}%")
    print(f"   平均车速: {results['mean_speed']:.2f} m/s")

    return results


def save_metrics_to_csv(all_results, models_to_evaluate, save_dir):
    """
    将量化指标保存为 CSV 文件，便于论文表格制作。
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'summary_metrics.csv')
    headers = ['模型版本 (Model)', '平均累计奖励 (Mean Reward)', '策略方差/标准差 (Std Reward)',
               '存活率 (Survival Rate %)', '平均纵向速度 (Mean Speed m/s)']

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for model_id, res in all_results.items():
            display_label = models_to_evaluate[model_id]["display_name"] # 从原始配置中获取 display_name
            writer.writerow([display_label, f"{res['mean_reward']:.2f}", f"{res['std_reward']:.2f}",
                             f"{res['survival_rate']:.1f}", f"{res['mean_speed']:.2f}"])
    print(f"\n💾 量化指标数据已保存至 CSV: {os.path.abspath(csv_path)}")


def plot_comparisons(all_results, models_to_evaluate, save_dir):
    """
    自动生成学术级对比箱线图和柱状图。
    包含四张核心图表：回报箱线图、策略方差柱状图、存活率柱状图、均速柱状图。
    """
    os.makedirs(save_dir, exist_ok=True)
    model_ids = list(all_results.keys())
    display_labels = [models_to_evaluate[mid]["display_name"] for mid in model_ids]
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_ids)))

    # ----------------------------------------------------
    # 图 1：累计奖励箱线图 (新增均值文本标注)
    # ----------------------------------------------------
    plt.figure(figsize=(12, 7))
    reward_data = [all_results[mid]['raw_rewards'] for mid in model_ids]
    # showfliers=False 会隐藏离群点，使得主流分布清晰可见
    plt.boxplot(reward_data, labels=display_labels, showmeans=True, showfliers=False) # 使用 display_labels
    plt.title('规控策略演进与消融实验对比 (Cumulative Reward)', fontsize=14, fontweight='bold')
    plt.ylabel('Episode Reward (Outliers Hidden)', fontsize=12)
    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 🆕 新增：在绿色均值三角形旁边标注具体的数值
    means = [all_results[m]['mean_reward'] for m in model_ids]
    for i, mean_val in enumerate(means): # 这里的 models 应该改为 model_ids
        # i + 1 是因为 boxplot 的 x 轴刻度是从 1 开始的
        plt.text(i + 1.05, mean_val, f'{mean_val:.1f}', va='center', ha='left',
                 color='green', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01_reward_boxplot.png'), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # 🆕 图 2：策略方差/标准差柱状图 (越低代表模型越稳定)
    # ----------------------------------------------------
    plt.figure(figsize=(12, 7))
    std_rewards = [all_results[m]['std_reward'] for m in model_ids]
    bars_std = plt.bar(display_labels, std_rewards, color=colors, alpha=0.8) # 使用 display_labels
    plt.title('规控策略稳定性对比 (Standard Deviation of Reward)', fontsize=14, fontweight='bold') # 标题不变
    plt.ylabel('Reward Std. Dev (Lower is Better)', fontsize=12)
    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 在柱子上标注具体数字
    for bar in bars_std:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '02_reward_variance_bar.png'), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # 图 3：存活率柱状图
    # ----------------------------------------------------
    plt.figure(figsize=(12, 7))
    survival_rates = [all_results[m]['survival_rate'] for m in model_ids]
    bars_surv = plt.bar(display_labels, survival_rates, color=colors, alpha=0.9) # 使用 display_labels
    plt.title('规控策略存活率对比 (Survival Rate)', fontsize=14, fontweight='bold') # 标题不变
    plt.ylabel('Survival Rate (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 在柱子上标注具体数字
    for bar in bars_surv:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03_survival_rate_bar.png'), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # 🆕 图 4：平均纵向速度柱状图 (修复自适应 Y 轴)
    # ----------------------------------------------------
    plt.figure(figsize=(12, 7))
    mean_speeds = [all_results[m]['mean_speed'] for m in model_ids]
    bars_speed = plt.bar(display_labels, mean_speeds, color=colors, alpha=0.8) # 使用 display_labels
    plt.title('规控策略平均纵向速度对比 (Mean Longitudinal Speed)', fontsize=14, fontweight='bold') # 标题不变
    plt.ylabel('Mean Speed (m/s)', fontsize=12)

    # [修复] 动态设定 Y 轴下限。之前写死了 20.0，导致 merge 环境 (均速 17 左右) 的柱子完全不可见！
    # 现在改为根据实际最小速度动态下潜，保证柱状图完整显示，同时保留差异放大效果。
    min_speed = min(mean_speeds)
    max_speed = max(mean_speeds)
    y_min = max(0.0, min_speed - 3.0) # 往下探 3m/s，但绝不低于 0
    plt.ylim(y_min, max_speed + 2.0)

    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 在柱子上标注具体数字
    for bar in bars_speed:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, f'{yval:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '04_mean_speed_bar.png'), dpi=300)
    plt.close()

    print(f"📈 4 张高质量对比图表已全部保存至: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    # ==========================================
    # 终端交互：选择评估环境
    # ==========================================
    print("🤖 欢迎使用统一模型评估终端")
    print("==========================================")
    print("[H] Highway 环境 (highway-v0)")
    print("[M] Merge 环境 (merge-v0)")
    env_choice = input("👉 请选择评估环境 (H 或 M，默认 H): ").strip().upper()
    TARGET_ENV = "merge-v0" if env_choice == 'M' else "highway-v0"
    print(f"✅ 已锁定评估环境: {TARGET_ENV}")
    print("==========================================")

    # ==========================================
    # 实验配置区：模型演进与消融实验大乱斗
    # ==========================================
    # 🚨 动态配置：根据选择的环境切换专家数据集和待评估模型列表
    if TARGET_ENV == "merge-v0":
        # 🚨 注意：请将这里的路径替换为您真实的 merge 专家数据和模型路径！
        # 评估 Diff-SAC 时，必须提供专家数据集路径以初始化归一化器 (Normalizer)
        EXPERT_DATA_PATH = "data/expert_data/merge-v0/dataset_base_20260422_014135/expert_transitions.npz"
        models_to_evaluate = {
            # === 第一期基础实验（加入TTC） ===
            #"M1": {"path": "outputs/merge-v0/models/SAC_M1_Base_Merge_20260420_150323/sac_merge_final.pth", "display_name": "M1 基础生存"},
            #"M2": {"path": "outputs/merge-v0/models/SAC_M2_Efficient_Smooth_20260420_154007/sac_merge_final.pth", "display_name": "M2 高效平滑"},
            #"M3": {"path": "outputs/merge-v0/models/SAC_M3_Aggressive_Gap_Finding_20260420_162217/sac_merge_final.pth", "display_name": "M3 激进寻隙"},
            #"M4": {"path": "outputs/merge-v0/models/SAC_M4_Safety_First_20260420_170911/sac_merge_final.pth", "display_name": "M4 安全至上"},
            #"M5": {"path": "outputs/merge-v0/models/SAC_M5_Patient_Merger_20260420_220108/sac_merge_final.pth", "display_name": "M5 耐心等待"},
            #"M6": {"path": "outputs/merge-v0/models/SAC_M6_Extreme_Penalty_20260420_232207/sac_merge_final.pth", "display_name": "M6 极限死刑"},
            #"M7": {"path": "outputs/merge-v0/models/SAC_M7_Smooth_Marathon_20260421_003822/sac_merge_final.pth", "display_name": "M7 平滑马拉松"},
            #"M8": {"path": "outputs/merge-v0/models/SAC_M8_Ultimate_Merge_20260421_023258/sac_merge_final.pth", "display_name": "M8 终极汇入"},

            # === 第一期 diff-SAC 实验 ===
            #"DM1": {"path": "outputs/merge-v0/models/DiffSAC_DM1_Zero_Q_20260422_020015/online_finetune/diff_sac_ep400.pth", "display_name": "DM1 纯模仿"},
            #"DM2": {"path": "outputs/merge-v0/models/DiffSAC_DM2_Micro_Q_20260422_021730/online_finetune/diff_sac_ep400.pth", "display_name": "DM2 保守试探"},
            "DM3": {"path": "outputs/merge-v0/models/DiffSAC_DM3_Gentle_Q_20260422_023433/online_finetune/diff_sac_ep400.pth", "display_name": "DM3 微弱提速"},
            "DM4": {"path": "outputs/merge-v0/models/DiffSAC_DM4_Standard_Q_20260422_025352/online_finetune/diff_sac_ep400.pth", "display_name": "DM4 激进提速"},

            # === 第二期 diff-SAC 实验 ===
            #"DM5": {"path": "outputs/merge-v0/models/DiffSAC_DM5_Mild_Transition_20260423_030738/online_finetune/diff_sac_ep400.pth", "display_name": "DM5 破冰试探"},
            #"DM6": {"path": "outputs/merge-v0/models/DiffSAC_DM6_Moderate_Override_20260423_032715/online_finetune/diff_sac_ep400.pth", "display_name": "DM6 中度干预"},
            #"DM7": {"path": "outputs/merge-v0/models/DiffSAC_DM7_Strong_Override_20260423_033850/online_finetune/diff_sac_ep400.pth", "display_name": "DM7 强力干预"},
            #"DM8": {"path": "outputs/merge-v0/models/DiffSAC_DM8_Extreme_Domination_20260423_034901/online_finetune/diff_sac_ep400.pth", "display_name": "DM8 极限干预"},

            # === 第三期 diff-SAC 实验 (基于 100% M4 专家数据) ===
            "DM9": {"path": "outputs/merge-v0/models/DiffSAC_DM9_M4_Prior_Only_20260423_184425/online_finetune/diff_sac_ep400.pth", "display_name": "DM9 M4纯模仿"},
            "DM10": {"path": "outputs/merge-v0/models/DiffSAC_DM10_M4_Standard_Q_20260423_190251/online_finetune/diff_sac_ep400.pth", "display_name": "DM10 M4弱度干预"},
            "DM11": {"path": "outputs/merge-v0/models/DiffSAC_DM11_M4_Strong_Q_20260423_191253/online_finetune/diff_sac_ep400.pth", "display_name": "DM11 M4强力干预"},
            "DM12": {"path": "outputs/merge-v0/models/DiffSAC_DM12_M4_Extreme_Q_20260423_192105/online_finetune/diff_sac_ep400.pth", "display_name": "DM12 M4极限干预"},
        }
    else: # highway-v0
        EXPERT_DATA_PATH = "data/expert_data/highway-v0/dataset_smart_mixed_90_10_20260413_031136/expert_transitions_smart_90_10.npz"
        models_to_evaluate = {
        #"H1": {"path": "outputs/models/SAC_H1_20260329_150543/sac_highway_final.pth", "display_name": "H1 无约束 SAC"},
        #"H2": {"path": "outputs/models/SAC_H2_20260329_185751/sac_highway_final.pth", "display_name": "H2 越野飙车 SAC"},
        #"H3": {"path": "outputs/models/SAC_H3_20260330_010914/sac_highway_final.pth", "display_name": "H3 LQR 欠拟合 SAC"},
        #"H5": {"path": "outputs/models/SAC_H5_20260330_135449/sac_highway_final.pth", "display_name": "H5 安全保守 SAC"},
        #"H6": {"path": "outputs/models/SAC_H6_20260330_213300/sac_highway_final.pth", "display_name": "H6 高效超车 SAC"},

        #"DH1": {"path": "outputs/models/highway_DiffSAC_20260405_031920/diff_sac_ep400.pth", "display_name": "DH1 微弱 Q 引导"},
        #"DH2": {"path": "outputs/models/DiffSAC_DH2_20260405_065603/diff_sac_ep400.pth", "display_name": "DH2 标准 Q 引导"},
        #"DH3": {"path": "outputs/models/DiffSAC_DH3_20260405_101704/diff_sac_ep400.pth", "display_name": "DH3 强力 Q 引导"},
        #"DH4": {"path": "outputs/models/DiffSAC_DH4_20260405_141352/diff_sac_ep500.pth", "display_name": "DH4 降学习率长跑"},

        #"DH5": {"path": "outputs/models/DiffSAC_DH5_20260406_023023/diff_sac_ep400.pth", "display_name": "DH5 极微引导"},
        #"DH6": {"path": "outputs/models/DiffSAC_DH6_20260406_040052/diff_sac_ep400.pth", "display_name": "DH6 铁壁底座"},
        #"DH7": {"path": "outputs/models/DiffSAC_DH7_20260406_052938/diff_sac_ep500.pth", "display_name": "DH7 冰封微调"},
        #"DH8": {"path": "outputs/models/DiffSAC_DH8_20260406_064236/diff_sac_ep400.pth", "display_name": "DH8 零引导对照"},

        #"DH9": {"path": "outputs/models/DiffSAC_DH9_20260406_153903/diff_sac_ep500.pth", "display_name": "DH9 终极防御底座"},
        #"DH10": {"path": "outputs/models/DiffSAC_DH10_20260406_172128/diff_sac_ep500.pth", "display_name": "DH10 加速冰封"},
        #"DH11": {"path": "outputs/models/DiffSAC_DH11_20260406_211102/diff_sac_ep400.pth", "display_name": "DH11 极限微丝引导"},
        #"DH12": {"path": "outputs/models/DiffSAC_DH12_20260407_013017/diff_sac_ep800.pth", "display_name": "DH12 冰封马拉松"},

        #"DH13": {"path": "outputs/models/DiffSAC_DH13_20260407_071544/diff_sac_ep400.pth", "display_name": "DH13 终极无坚不摧"},
        #"DH14": {"path": "outputs/models/DiffSAC_DH14_20260407_110205/diff_sac_ep400.pth", "display_name": "DH14 厚甲利刃"},
        #"DH15": {"path": "outputs/models/DiffSAC_DH15_20260407_140041/diff_sac_ep400.pth", "display_name": "DH15 深度 BC 对照"},
        #"DH16": {"path": "outputs/models/DiffSAC_DH16_20260407_170756/diff_sac_ep600.pth", "display_name": "DH16 微丝引导马拉松"},

        #"DH17": {"path": "outputs/models/DiffSAC_DH17_20260408_141756/diff_sac_ep400.pth", "display_name": "DH17 混合 BC 对照"},
        #"DH18": {"path": "outputs/models/DiffSAC_DH18_20260408_155039/diff_sac_ep400.pth", "display_name": "DH18 混合微丝引导"},
        #"DH19": {"path": "outputs/models/DiffSAC_DH19_20260408_190001/diff_sac_ep400.pth", "display_name": "DH19 混合厚底座"},
        #"DH20": {"path": "outputs/models/DiffSAC_DH20_20260408_224554/diff_sac_ep600.pth", "display_name": "DH20 混合马拉松"},

        #"DH21": {"path": "outputs/models/DiffSAC_DH21_20260410_031928/diff_sac_ep600.pth", "display_name": "DH21 黄金比例马拉松"},
        #"DH22": {"path": "outputs/models/DiffSAC_DH22_20260413_031458/diff_sac_ep400.pth", "display_name": "DH22 智能 BC 对照"},
        #"DH23": {"path": "outputs/models/DiffSAC_DH23_20260413_041749/diff_sac_ep400.pth", "display_name": "DH23 智能微丝引导"},
        #"DH24": {"path": "outputs/models/DiffSAC_DH24_20260413_052105/diff_sac_ep400.pth", "display_name": "DH24 智能厚底座"},

        #"DH25": {"path": "outputs/models/DiffSAC_DH25_20260413_062853/diff_sac_ep600.pth", "display_name": "DH25 智能混合马拉松"},

        #"DH8_Run2": {"path": "outputs/models/DiffSAC_DH8_Run2_20260406_080556/diff_sac_ep500.pth", "display_name": "DH8_Run2 零引导对照"},
        #"DH8_Run3": {"path": "outputs/models/DiffSAC_DH8_Run3_20260406_095201/diff_sac_ep500.pth", "display_name": "DH8_Run3 零引导对照"},
        }


    # 大样本测试局数，100 局是黄金免检样本量
    NUM_EVAL_EPISODES = 100

    # ==========================================
    # 动态文件夹命名逻辑：自动提取参与评估的模型简称
    # ==========================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 因为字典的 key 已经精简为 M/DM 格式，直接拼接即可
    version_tags = list(models_to_evaluate.keys())
    versions_str = "_".join(version_tags)
    eval_run_name = f"[{versions_str}]_{timestamp}" # Simplified, as TARGET_ENV is now a parent folder

    # 构建最终的保存路径
    eval_run_dir = os.path.join("outputs", TARGET_ENV, "eval_results", eval_run_name)
    plot_save_dir = os.path.join(eval_run_dir, "plots") # These are relative to eval_run_dir, so no change needed
    data_save_dir = os.path.join(eval_run_dir, "data") # These are relative to eval_run_dir, so no change needed

    # 启动批量测评
    all_results = {}
    for model_id, model_config in models_to_evaluate.items(): # 遍历新的字典结构
        model_path = model_config["path"]
        display_label = model_config["display_name"]
        if os.path.exists(model_path):
            res = evaluate_single_model(
                model_id=model_id, # 传递内部 ID
                model_path=model_path,
                display_label=display_label, # 传递显示名称
                env_name=TARGET_ENV,
                eval_run_dir=eval_run_dir,
                num_episodes=NUM_EVAL_EPISODES,
                expert_data_path=EXPERT_DATA_PATH
            )
            if res:
                all_results[model_id] = res # 结果仍然用 model_id 作为键
        else:
            print(f"⚠️ 找不到权重文件，跳过评估: {model_path}")

    # 出图与落盘
    if len(all_results) > 0:
        save_metrics_to_csv(all_results, models_to_evaluate, save_dir=data_save_dir) # 传入 models_to_evaluate
        plot_comparisons(all_results, models_to_evaluate, save_dir=plot_save_dir) # 传入 models_to_evaluate