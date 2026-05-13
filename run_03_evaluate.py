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
import seaborn as sns
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

def set_publication_style():
    """
    全局学术期刊图表样式配置滤镜，支持高度个性化排版定制。
    """
    custom_params = {
        # --- 字体配置 (注意中文兼容性) ---
        "font.family": "sans-serif",      # 默认字体族 (若需纯英文论文可改为 "serif")
        "font.sans-serif": ["SimSun", "SimHei", "PingFang SC", "Microsoft YaHei", "sans-serif"], # 中文后备字体栈，优先使用宋体(SimSun)
        "font.serif": ["SimSun", "Times New Roman"],# 衬线字体也加入宋体防乱码
        "font.size": 12,                  # 全局基础字号
        
        # --- 轴与标签字号配置 ---
        "axes.titlesize": 14,             # 图表标题字号
        "axes.labelsize": 12,             # 坐标轴标签字号
        "xtick.labelsize": 10,            # X轴刻度字号
        "ytick.labelsize": 10,            # Y轴刻度字号
        "legend.fontsize": 10,            # 图例字号 (如需使用图例)
        
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
    is_diff = "Diff" in model_id or "diff" in model_id or "DM" in model_id or "DH" in model_id or "DR" in model_id
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
                elif env_name == "racetrack-v0" and ep_steps >= 500:
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
                    elif env_name == "racetrack-v0":
                        try:
                            actual_crash = getattr(env_video.unwrapped.vehicle, "crashed", False)
                            is_not_on_road = not getattr(env_video.unwrapped.vehicle, "on_road", True)
                            is_crashed = actual_crash or is_not_on_road or info.get("crashed", False)
                        except Exception: pass
                    print(f"\n└─ 录像完成: {'💥 撞车/越野' if is_crashed else '🏁 完赛'}")
                    break
        env_video.close()

    # ==========================================
    # 阶段二：最高速的大样本定量评估环节
    # ==========================================
    print(f"\n⚡ [阶段 2] 执行 {num_episodes} 局大样本闭门测试...")
    env_eval = create_environment(env_name, is_eval=True, algo=algo_type) # 再次确认开启纯净评估模式
    metrics = {'rewards': [], 'lengths': [], 'speeds': [], 'crashes': 0, 'actions': []}

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

            # [新增] 收集动作数据用于绘制 4.3.3 节的二维分布散点图
            metrics['actions'].append(action.copy())
            
            state, reward, terminated, truncated, info = env_eval.step(action)
            ep_reward += reward
            ep_steps += 1
            ep_speeds.append(info.get("ego_speed_vx", 0.0))
            
            # [核心修复] 强制截断，与训练脚本 (main_merge.py) 的 100 步“完赛”标准保持一致
            # 解决了评估时没有“完赛”出口，导致最终必然“被判负”的问题。 
            # 仅对 merge-v0 生效，防止影响 highway-v0 等其他环境的评估。
            if env_name == "merge-v0" and ep_steps >= 100:
                truncated = True
            elif env_name == "racetrack-v0" and ep_steps >= 500:
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
                elif env_name == "racetrack-v0":
                    try:
                        actual_crash = getattr(env_eval.unwrapped.vehicle, "crashed", False)
                        is_not_on_road = not getattr(env_eval.unwrapped.vehicle, "on_road", True)
                        is_crashed = actual_crash or is_not_on_road or info.get("crashed", False)
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
        'raw_rewards': metrics['rewards'],
        'actions': np.array(metrics['actions']) # 暴露出全部的动作张量
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
    # 激活全局学术样式滤镜
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)
    model_ids = list(all_results.keys())
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
    ]
    colors = sns.color_palette(academic_colors, n_colors=len(model_ids))

    def _save_and_close_fig(filename_base):
        """内部辅助函数：消除重复的图表保存代码"""
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{filename_base}.png'), dpi=300)
        # 暂时关闭 PDF 导出，以提升出图速度，后续需要写论文时可取消注释
        # plt.savefig(os.path.join(save_dir, f'{filename_base}.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

    # ----------------------------------------------------
    # 图 1：累计奖励箱线图 (新增均值文本标注)
    # ----------------------------------------------------
    plt.figure(figsize=(8.0, 6.0)) # 放大画幅，保持 4:3 比例，使 12 号字体显示更舒展
    reward_data = [all_results[mid]['raw_rewards'] for mid in model_ids]
    
    # 美学升级：使用实体填充的箱线图，定制均值点和中位数线
    bplot = plt.boxplot(reward_data, labels=display_labels, showmeans=True, showfliers=False, patch_artist=True,
                        boxprops=dict(color='black', linewidth=1.2),
                        capprops=dict(color='black', linewidth=1.2),
                        whiskerprops=dict(color='black', linewidth=1.2),
                        medianprops=dict(color='firebrick', linewidth=2.0),
                        meanprops=dict(marker='^', markeredgecolor='green', markerfacecolor='green', markersize=8))
    
    # 为每个箱体填上对应的调色盘颜色
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    plt.title('规控策略演进与消融实验对比 (Cumulative Reward)')
    plt.ylabel('回合累计奖励')
    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 🆕 新增：在绿色均值三角形旁边标注具体的数值
    means = [all_results[m]['mean_reward'] for m in model_ids]
    for i, mean_val in enumerate(means): # 这里的 models 应该改为 model_ids
        # i + 1 是因为 boxplot 的 x 轴刻度是从 1 开始的
        plt.text(i + 1.05, mean_val, f'{mean_val:.1f}', va='center', ha='left',
                 color='green', fontsize=10, fontweight='bold')

    _save_and_close_fig('01_reward_boxplot')

    # ----------------------------------------------------
    # 🆕 图 2：策略方差/标准差柱状图 (越低代表模型越稳定)
    # ----------------------------------------------------
    plt.figure(figsize=(8.0, 6.0))
    std_rewards = [all_results[m]['std_reward'] for m in model_ids]
    bars_std = plt.bar(display_labels, std_rewards, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2) # 增加物理描边
    plt.title('规控策略稳定性对比 (Standard Deviation of Reward)')
    plt.ylabel('奖励标准差')
    
    # [自适应 Y 轴] 顶部预留 15% 的动态空间，确保文本绝对不会出界
    max_std = max(std_rewards) if len(std_rewards) > 0 else 1.0
    plt.ylim(0, max_std * 1.15)
    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 在柱子上标注具体数字
    for bar in bars_std:
        yval = bar.get_height()
        # 文本高度偏移也改为图表量级的 2%，实现动态自适应
        plt.text(bar.get_x() + bar.get_width() / 2, yval + max_std * 0.02, f'{yval:.1f}', ha='center', va='bottom', fontsize=10)

    _save_and_close_fig('02_reward_variance_bar')

    # ----------------------------------------------------
    # 图 3：存活率柱状图
    # ----------------------------------------------------
    plt.figure(figsize=(8.0, 6.0))
    survival_rates = [all_results[m]['survival_rate'] for m in model_ids]
    bars_surv = plt.bar(display_labels, survival_rates, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    plt.title('规控策略存活率对比 (Survival Rate)')
    plt.ylabel('存活率 (%)')
    plt.ylim(0, 110) # 扩大顶部留白，防止 100.0% 标签被切角
    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 在柱子上标注具体数字
    for bar in bars_surv:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    _save_and_close_fig('03_survival_rate_bar')

    # ----------------------------------------------------
    # 🆕 图 4：平均纵向速度柱状图 (修复自适应 Y 轴)
    # ----------------------------------------------------
    plt.figure(figsize=(8.0, 6.0))
    mean_speeds = [all_results[m]['mean_speed'] for m in model_ids]
    bars_speed = plt.bar(display_labels, mean_speeds, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    plt.title('规控策略平均纵向速度对比 (Mean Longitudinal Speed)')
    plt.ylabel('平均纵向速度 (m/s)')

    # [重构自适应缩放] 根据数据的真实极差动态计算缩放边界，确保完美居中且不过度裁剪
    min_speed = min(mean_speeds) if len(mean_speeds) > 0 else 0.0
    max_speed = max(mean_speeds) if len(mean_speeds) > 0 else 1.0
    y_range = max_speed - min_speed
    margin = y_range * 0.15 if y_range > 0 else max_speed * 0.15
    
    y_min = max(0.0, min_speed - margin - 1.0)
    y_max = max_speed + margin + 1.0
    plt.ylim(y_min, y_max)

    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 在柱子上标注具体数字
    for bar in bars_speed:
        yval = bar.get_height()
        # 文本高度也根据动态域按比例抬升
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (y_max - y_min) * 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    _save_and_close_fig('04_mean_speed_bar')

    # ----------------------------------------------------
    # [准备子图布局] 动态计算最优网格排列 (如 1x3, 2x2, 2x3, 2x4)
    # ----------------------------------------------------
    n_models = len(model_ids)
    if n_models <= 3:
        nrows, ncols = 1, max(1, n_models)
    elif n_models == 4:
        nrows, ncols = 2, 2
    elif n_models in [5, 6]:
        nrows, ncols = 2, 3
    else:
        ncols = 4
        nrows = int(np.ceil(n_models / ncols))
        
    subplot_width = 4.0
    subplot_height = 4.0
    
    # ----------------------------------------------------
    # 🆕 图 5：动作分布 - 极小散点网格图 (Scatter Grid)
    # ----------------------------------------------------
    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_width * ncols, subplot_height * nrows), sharex=True, sharey=True)
    axes_flat = [axes] if nrows * ncols == 1 else axes.flatten()
    
    for i in range(nrows * ncols):
        ax = axes_flat[i]
        if i < n_models:
            mid = model_ids[i]
            color = colors[i]
            label = display_labels[i]
            
            actions = all_results[mid].get('actions', np.array([]))
            if len(actions) > 0:
                actions = actions.reshape(actions.shape[0], -1)
                if actions.shape[1] >= 2:
                    # 策略：极小 Size + 极低 Alpha
                    ax.scatter(actions[:, 1], actions[:, 0], color=color, alpha=0.2, s=10, edgecolors='none')
                    
            ax.set_title(label, color=color, fontweight='bold')
            if i % ncols == 0:
                ax.set_ylabel('纵向控制 / 加减速')
            # 处于最底层，或者正下方是空位的图表，强制显示 X 轴标签和刻度
            if i >= (nrows - 1) * ncols or (i + ncols >= n_models):
                ax.set_xlabel('横向控制 / 转向')
                ax.xaxis.set_tick_params(labelbottom=True)
                
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.5)
        else:
            ax.axis('off')
            
    _save_and_close_fig('05_action_scatter_grid')

    # ----------------------------------------------------
    # 🆕 图 6：动作分布 - 核密度分布网格图 (KDE Grid)
    # ----------------------------------------------------
    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_width * ncols, subplot_height * nrows), sharex=True, sharey=True)
    axes_flat = [axes] if nrows * ncols == 1 else axes.flatten()
    
    for i in range(nrows * ncols):
        ax = axes_flat[i]
        if i < n_models:
            mid = model_ids[i]
            color = colors[i]
            label = display_labels[i]
            
            actions = all_results[mid].get('actions', np.array([]))
            if len(actions) > 0:
                actions = actions.reshape(actions.shape[0], -1)
                if actions.shape[1] >= 2:
                    x_plot, y_plot = actions[:, 1], actions[:, 0]
                    # 防止数万点导致 KDE 渲染卡死，限流下采样
                    if len(x_plot) > 10000:
                        idx = np.random.choice(len(x_plot), 10000, replace=False)
                        x_plot, y_plot = x_plot[idx], y_plot[idx]
                        
                    try:
                        # fill=True: 绘制如热力图般渐变的实心分布区域；thresh=0.05: 隐藏边缘极低密度的孤点
                        sns.kdeplot(x=x_plot + np.random.normal(0, 1e-5, size=x_plot.shape), 
                                    y=y_plot + np.random.normal(0, 1e-5, size=y_plot.shape), 
                                    color=color, fill=True, alpha=0.8, thresh=0.05, levels=10, ax=ax)
                    except Exception:
                        pass
                    
            ax.set_title(label, color=color, fontweight='bold')
            if i % ncols == 0:
                ax.set_ylabel('纵向控制 / 加减速')
            if i >= (nrows - 1) * ncols or (i + ncols >= n_models):
                ax.set_xlabel('横向控制 / 转向')
                ax.xaxis.set_tick_params(labelbottom=True)
                
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.5)
        else:
            ax.axis('off')
            
    _save_and_close_fig('06_action_kde_grid')

    # ----------------------------------------------------
    # 图 7：动作分布 - 六边形分箱网格图 (Hexbin Grid)
    # ----------------------------------------------------
    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_width * ncols, subplot_height * nrows), sharex=True, sharey=True)
    axes_flat = [axes] if nrows * ncols == 1 else axes.flatten()
    
    for i in range(nrows * ncols):
        ax = axes_flat[i]
        if i < n_models:
            mid = model_ids[i]
            color = colors[i]
            label = display_labels[i]
            
            actions = all_results[mid].get('actions', np.array([]))
            if len(actions) > 0:
                actions = actions.reshape(actions.shape[0], -1)
                if actions.shape[1] >= 2:
                    # gridsize 控制蜂窝细度，mincnt=1 去除空白区域，cmap 选用美观的深海/火箭色系
                    ax.hexbin(actions[:, 1], actions[:, 0], gridsize=30, cmap='mako_r', mincnt=1, alpha=0.9)
                    
            ax.set_title(label, color=color, fontweight='bold')
            if i % ncols == 0:
                ax.set_ylabel('纵向控制 / 加减速')
            if i >= (nrows - 1) * ncols or (i + ncols >= n_models):
                ax.set_xlabel('横向控制 / 转向')
                ax.xaxis.set_tick_params(labelbottom=True)
                
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.5)
        else:
            ax.axis('off')
            
    _save_and_close_fig('07_action_hexbin_grid')

    print(f"📈 7 张高质量对比图表已全部保存至: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    # ==========================================
    # 终端交互：选择评估环境
    # ==========================================
    print("🤖 使用统一模型评估终端")
    print("==========================================")
    print("[H] Highway 环境 (highway-v0)")
    print("[M] Merge 环境 (merge-v0)")
    print("[R] Racetrack 环境 (racetrack-v0)")
    env_choice = input("👉 请选择评估环境 (H, M 或 R，默认 H): ").strip().upper()
    if env_choice == 'M':
        TARGET_ENV = "merge-v0"
    elif env_choice == 'R':
        TARGET_ENV = "racetrack-v0"
    else:
        TARGET_ENV = "highway-v0"
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
            # === 第一期 SAC 消融矩阵 ===
            "M01": {"path": "outputs/merge-v0/models/SAC_M01_Base_Merge_20260511_042953/sac_merge_final.pth", "display_name": "M01 基础生存"},
            "M02": {"path": "outputs/merge-v0/models/SAC_M02_Efficient_Smooth_20260420_154007/sac_merge_final.pth", "display_name": "M02 高效平滑"},
            "M03": {"path": "outputs/merge-v0/models/SAC_M03_Aggressive_Gap_Finding_20260420_162217/sac_merge_final.pth", "display_name": "M03 激进寻隙"},
            "M04": {"path": "outputs/merge-v0/models/SAC_M04_Safety_First_20260420_170911/sac_merge_final.pth", "display_name": "M04 安全至上"},
            "M05": {"path": "outputs/merge-v0/models/SAC_M05_Patient_Merger_20260420_220108/sac_merge_final.pth", "display_name": "M05 耐心等待"},
            "M06": {"path": "outputs/merge-v0/models/SAC_M06_Extreme_Penalty_20260420_232207/sac_merge_final.pth", "display_name": "M06 极限死刑"},
            "M07": {"path": "outputs/merge-v0/models/SAC_M07_Smooth_Marathon_20260421_003822/sac_merge_final.pth", "display_name": "M07 平滑马拉松"},
            "M08": {"path": "outputs/merge-v0/models/SAC_M08_Ultimate_Merge_20260421_023258/sac_merge_final.pth", "display_name": "M08 终极汇入"},

            # === 第一期 diff-SAC 实验 ===
            #"DM01": {"path": "outputs/merge-v0/models/DiffSAC_DM01_Pure_BC_20260511_135709/online_finetune/diff_sac_final.pth", "display_name": "DM01 纯 BC 克隆"},
            #"DM02": {"path": "outputs/merge-v0/models/DiffSAC_DM02_Micro_Q_20260511_152002/online_finetune/diff_sac_final.pth", "display_name": "DM02 微引导"},
            #"DM03": {"path": "outputs/merge-v0/models/DiffSAC_DM03_Standard_Q_20260511_170511/online_finetune/diff_sac_final.pth", "display_name": "DM03 标准引导"},
            #"DM04": {"path": "outputs/merge-v0/models/DiffSAC_DM04_Strong_Q_20260511_183810/online_finetune/diff_sac_final.pth", "display_name": "DM04 强力干预"},

            # === 第二期 diff-SAC 混合专家实验 ===

        }
    elif TARGET_ENV == "racetrack-v0":
        #EXPERT_DATA_PATH = "data/expert_data/racetrack-v0/dataset_R05_mode1_20260506_011817/expert_transitions.npz"
        EXPERT_DATA_PATH = "data/expert_data/racetrack-v0/dataset_mixed_0.8R05_0.2R01_20260506_142446/expert_transitions_mixed_0.8R05_0.2R01.npz"
        models_to_evaluate = {
            # === 第一期 SAC 消融矩阵 ===
            "R01": {"path": "outputs/racetrack-v0/models/SAC_R01_SAC_Baseline_20260505_033212/sac_racetrack_final.pth", "display_name": "R01 基础 SAC"},
            "R02": {"path": "outputs/racetrack-v0/models/SAC_R02_SAC_Speed_Priority_20260505_060152/sac_racetrack_final.pth", "display_name": "R02 速度优先"},
            "R03": {"path": "outputs/racetrack-v0/models/SAC_R03_SAC_Safety_Priority_20260505_083207/sac_racetrack_final.pth", "display_name": "R03 安全优先"},
            "R04": {"path": "outputs/racetrack-v0/models/SAC_R04_SAC_Extreme_Drift_20260505_110254/sac_racetrack_final.pth", "display_name": "R04 极限漂移"},
            "R05": {"path": "outputs/racetrack-v0/models/SAC_R05_SAC_Smooth_Racing_20260505_131614/sac_racetrack_final.pth", "display_name": "R05 单专家"},
            "R06": {"path": "outputs/racetrack-v0/models/SAC_R06_SAC_Wide_Dynamic_20260505_152958/sac_racetrack_final.pth", "display_name": "R06 宽域动态"},
            "R07": {"path": "outputs/racetrack-v0/models/SAC_R07_SAC_Zero_Tolerance_20260505_173235/sac_racetrack_final.pth", "display_name": "R07 零容忍"},
            "R08": {"path": "outputs/racetrack-v0/models/SAC_R08_SAC_Expert_Pro_20260505_184949/sac_racetrack_final.pth", "display_name": "R08 专家底座"},
                        
            # === 第一期 diff-SAC 实验 ===
            "DR01": {"path": "outputs/racetrack-v0/models/DiffSAC_DR01_Pure_BC_20260510_025310/online_finetune/diff_sac_final.pth", "display_name": "DR01 纯 BC 克隆"},
            "DR02": {"path": "outputs/racetrack-v0/models/DiffSAC_DR02_Micro_Q_20260510_060657/online_finetune/diff_sac_final.pth", "display_name": "DR02 微引导"},
            "DR03": {"path": "outputs/racetrack-v0/models/DiffSAC_DR03_Standard_Q_20260510_092222/online_finetune/diff_sac_final.pth", "display_name": "DR03 标准引导"},
            "DR04": {"path": "outputs/racetrack-v0/models/DiffSAC_DR04_Strong_Q_20260510_123826/online_finetune/diff_sac_final.pth", "display_name": "DR04 强力干预"},

            # === 第二期 Diff-SAC 混合专家实验 ===
            "DR05": {"path": "outputs/racetrack-v0/models/DiffSAC_DR05_Mixed_BC_20260510_155536/online_finetune/diff_sac_final.pth", "display_name": "DR05 混合纯BC"},
            "DR06": {"path": "outputs/racetrack-v0/models/DiffSAC_DR06_Mixed_Micro_Q_20260510_191112/online_finetune/diff_sac_final.pth", "display_name": "DR06 混合微引导"},
            "DR07": {"path": "outputs/racetrack-v0/models/DiffSAC_DR07_Mixed_Standard_Q_20260510_223055/online_finetune/diff_sac_final.pth", "display_name": "DR07 混合标引导"},
            "DR08": {"path": "outputs/racetrack-v0/models/DiffSAC_DR08_Mixed_Strong_Q_20260511_015410/online_finetune/diff_sac_final.pth", "display_name": "DR08 混合强干预"},
        }
    else: # highway-v0
        EXPERT_DATA_PATH = "data/expert_data/highway-v0/dataset_smart_mixed_90_10_20260413_031136/expert_transitions_smart_90_10.npz"
        models_to_evaluate = {
            # === 第一期 SAC 消融矩阵 ===
            "H01": {"path": "outputs/highway-v0/models/SAC_H01_Base_Highway_20260511_225245/sac_highway_final.pth", "display_name": "H01 基础高速"},
            "H02": {"path": "outputs/highway-v0/models/SAC_H02_Safety_Priority_20260512_040012/sac_highway_final.pth", "display_name": "H02 安全优先"},
            "H03": {"path": "outputs/highway-v0/models/SAC_H03_Speed_Priority_20260512_100655/sac_highway_final.pth", "display_name": "H03 速度优先"},
            "H04": {"path": "outputs/highway-v0/models/SAC_H04_Traffic_Jam_20260512_154634/sac_highway_final.pth", "display_name": "H04 拥堵路况"},

            # === 第一期 diff-SAC 实验 ===

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