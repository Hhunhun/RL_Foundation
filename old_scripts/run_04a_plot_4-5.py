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
import sys
import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, MultipleLocator
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
        "axes.labelsize": 22,             # 坐标轴标签字号
        "xtick.labelsize": 20,            # X轴刻度字号
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
    # [新增] 记录每局的崩溃状态 (is_crashed)，用于计算条件方差
    metrics = {'rewards': [], 'lengths': [], 'speeds': [], 'crashes': 0, 'actions': [], 'is_crashed': []}

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
                metrics['is_crashed'].append(is_crashed) # 记录当局是否崩溃
                
                # [优化] 打印更详细的局末总结
                status = "💥 撞车/越野" if is_crashed else "🏁 完赛"
                print(f"\n└─ Episode {ep + 1} 结束: 存活 {ep_steps} 步 | 总回报: {ep_reward:.2f} | 结局: {status}")
                break
    env_eval.close()

    # 3. 结算统计学指标
    mean_reward = np.mean(metrics['rewards'])
    std_reward = np.std(metrics['rewards'])
    cv = std_reward / max(1e-3, abs(mean_reward)) if mean_reward != 0 else 0.0
    
    actions_arr = np.array(metrics['actions'])
    action_jerk_std = np.mean(np.std(np.diff(actions_arr, axis=0), axis=0)) if len(actions_arr) > 1 else 0.0

    results = {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'cv': cv,
        'action_jerk_std': action_jerk_std,
        'survival_rate': (num_episodes - metrics['crashes']) / num_episodes * 100,
        'mean_speed': np.mean(metrics['speeds']),
        'raw_rewards': metrics['rewards'],
        'actions': actions_arr, # 暴露出全部的动作张量
        'is_crashed': np.array(metrics['is_crashed']) # 暴露出每局的崩溃布尔值
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
    headers = ['模型版本 (Model)', '平均累计奖励 (Mean Reward)', '变异系数 (CV)',
               '存活率 (Survival Rate %)', '平均纵向速度 (Mean Speed m/s)', '控制平顺性 (Action Jerk Std)']

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for model_id, res in all_results.items():
            display_label = models_to_evaluate[model_id]["display_name"] # 从原始配置中获取 display_name
            writer.writerow([
                display_label, 
                f"{res['mean_reward']:.2f}", 
                f"{res['cv']:.3f}",
                f"{res['survival_rate']:.1f}", 
                f"{res['mean_speed']:.2f}",
                f"{res['action_jerk_std']:.3f}"
            ])
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
        "#00A087", # 翠绿色 (Teal Green)
        "#4DBBD5", # 蔚蓝色 (Cerulean Blue)
        "#91D1C2", # 薄荷青 (Mint)
        "#3C5488", # 午夜蓝 (Midnight Blue)
        "#F39B7F", # 珊瑚粉 (Salmon Pink)
        "#DC0000", # 深红色 (Crimson)
        "#7E6148", # 咖啡褐 (Coffee Brown)
        "#B09C85", # 浅卡其 (Light Khaki)
        "#4E79A7", # 稳重蓝 (Muted Blue)
        "#A73030", # 暗红色 (Dark Red)
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
    # 图 05：累计奖励雨云图
    # 结合了散点、箱线和半小提琴图，是展示核密度与真实数据分布的终极形态
    # ----------------------------------------------------
    # 🎨 [视觉精修配置区] 独立解耦四大元素的视觉参数，随心所欲定制您的完美图表
    # 👉 间距调节核心区：通过改变这几个数值，控制云、箱、雨的三层分离！
    CLOUD_STYLE     = {'alpha': 0.40, 'linewidth': 1.0, 'width': 0.8, 'offset': 0.14} # 云朵(KDE): width 控制厚度，offset 控制云朵与伞的悬空距离 (调小即可拉近)
    RAIN_STYLE      = {'alpha': 0.6, 'size': 4.0, 'jitter': 0.08, 'move': 0.25} # 雨滴(散点): move 控制散点往下平移的距离；jitter 控制雨滴上下散开的宽度
    
    # 🛠️ [箱线图(伞)参数调节区] 
    UMBRELLA_STYLE  = {
        'alpha': 0.85,       # 箱体(包括伞面、边框、伞骨)的透明度，可单独与云层分离调节
        'linewidth': 1.5,    # 箱线图边框、中位数线、上下四分位须线的粗细
        'width': 0.15,       # 箱线图的整体高度/胖瘦 (居中显示，已调小至 0.15 避免与雨滴重叠)
        # 'match_cloud' 已弃用，伞的颜色现直接按顺序从 academic_colors 读取
        'fallback_edgecolor': 'black', # 兜底边框颜色
        'fill_face': False   # 👉 [新增] 是否在箱框内部填充颜色，设为 False 则彻底透明(中空)
    }
    LIGHTNING_STYLE = {'alpha': 1.0, 'size': 60, 'edgewidth': 0.8, 'font_size': 13, 'y_offset': 0.0, 'match_cloud': True, 'fallback_color': 'gold', 'edgecolor': 'black'} # 闪电(均值): edgewidth 单独调节外边框粗细，font_size 调节均值标签字号
    
    # 🛠️ [坐标轴与外框调节区]
    AXES_STYLE      = {
        'grid_alpha': 0.4, 'spine_width': 1.2, 'spine_color': 'black',
        'x_range': (39.0, 54.0),         # 👉 [接口: X轴显示范围] 设为 None 表示自适应极值。若想固定，请改为如 (-20, 160)
        'x_ticks_step': None     # 👉 [核心修复] 设为 None 自动分5格。防止在高速等大奖励区间时，刻度过于密集导致被系统强制隐藏
    }
    
    plt.figure(figsize=(12.0, 8.0)) 
    
    import ptitprince as pt
    import pandas as pd
    import matplotlib.collections as mcoll
    
    # 1. 数据清洗与挂载
    flat_rewards, flat_labels = [], []
    for mid, label in zip(model_ids, display_labels):
        rewards = all_results[mid]['raw_rewards']
        flat_rewards.extend(rewards)
        flat_labels.extend([label] * len(rewards))
        
    df_rain = pd.DataFrame({'Reward': flat_rewards, 'Algorithm': flat_labels})
    df_rain = df_rain.dropna(subset=['Reward'])
    df_rain['Reward'] = df_rain['Reward'].astype(float)
    
    # 2. 调用引擎铺设底图 (利用 ptitprince 处理复杂的几何偏移，忽略其全局样式)
    pt.RainCloud(x='Algorithm', y='Reward', hue='Algorithm', data=df_rain, 
                 palette=colors, orient='h', ax=plt.gca(),
                 order=display_labels,         # 【核心修复】强制锁定渲染顺序，保障与后续均值对齐
                 point_size=RAIN_STYLE['size'], 
                 width_viol=CLOUD_STYLE['width'], width_box=UMBRELLA_STYLE['width'],
                 offset=CLOUD_STYLE.get('offset', 0.1), # 将云朵的悬空高度下发给底层引擎
                 bw=0.2, dodge=False, pointplot=False, move=RAIN_STYLE['move'],
                 jitter=RAIN_STYLE['jitter'], cut=2) 
                 
    ax = plt.gca()
    if ax.legend_ is not None:
        ax.legend_.remove()
        
    # 3. 🎯 核心黑科技：底层多态对象劫持 (Deep Object Takeover)
    # 分别识别并重塑云、雨、伞的独有属性，打破同质化
    for collection in ax.collections:
        if isinstance(collection, mcoll.PolyCollection):
            # [定制云朵] 核密度多边形
            collection.set_alpha(CLOUD_STYLE['alpha'])
            collection.set_linewidth(CLOUD_STYLE['linewidth'])
        elif isinstance(collection, mcoll.PathCollection):
            # [定制雨滴] 散点集合
            collection.set_alpha(RAIN_STYLE['alpha'])
            
    for i, patch in enumerate(ax.patches):
        target_color = colors[i % len(colors)]
        
        # [定制伞面] 四分位距箱体
        if UMBRELLA_STYLE.get('fill_face', False):
            patch.set_facecolor(target_color)
        else:
            patch.set_facecolor('none') # 强制内部透明 (中空)
            
        patch.set_edgecolor(target_color)
        patch.set_linewidth(UMBRELLA_STYLE['linewidth'])
        patch.set_alpha(UMBRELLA_STYLE['alpha'])
        patch.set_zorder(5)
        
    for line in ax.lines:
        # [隐藏异常值] 箱线图自带的异常值点是带有 marker 的线条，直接将其强制隐形，防止与雨滴散点叠印
        if line.get_marker() not in ['None', ' ', '', None]:
            line.set_visible(False)
            
    # 过滤出伞骨有效线条，采用绝对空间坐标匹配，彻底解决 Seaborn 底层线段乱序的 Bug
    valid_lines = [line for line in ax.lines if line.get_visible()]
    
    # 提取所有箱体的物理 Y 轴中心坐标
    box_y_centers = []
    for patch in ax.patches:
        if hasattr(patch, 'get_y') and hasattr(patch, 'get_height'):
            box_y_centers.append(patch.get_y() + patch.get_height() / 2)
        else:
            # 兼容高版本 Seaborn/Matplotlib 中箱体变为 PathPatch 的情况
            bbox = patch.get_path().get_extents()
            box_y_centers.append((bbox.y0 + bbox.y1) / 2)
    
    for line in valid_lines:
        # 计算当前线条的 Y 轴物理中心点
        line_y_center = np.mean(line.get_ydata())
        
        # 找到在空间上最贴近的箱体索引
        if len(box_y_centers) > 0:
            patch_idx = np.argmin([abs(line_y_center - bc) for bc in box_y_centers])
            target_color = colors[patch_idx % len(colors)]
            lightning_target_color = target_color if LIGHTNING_STYLE.get('match_cloud', True) else LIGHTNING_STYLE.get('fallback_color', 'gold')
        else:
            target_color = UMBRELLA_STYLE.get('fallback_edgecolor', 'black')
            lightning_target_color = LIGHTNING_STYLE.get('fallback_color', 'gold')
        
        # [区分线型] 判断是否为水平长横线(须线)
        ydata = line.get_ydata()
        is_whisker = (len(ydata) == 2 and ydata[0] == ydata[1])
        
        # [定制伞骨] 中位数与极值虚线
        final_color = lightning_target_color if is_whisker else target_color
        line.set_color(final_color)
        line.set_linewidth(UMBRELLA_STYLE['linewidth'])
        line.set_alpha(UMBRELLA_STYLE['alpha'])
        line.set_zorder(5)
        
        if is_whisker:
            line.set_linestyle('--') # 如果 Y 坐标不变，说明是水平横线，设为虚线
        else:
            line.set_linestyle('-')  # 否则为竖线，保持实线

    # 4. ⚡ 独立绘制“闪电” (均值标记)
    means = [all_results[m]['mean_reward'] for m in model_ids]
    y_positions = np.arange(len(model_ids))
    
    lightning_colors = colors[:len(model_ids)] if LIGHTNING_STYLE.get('match_cloud', True) else [LIGHTNING_STYLE.get('fallback_color', 'gold')] * len(model_ids)
    ax.scatter(means, y_positions + LIGHTNING_STYLE['y_offset'], 
               marker='D', c=lightning_colors, 
               edgecolors=LIGHTNING_STYLE.get('edgecolor', 'black'), 
               linewidths=LIGHTNING_STYLE.get('edgewidth', 1.2), 
               s=LIGHTNING_STYLE['size'], alpha=LIGHTNING_STYLE['alpha'], zorder=10)
                
    # [新增] 在闪电正左方标注具体均值数值
    for mean_val, y_pos, color in zip(means, y_positions, lightning_colors):
        ax.annotate(f"{mean_val:.2f}",
                    xy=(mean_val, y_pos + LIGHTNING_STYLE['y_offset']),
                    xytext=(-7, 0), textcoords='offset points', # 向左偏移 12 个像素，避开菱形
                    ha='right', va='center',
                    fontsize=LIGHTNING_STYLE.get('font_size', 11), fontweight='bold', color=color)

    # 5. [定制坐标与外框]
    # plt.title('规控策略奖励双峰分布雨云图', pad=AXES_STYLE['title_pad']) # 已应要求移除大标题
    plt.xlabel('回合累计奖励')
    plt.ylabel('')
    plt.grid(axis='x', linestyle='--', alpha=AXES_STYLE['grid_alpha'])
    
    # 🛠️ [横坐标刻度步长应用]
    if AXES_STYLE.get('x_ticks_step') is not None:
        ax.xaxis.set_major_locator(MultipleLocator(AXES_STYLE['x_ticks_step']))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    
    # 🛠️ [横坐标范围应用]
    if AXES_STYLE['x_range'] is not None:
        ax.set_xlim(AXES_STYLE['x_range'])
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXES_STYLE['spine_color'])
        spine.set_linewidth(AXES_STYLE['spine_width'])
        
    # 强制显示被 seaborn 隐藏的物理刻度短线与文字，支持 which='both' 贯穿主副刻度
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=True, direction='in', length=0, width=AXES_STYLE['spine_width'], color=AXES_STYLE['spine_color'])
    ax.tick_params(axis='y', which='both', left=True, labelleft=True, direction='in', length=0, width=AXES_STYLE['spine_width'], color=AXES_STYLE['spine_color'])
    
    # [兜底保障] 强制唤醒所有 X 轴的刻度文本对象
    for label in ax.get_xticklabels():
        label.set_visible(True)
    
    _save_and_close_fig('05_reward_raincloud')

    print(f"📈 精简后的核心对比图表已全部保存至: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    # ==========================================
    # 终端交互配置终端
    # ==========================================
    print("🤖 使用统一模型评估终端")
    print("==========================================")
    
    # 1. 选择运行模式
    print("👉 请选择运行模式:")
    print("  [1] 全量评估 (重新运行仿真测试并保存数据)")
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
        # 支持硬编码路径直达，节省检索时间
        HARDCODED_PKL_PATHS = {
            "merge-v0": r"E:\Autol_Lab\RL_Foundation\outputs\merge-v0\eval_results\[M01_M02_M03_M04_M05_M06_M07_M08_DM01_DM02_DM03_DM04_DM05_DM06_DM07_DM08]_20260516_021146\data\all_results.pkl",
            "racetrack-v0": r"E:\Autol_Lab\RL_Foundation\outputs\racetrack-v0\eval_results\[R01_R02_R03_R04_R05_R06_R07_R08_DR01_DR02_DR03_DR04_DR05_DR06_DR07_DR08]_20260516_125952\data\all_results.pkl",
            "highway-v0": r"E:\Autol_Lab\RL_Foundation\outputs\highway-v0\eval_results\[H01_H02_H03_H04_DH01_DH02_DH03_DH04_DH05_DH06_DH07_DH08]_20260516_152551\data\all_results.pkl"
        }
        
        if TARGET_ENV in HARDCODED_PKL_PATHS and os.path.exists(HARDCODED_PKL_PATHS[TARGET_ENV]):
            LOAD_PKL_PATH = HARDCODED_PKL_PATHS[TARGET_ENV]
            print(f"✅ 已应用硬编码指定的数据文件: {LOAD_PKL_PATH}")
        else:
            eval_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", TARGET_ENV, "eval_results")
            if os.path.exists(eval_base_dir):
                subdirs = [os.path.join(eval_base_dir, d) for d in os.listdir(eval_base_dir) if os.path.isdir(os.path.join(eval_base_dir, d))]
                subdirs.sort(key=os.path.getmtime, reverse=True)
                for subdir in subdirs:
                    pkl_file = os.path.join(subdir, "data", "all_results.pkl")
                    if os.path.exists(pkl_file):
                        LOAD_PKL_PATH = pkl_file
                        break
                
                if LOAD_PKL_PATH:
                    print(f"✅ 自动找到最新数据文件: {LOAD_PKL_PATH}")
                else:
                    print(f"❌ 未在该环境下找到任何 all_results.pkl 文件，请先运行 [1] 全量评估！")
                    sys.exit(1)
            else:
                print(f"❌ 目录 {eval_base_dir} 不存在，请先运行 [1] 全量评估！")
                sys.exit(1)

    # 4. 选择图表标签风格
    print("==========================================")
    print("👉 请选择图表标签风格:")
    print("  [1] 原始工程调试标签 (例如: DM01 纯 BC 克隆)")
    print("  [2] 学术中文规范标签 (例如: Diff-SAC (纯行为克隆))")
    label_choice = input("请输入 1 或 2 (默认 1): ").strip()
    USE_ACADEMIC_LABELS = (label_choice == '2')
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
            "M01": {"path": "outputs/merge-v0/models/SAC_M01_Base_Merge_20260511_042953/sac_merge_final.pth", "raw_name": "M01 基础生存", "acad_name": "SAC 基线"},
            #"M02": {"path": "outputs/merge-v0/models/SAC_M02_Efficient_Smooth_20260420_154007/sac_merge_final.pth", "raw_name": "M02 高效平滑", "acad_name": "SAC-平顺偏好"},
            #"M03": {"path": "outputs/merge-v0/models/SAC_M03_Aggressive_Gap_Finding_20260420_162217/sac_merge_final.pth", "raw_name": "M03 激进寻隙", "acad_name": "SAC-效率导向"},
            "M04": {"path": "outputs/merge-v0/models/SAC_M04_Safety_First_20260420_170911/sac_merge_final.pth", "raw_name": "M04 安全至上", "acad_name": "SAC 专家"},
            #"M05": {"path": "outputs/merge-v0/models/SAC_M05_Patient_Merger_20260420_220108/sac_merge_final.pth", "raw_name": "M05 耐心等待", "acad_name": "SAC-保守适应"},
            #"M06": {"path": "outputs/merge-v0/models/SAC_M06_Extreme_Penalty_20260420_232207/sac_merge_final.pth", "raw_name": "M06 极限死刑", "acad_name": "SAC-强安全约束"},
            #"M07": {"path": "outputs/merge-v0/models/SAC_M07_Smooth_Marathon_20260421_003822/sac_merge_final.pth", "raw_name": "M07 平滑马拉松", "acad_name": "SAC-长视界平顺"},
            #"M08": {"path": "outputs/merge-v0/models/SAC_M08_Ultimate_Merge_20260421_023258/sac_merge_final.pth", "raw_name": "M08 终极汇入", "acad_name": "SAC-综合强约束"},

            # === 第一期 diff-SAC 单专家实验 ===
            "DM01": {"path": "outputs/merge-v0/models/DiffSAC_DM01_Pure_BC_20260511_135709/online_finetune/diff_sac_final.pth", "raw_name": "DM01 纯 BC 克隆", "acad_name": "Diff 纯BC", "data_path": SINGLE_DATA_PATH},
            #"DM02": {"path": "outputs/merge-v0/models/DiffSAC_DM02_Micro_Q_20260511_152002/online_finetune/diff_sac_final.pth", "raw_name": "DM02 微引导", "acad_name": "单专家 Diff-SAC-微引导", "data_path": SINGLE_DATA_PATH},
            #"DM03": {"path": "outputs/merge-v0/models/DiffSAC_DM03_Standard_Q_20260511_170511/online_finetune/diff_sac_final.pth", "raw_name": "DM03 标准引导", "acad_name": "单专家 Diff-SAC-标准引导", "data_path": SINGLE_DATA_PATH},
            #"DM04": {"path": "outputs/merge-v0/models/DiffSAC_DM04_Strong_Q_20260511_183810/online_finetune/diff_sac_final.pth", "raw_name": "DM04 强力干预", "acad_name": "单专家 Diff-SAC-强引导", "data_path": SINGLE_DATA_PATH},

            # === 第二期 diff-SAC 混合专家实验 ===
            #"DM05": {"path": "outputs/merge-v0/models/DiffSAC_DM05_Mixed_BC_20260513_163546/online_finetune/diff_sac_final.pth", "raw_name": "DM05 混合纯BC", "acad_name": "混合专家 Diff-SAC-纯BC", "data_path": MIXED_DATA_PATH},
            "DM06": {"path": "outputs/merge-v0/models/DiffSAC_DM06_Mixed_Micro_Q_20260513_175844/online_finetune/diff_sac_final.pth", "raw_name": "DM06 混合微引导", "acad_name": "Diff 引导", "data_path": MIXED_DATA_PATH},
            #"DM07": {"path": "outputs/merge-v0/models/DiffSAC_DM07_Mixed_Standard_Q_20260513_194923/online_finetune/diff_sac_final.pth", "raw_name": "DM07 混合标引导", "acad_name": "混合专家 Diff-SAC-标准引导", "data_path": MIXED_DATA_PATH},
            #"DM08": {"path": "outputs/merge-v0/models/DiffSAC_DM08_Mixed_Strong_Q_20260513_211948/online_finetune/diff_sac_final.pth", "raw_name": "DM08 混合强干预", "acad_name": "混合专家 Diff-SAC-强引导", "data_path": MIXED_DATA_PATH},
        }

    elif TARGET_ENV == "racetrack-v0":
        SINGLE_DATA_PATH = "data/expert_data/racetrack-v0/dataset_R05_mode1_20260506_011817/expert_transitions.npz"
        MIXED_DATA_PATH = "data/expert_data/racetrack-v0/dataset_mixed_0.8R05_0.2R01_20260506_142446/expert_transitions_mixed_0.8R05_0.2R01.npz"
        
        models_to_evaluate = {
            # === 第一期 SAC 消融矩阵 ===
            "R01": {"path": "outputs/racetrack-v0/models/SAC_R01_SAC_Baseline_20260505_033212/sac_racetrack_final.pth", "raw_name": "R01 基础 SAC", "acad_name": "SAC-标准基线"},
            #"R02": {"path": "outputs/racetrack-v0/models/SAC_R02_SAC_Speed_Priority_20260505_060152/sac_racetrack_final.pth", "raw_name": "R02 速度优先", "acad_name": "SAC-效率导向"},
            #"R03": {"path": "outputs/racetrack-v0/models/SAC_R03_SAC_Safety_Priority_20260505_083207/sac_racetrack_final.pth", "raw_name": "R03 安全优先", "acad_name": "SAC-安全约束"},
            #"R04": {"path": "outputs/racetrack-v0/models/SAC_R04_SAC_Extreme_Drift_20260505_110254/sac_racetrack_final.pth", "raw_name": "R04 极限漂移", "acad_name": "SAC-无约束探索"},
            "R05": {"path": "outputs/racetrack-v0/models/SAC_R05_SAC_Smooth_Racing_20260505_131614/sac_racetrack_final.pth", "raw_name": "R05 单专家", "acad_name": "SAC-平顺专家"},
            #"R06": {"path": "outputs/racetrack-v0/models/SAC_R06_SAC_Wide_Dynamic_20260505_152958/sac_racetrack_final.pth", "raw_name": "R06 宽域动态", "acad_name": "SAC-宽域动态"},
            #"R07": {"path": "outputs/racetrack-v0/models/SAC_R07_SAC_Zero_Tolerance_20260505_173235/sac_racetrack_final.pth", "raw_name": "R07 零容忍", "acad_name": "SAC-强安全约束"},
            #"R08": {"path": "outputs/racetrack-v0/models/SAC_R08_SAC_Expert_Pro_20260505_184949/sac_racetrack_final.pth", "raw_name": "R08 专家底座", "acad_name": "SAC-专家基准"},
    
            # === 第一期 diff-SAC 单专家实验 ===
            #"DR01": {"path": "outputs/racetrack-v0/models/DiffSAC_DR01_Pure_BC_20260510_025310/online_finetune/diff_sac_final.pth", "raw_name": "DR01 纯 BC 克隆", "acad_name": "单专家 Diff-SAC-纯BC", "data_path": SINGLE_DATA_PATH},
            #"DR02": {"path": "outputs/racetrack-v0/models/DiffSAC_DR02_Micro_Q_20260510_060657/online_finetune/diff_sac_final.pth", "raw_name": "DR02 微引导", "acad_name": "单专家 Diff-SAC-微引导", "data_path": SINGLE_DATA_PATH},
            #"DR03": {"path": "outputs/racetrack-v0/models/DiffSAC_DR03_Standard_Q_20260510_092222/online_finetune/diff_sac_final.pth", "raw_name": "DR03 标准引导", "acad_name": "单专家 Diff-SAC-标准引导", "data_path": SINGLE_DATA_PATH},
            #"DR04": {"path": "outputs/racetrack-v0/models/DiffSAC_DR04_Strong_Q_20260510_123826/online_finetune/diff_sac_final.pth", "raw_name": "DR04 强力干预", "acad_name": "单专家 Diff-SAC-强引导", "data_path": SINGLE_DATA_PATH},

            # === 第二期 Diff-SAC 混合专家实验 ===
            "DR05": {"path": "outputs/racetrack-v0/models/DiffSAC_DR05_Mixed_BC_20260510_155536/online_finetune/diff_sac_final.pth", "raw_name": "DR05 混合纯BC", "acad_name": "混合专家 Diff-SAC-纯BC", "data_path": MIXED_DATA_PATH},
            "DR06": {"path": "outputs/racetrack-v0/models/DiffSAC_DR06_Mixed_Micro_Q_20260510_191112/online_finetune/diff_sac_final.pth", "raw_name": "DR06 混合微引导", "acad_name": "混合专家 Diff-SAC-微引导", "data_path": MIXED_DATA_PATH},
            #"DR07": {"path": "outputs/racetrack-v0/models/DiffSAC_DR07_Mixed_Standard_Q_20260510_223055/online_finetune/diff_sac_final.pth", "raw_name": "DR07 混合标引导", "acad_name": "混合专家 Diff-SAC-标准引导", "data_path": MIXED_DATA_PATH},
            #"DR08": {"path": "outputs/racetrack-v0/models/DiffSAC_DR08_Mixed_Strong_Q_20260511_015410/online_finetune/diff_sac_final.pth", "raw_name": "DR08 混合强干预", "acad_name": "混合专家 Diff-SAC-强引导", "data_path": MIXED_DATA_PATH},
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

    if not PLOT_ONLY:
        # 大样本测试局数，100 局是黄金免检样本量
        NUM_EVAL_EPISODES = 100

        # 动态文件夹命名逻辑
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_tags = list(models_to_evaluate.keys())
        versions_str = "_".join(version_tags)
        eval_run_name = f"[{versions_str}]_{timestamp}"

        base_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plot", "plot4-5")
        eval_run_dir = os.path.join(base_out_dir, TARGET_ENV, eval_run_name)
        plot_dir_name = "plots_academic_cn" if USE_ACADEMIC_LABELS else "plots_raw_tags"
        plot_save_dir = os.path.join(eval_run_dir, plot_dir_name)
        data_save_dir = os.path.join(eval_run_dir, "data")

        # 启动批量测评
        all_results = {}
        for model_id, model_config in models_to_evaluate.items():
            model_path = model_config["path"]
            display_label = model_config["display_name"]
            expert_data_path = model_config.get("data_path", None)
            
            if os.path.exists(model_path):
                res = evaluate_single_model(
                    model_id=model_id, model_path=model_path, display_label=display_label,
                    env_name=TARGET_ENV, eval_run_dir=eval_run_dir,
                    num_episodes=NUM_EVAL_EPISODES, expert_data_path=expert_data_path
                )
                if res: all_results[model_id] = res
            else:
                print(f"⚠️ 找不到权重文件，跳过评估: {model_path}")

        # 出图与落盘
        if len(all_results) > 0:
            # 💾 核心新增：将全量原始评估数据备份至硬盘，防崩溃
            os.makedirs(data_save_dir, exist_ok=True)
            pkl_path = os.path.join(data_save_dir, 'all_results.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"\n💾 原始评估数据已备份至: {os.path.abspath(pkl_path)}")
            print(f"💡 (下次若仅需微调图表参数，可将脚本上方 PLOT_ONLY 设为 True 并传入此路径即可秒出图)")
            
            save_metrics_to_csv(all_results, models_to_evaluate, save_dir=data_save_dir)
            plot_comparisons(all_results, models_to_evaluate, save_dir=plot_save_dir)
    else:
        print(f"\n⏩ [极速出图模式] 正在跳过长时间的物理评估，直接加载本地数据...")
        print(f"📦 读取路径: {LOAD_PKL_PATH}")
        if os.path.exists(LOAD_PKL_PATH):
            with open(LOAD_PKL_PATH, 'rb') as f:
                all_results = pickle.load(f)
            
            # 💡 [核心新增] 基于当前未被注释的 models_to_evaluate 字典，对全量数据进行按需过滤
            all_results = {k: v for k, v in all_results.items() if k in models_to_evaluate}
            
            if len(all_results) > 0:
                # [核心修复] 兼容老版本 pkl，动态补充 cv 和 action_jerk_std 字段
                for m, res in all_results.items():
                    if 'cv' not in res:
                        mean_r = res.get('mean_reward', 0.0)
                        res['cv'] = res.get('std_reward', 0.0) / max(1e-3, abs(mean_r)) if mean_r != 0 else 0.0
                    if 'action_jerk_std' not in res:
                        acts = res.get('actions', np.array([]))
                        res['action_jerk_std'] = np.mean(np.std(np.diff(acts, axis=0), axis=0)) if len(acts) > 1 else 0.0

                # 根据开关选择不同的出图子目录，防止互相覆盖
                base_dir = os.path.dirname(os.path.dirname(LOAD_PKL_PATH))
                run_name = os.path.basename(base_dir)
                target_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plot", "plot4-5", TARGET_ENV, run_name)
                plot_dir_name = "plots_academic_cn" if USE_ACADEMIC_LABELS else "plots_raw_tags"
                plot_save_dir = os.path.join(target_base, plot_dir_name)
                data_save_dir = os.path.join(target_base, "data")
                os.makedirs(plot_save_dir, exist_ok=True)
                
                print(f"✅ 数据过滤成功！已提取当前激活的 {len(all_results)} 个模型的数据。正在重新绘制图表...")
                save_metrics_to_csv(all_results, models_to_evaluate, save_dir=data_save_dir)
                plot_comparisons(all_results, models_to_evaluate, save_dir=plot_save_dir)
                print(f"🎨 所有图表已重新渲染并保存至: {plot_save_dir}")
            else:
                print("❌ 过滤后没有任何模型数据！请检查您解除注释的模型 ID 是否存在于该 .pkl 文件中。")
        else:
            print(f"❌ 找不到指定的 .pkl 数据文件，请检查 LOAD_PKL_PATH 路径！")