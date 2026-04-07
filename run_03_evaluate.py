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
from envs.highway_wrapper import create_highway_env

# 统一设置中文字体，防止 matplotlib 画图时出现乱码 (针对 Windows 系统)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_single_model(model_name, model_path, env_name, eval_run_dir, num_episodes=100, record_video=True, expert_data_path=None):
    """
    对单个模型进行双线程评估：定性录像 (防崩溃) + 定量统计 (出数据)。
    包含智能路由逻辑：根据模型名称自动选择加载 SAC 还是 Diffusion 网络。
    """
    print(f"\n" + "=" * 60)
    print(f"🚀 开始公平评估模型: [{model_name}] | 测试回合数: {num_episodes}")
    print(f"📁 权重路径: {model_path}")
    print("=" * 60)

    # 智能路由判定：如果名字里带 diff，就走扩散模型那套复杂的处理逻辑
    is_diff = "Diff" in model_name or "diff" in model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 探针环境：开一个临时的环境，仅仅是为了读取状态和动作的维度
    dummy_env = create_highway_env(env_name, is_eval=True)
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.shape[0]
    max_action = float(dummy_env.action_space.high[0])
    dummy_env.close()

    # 2. 根据模型类型加载大脑
    if is_diff:
        print("🧠 检测到 Diffusion 架构，正在挂载 DiffSACAgent 与数据归一化器...")
        # Diffusion 模型极其依赖数据归一化。我们通过传入专家数据，建立统一的归一化基准
        buffer = MixedReplayBuffer(expert_data_path=expert_data_path, max_online_size=10, device=device)
        agent = DiffSACAgent(state_dim, action_dim, device=device)
        agent.load_pretrained_actor(model_path)

        # 🚨 [极其核心的修复] 🚨
        # 强行将主网络的权重 100% 覆盖给 EMA 影子网络！
        # 否则因为 EMA 的 decay=0.995，影子网络将保留 99.5% 的初始随机垃圾权重，导致出门就撞车。
        agent.ema_actor.model.load_state_dict(agent.actor.state_dict())
        print("🔧 EMA 权重 100% 同步修复完成，解除随机驾驶锁定！")

    else:
        print("🧠 检测到 SAC 架构，正在挂载经典 SACAgent...")
        agent = SACAgent(state_dim, action_dim, action_scale=max_action)
        try:
            agent.load_model(model_path)
        except Exception as e:
            print(f"❌ 加载模型 {model_name} 失败: {e}")
            return None

    # ==========================================
    # 阶段一：纯粹的视频录制定性环节 (仅录制前 2 局)
    # ==========================================
    if record_video:
        print(f"🎬 [阶段 1] 正在为 {model_name} 录制实战视频 (前 2 局)...")
        env_video = create_highway_env(env_name, is_eval=True)
        video_dir = os.path.join(eval_run_dir, "videos", model_name)
        os.makedirs(video_dir, exist_ok=True)
        env_video = RecordVideo(env_video, video_folder=video_dir, name_prefix=f"{model_name}_eval")

        for ep in range(2):
            state, _ = env_video.reset()
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

                print(f"\r├─ 录制 Ep {ep + 1}/2 | Step {ep_steps:3d} | 车速 vx: {ego_speed:5.2f} m/s", end="")
                if terminated or truncated:
                    print(f"\n└─ 录像完成: {'撞车/越野' if terminated else '完赛'}")
                    break
        env_video.close()

    # ==========================================
    # 阶段二：最高速的大样本定量评估环节
    # ==========================================
    print(f"\n⚡ [阶段 2] 执行 {num_episodes} 局大样本闭门测试...")
    env_eval = create_highway_env(env_name, is_eval=True) # 再次确认开启纯净评估模式
    metrics = {'rewards': [], 'lengths': [], 'speeds': [], 'crashes': 0}

    for ep in range(num_episodes):
        state, _ = env_eval.reset()
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

            if terminated or truncated:
                metrics['rewards'].append(ep_reward)
                metrics['lengths'].append(ep_steps)
                metrics['speeds'].append(np.mean(ep_speeds))
                if terminated:
                    metrics['crashes'] += 1 # 统计事故率

                if (ep + 1) % 10 == 0 or (ep + 1) == num_episodes:
                    print(f"进度: {ep + 1}/{num_episodes} 局已完成...")
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

    print(f"\n📊 [{model_name}] 评估报告:")
    print(f"   平均累计奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"   存活率(完赛率): {results['survival_rate']:.1f}%")
    print(f"   平均车速: {results['mean_speed']:.2f} m/s")

    return results


def save_metrics_to_csv(all_results, save_dir):
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
        for model_name, res in all_results.items():
            writer.writerow([model_name, f"{res['mean_reward']:.2f}", f"{res['std_reward']:.2f}",
                             f"{res['survival_rate']:.1f}", f"{res['mean_speed']:.2f}"])
    print(f"\n💾 量化指标数据已保存至 CSV: {os.path.abspath(csv_path)}")


def plot_comparisons(all_results, save_dir):
    """
    自动生成学术级对比箱线图和柱状图。
    """
    os.makedirs(save_dir, exist_ok=True)
    models = list(all_results.keys())

    # 图 1：累计奖励箱线图
    plt.figure(figsize=(12, 7))
    reward_data = [all_results[m]['raw_rewards'] for m in models]
    # showfliers=False 会隐藏那些极其偶然的严重扣分(离群点)，使得主流分布清晰可见
    plt.boxplot(reward_data, labels=models, showmeans=True, showfliers=False)
    plt.title('规控策略演进与消融实验对比 (Cumulative Reward)', fontsize=14)
    plt.ylabel('Episode Reward (Outliers Hidden)', fontsize=12)
    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reward_boxplot.png'), dpi=300)
    plt.close()

    # 图 2：存活率柱状图
    plt.figure(figsize=(12, 7))
    survival_rates = [all_results[m]['survival_rate'] for m in models]
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    bars = plt.bar(models, survival_rates, color=colors)
    plt.title('规控策略存活率对比 (Survival Rate)', fontsize=14)
    plt.ylabel('Survival Rate (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.xticks(rotation=25, ha='right')
    # 在柱子上标注具体数字
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'survival_rate_bar.png'), dpi=300)
    plt.close()
    print(f"📈 对比图表已保存至: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    # 🚨 Diff-SAC 需要依赖专家数据来统一尺度，填写你的专家数据集路径
    EXPERT_DATA_PATH = "data/expert_data/dataset_v5_20260404_035105/expert_transitions.npz"

    # ==========================================
    # 实验配置区：模型演进与消融实验大乱斗
    # ==========================================
    models_to_evaluate = {
        # v1.0: 没有任何约束，表现为“原地发癫”和“极度胆小”
        #"v1.0_Unconstrained": "outputs/models/highway-v0_SAC_20260329_150543/sac_highway_final.pth",

        # v2.0: 强化了速度奖励但没关草地，表现为“草地飙车党”
        #"v2.0_Offroad_Hacker": "outputs/models/highway-v0_SAC_20260329_185751/sac_highway_final.pth",

        # v3.0: 开启了草地死刑和LQR惩罚，但因局数制导致“严重欠拟合/早产”
        #"v3.0_LQR_Underfit": "outputs/models/highway-v0_SAC_20260330_010914/sac_highway_final.pth",

        # v5.0: 锁死了倒车和草地，并练够12万步，表现为求稳的“法规级专家”
        "v5.0_Safety_Conservative": "outputs/models/highway-v0_SAC_20260330_135449/sac_highway_final.pth",

        # v6.0: 引入绝对转向约束和高速重塑，表现为敢踩油门的“高效超车专家”
        "v6.0_Efficiency_Pro": "outputs/models/highway-v0_SAC_20260330_213300/sac_highway_final.pth",

        # Diff-Exp1: 微弱 Q 引导 (q=0.01)，高度依赖专家先验，理论上的“无冕之王/最稳老司机”
        "Diff_Exp1_Gentle_Q": "outputs/models/highway_DiffSAC_20260405_031920/diff_sac_ep400.pth",

        # Diff-Exp2: 标准 Q 引导 (q=0.05)，模仿与自主探索的平衡，偶尔会在超车时发生失误
        #"Diff_Exp2_Standard_Q": "outputs/models/highway_DiffSAC_20260405_065603/diff_sac_ep400.pth",

        # Diff-Exp3: 强力 Q 引导 (q=0.10)，完全陷入过估计陷阱，表现为“理论满分，实操零分”的翻车司机
        #"Diff_Exp3_Strong_Q": "outputs/models/highway_DiffSAC_20260405_101704/diff_sac_ep400.pth",

        # Diff-Exp4: 降学习率长跑 (lr=1e-4)，企图驯服高方差，但依然未能逃脱 OOD 陷阱
        #"Diff_Exp4_Stable_Long": "outputs/models/highway_DiffSAC_20260405_141352/diff_sac_ep500.pth",

        # Diff-Exp5: 极微引导 (q=0.005)，大幅削弱 RL 话语权，成功将 Actor 拉回安全边界，表现为“试探边界的行者”
        #"Diff_Exp5_Micro_Q": "outputs/models/highway_DiffSAC_20260406_023023/diff_sac_ep400.pth",

        # Diff-Exp6: 铁壁底座 (bc_epochs=120, q=0.05)，企图用超长预训练对抗分布偏移，但安全底线依然被 RL 粉碎的“反面教材”
        "Diff_Exp6_Bulletproof_BC": "outputs/models/highway_DiffSAC_20260406_040052/diff_sac_ep400.pth",

        # Diff-Exp7: 冰封微调 (q=0.005, lr=5e-5)，通过极致的保守实现安全与效率的完美平衡，理论上的“SOTA 冠军候选人”
        #"Diff_Exp7_Frozen_Finetune": "outputs/models/highway_DiffSAC_20260406_052938/diff_sac_ep500.pth",

        # Diff-Exp8: 零引导对照组 (q=0.0)，彻底关闭 Critic，退化为纯行为克隆的“循规蹈矩模仿者”，用于证明 RL 的必要性
        "Diff_Exp8_Zero_Q_Control": "outputs/models/highway_DiffSAC_20260406_064236/diff_sac_ep400.pth",

        # Diff-Exp9: 终极防御底座 (bc=120, q=0.005, lr=5e-5)，结合最厚装甲与最温柔微调，成功压制了在线微调初期的震荡
        #"Diff_Exp9_Ultimate_Safe_SOTA": "outputs/models/highway_DiffSAC_20260406_153903/diff_sac_ep500.pth",

        # Diff-Exp10: 加速冰封 (q=0.005, lr=1e-4)，在微弱引导下适度提升学习率，在保证不崩溃的前提下提升了环境适应效率
        #"Diff_Exp10_Accelerated_Finetune": "outputs/models/highway_DiffSAC_20260406_172128/diff_sac_ep500.pth",

        # Diff-Exp11: 极限微丝引导 (q=0.001)，进一步压低 RL 权重，Q值有效受到抑制并贴近专家分布，生存下限得到极大保障
        "Diff_Exp11_Ultra_Micro_Q": "outputs/models/highway_DiffSAC_20260406_211102/diff_sac_ep400.pth",

        # Diff-Exp12: 冰封马拉松 (ep=800)，进行超长周期的极限保守微调，Q值平滑攀升且全程未发生延迟崩溃，验证了长期稳定性
        #"Diff_Exp12_Frozen_Marathon": "outputs/models/highway_DiffSAC_20260407_013017/diff_sac_ep800.pth",

        # Diff-Exp13: 终极无坚不摧 (BC=120, q=0.001, lr=3e-4)，最厚护甲与最轻引导的黄金融合，Actor Loss 平稳缓降，预期防守反击 SOTA
        "Diff_Exp13_Unbreakable_SOTA": "outputs/models/highway_DiffSAC_20260407_071544/diff_sac_ep400.pth",

        # Diff-Exp14: 厚甲利刃 (BC=120, q=0.01, lr=3e-4)，更高 Q 引导导致 Actor Loss 显著下探但未崩溃，Q 值全场最高，预期均速 SOTA
        "Diff_Exp14_Thick_Shield_Gentle_Q": "outputs/models/highway_DiffSAC_20260407_110205/diff_sac_ep400.pth",

        # Diff-Exp15: 纯粹克隆的物理极限 (BC=120, q=0.0, lr=3e-4)，彻底关闭引导，Actor Loss 最平缓，提供 120 轮先验下的最高安全基准
        "Diff_Exp15_Deep_BC_Control": "outputs/models/highway_DiffSAC_20260407_140041/diff_sac_ep400.pth",

        # Diff-Exp16: 微丝引导马拉松 (BC=50, q=0.001, lr=3e-4, ep=600)，第三期冠军参数的加长版验证，600 局探索全程平滑无延迟崩溃
        "Diff_Exp16_Ultra_Micro_Marathon": "outputs/models/highway_DiffSAC_20260407_170756/diff_sac_ep600.pth",

        # Diff-Exp8_Run1: 零引导对照组 (首次测试，q=0.0, ep=400)，Critic 曲线全程平滑
        #"Diff_Exp8_Zero_Q_Run1": "outputs/models/highway_DiffSAC_20260406_064236/diff_sac_ep400.pth",

        # Diff-Exp8_Run2: 零引导对照组 (第二次测试，q=0.0, ep=500)，Critic 在后期发生震荡，但由于 q=0 被隔离
        #"Diff_Exp8_Zero_Q_Run2": "outputs/models/highway_DiffSAC_20260406_080556/diff_sac_ep500.pth",

        # Diff-Exp8_Run3: 零引导对照组 (第三次测试，q=0.0, ep=500)，Critic 在中期发生剧烈脉冲，同样被 q=0 隔离
        #"Diff_Exp8_Zero_Q_Run3": "outputs/models/highway_DiffSAC_20260406_095201/diff_sac_ep500.pth",
    }

    # 大样本测试局数，100 局是学术界的黄金免检样本量
    NUM_EVAL_EPISODES = 100

    # ==========================================
    # 动态文件夹命名逻辑：自动提取参与评估的模型简称
    # ==========================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_tags = []

    for name in models_to_evaluate.keys():
        if name.startswith("v"):
            # 提取 v5.0, v6.0
            version_tags.append(name.split('_')[0])
        else:
            # 提取 Exp1, Exp2, Exp5 等
            version_tags.append(name.split('_')[1])

    # 将提取的标签拼接起来，例如: v5.0_v6.0_Exp1_Exp2_Exp5_Exp6_Exp7_Exp8
    versions_str = "_".join(version_tags)
    eval_run_name = f"eval_[{versions_str}]_{timestamp}"

    # 构建最终的保存路径
    eval_run_dir = os.path.join("outputs", "eval_results", eval_run_name)
    plot_save_dir = os.path.join(eval_run_dir, "plots")
    data_save_dir = os.path.join(eval_run_dir, "data")

    # 启动批量测评
    all_results = {}
    for model_name, path in models_to_evaluate.items():
        if os.path.exists(path):
            res = evaluate_single_model(
                model_name=model_name,
                model_path=path,
                env_name='highway-v0',
                eval_run_dir=eval_run_dir,
                num_episodes=NUM_EVAL_EPISODES,
                expert_data_path=EXPERT_DATA_PATH # 将数据基准传给里面的 Diff 代理
            )
            if res:
                all_results[model_name] = res
        else:
            print(f"⚠️ 找不到权重文件，跳过评估: {path}")

    # 出图与落盘
    if len(all_results) > 0:
        save_metrics_to_csv(all_results, save_dir=data_save_dir)
        plot_comparisons(all_results, save_dir=plot_save_dir)