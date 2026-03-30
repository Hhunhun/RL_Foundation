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

import gymnasium as gym
from gymnasium.wrappers import RecordVideo


# 导入你现有的模块
from algorithms.sac.sac_agent import SACAgent
from envs.highway_wrapper import create_highway_env

# 统一设置中文字体，防止图表乱码 (针对 Windows)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_single_model(model_name, model_path, env_name, eval_run_dir, num_episodes=30, record_video=True):
    """
    对单个模型进行双线程评估：定性录像 (防崩溃) + 定量统计。
    """
    print(f"\n" + "=" * 60)
    print(f"🚀 开始评估模型: [{model_name}] | 测试回合数: {num_episodes}")
    print(f"📁 权重路径: {model_path}")
    print("=" * 60)

    # 1. 初始化预备环境获取维度
    dummy_env = create_highway_env(env_name)
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.shape[0]
    max_action = float(dummy_env.action_space.high[0])
    dummy_env.close()

    # 加载大脑
    agent = SACAgent(state_dim, action_dim, action_scale=max_action)
    try:
        agent.load_model(model_path)
    except Exception as e:
        print(f"❌ 加载模型 {model_name} 失败: {e}")
        return None

    # ==========================================
    # 阶段一：纯粹的视频录制定性环节 (跑 2 局即关)
    # ==========================================
    if record_video:
        print(f"🎬 [阶段 1] 正在为 {model_name} 录制定性实战视频 (前 2 局)...")
        env_video = create_highway_env(env_name)

        # 将视频存入本次专属实验档案袋中
        video_dir = os.path.join(eval_run_dir, "videos", model_name)
        os.makedirs(video_dir, exist_ok=True)

        env_video = RecordVideo(env_video, video_folder=video_dir, name_prefix=f"{model_name}_eval")

        for ep in range(2):
            state, _ = env_video.reset()
            ep_steps = 0
            while True:
                action = agent.select_action(state, evaluate=True)
                state, reward, terminated, truncated, info = env_video.step(action)
                ep_steps += 1
                ego_speed = info.get("ego_speed_vx", 0.0)

                print(f"\r├─ 录制 Ep {ep + 1}/2 | Step {ep_steps:3d} | 车速 vx: {ego_speed:5.2f} m/s", end="")
                if terminated or truncated:
                    print(f"\n└─ 录像完成: {'撞车/越野' if terminated else '完赛'}")
                    break
        env_video.close()  # 安全释放视频资源

    # ==========================================
    # 阶段二：最高速的大样本定量评估环节
    # ==========================================
    print(f"\n⚡ [阶段 2] 执行 {num_episodes} 局大样本蒙特卡洛测试...")
    env_eval = create_highway_env(env_name)
    metrics = {'rewards': [], 'lengths': [], 'speeds': [], 'crashes': 0}

    for ep in range(num_episodes):
        state, _ = env_eval.reset()
        ep_reward = 0
        ep_steps = 0
        ep_speeds = []

        while True:
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
                    metrics['crashes'] += 1

                if (ep + 1) % 10 == 0 or (ep + 1) == num_episodes:
                    print(f"进度: {ep + 1}/{num_episodes} 局已完成...")
                break
    env_eval.close()

    # 3. 计算统计学指标
    results = {
        'mean_reward': np.mean(metrics['rewards']),
        'std_reward': np.std(metrics['rewards']),  # 策略性能反差(标准差)
        'survival_rate': (num_episodes - metrics['crashes']) / num_episodes * 100,
        'mean_speed': np.mean(metrics['speeds']),
        'raw_rewards': metrics['rewards']  # 保留原始数据用于画图
    }

    print(f"\n📊 [{model_name}] 评估报告:")
    print(f"   平均累计奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"   存活率(完赛率): {results['survival_rate']:.1f}%")
    print(f"   平均车速: {results['mean_speed']:.2f} m/s")

    return results


def save_metrics_to_csv(all_results, save_dir):
    """
    将量化指标保存为 CSV 文件，用于论文图表或后续分析。
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'summary_metrics.csv')

    headers = ['模型版本 (Model)', '平均累计奖励 (Mean Reward)', '策略反差/标准差 (Std Reward)',
               '存活率 (Survival Rate %)', '平均纵向速度 (Mean Speed m/s)']

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for model_name, res in all_results.items():
            writer.writerow([
                model_name,
                f"{res['mean_reward']:.2f}",
                f"{res['std_reward']:.2f}",
                f"{res['survival_rate']:.1f}",
                f"{res['mean_speed']:.2f}"
            ])

    print(f"\n💾 量化指标数据已保存至 CSV: {os.path.abspath(csv_path)}")


def plot_comparisons(all_results, save_dir):
    """
    生成学术级对比图表，存入本次实验档案袋。
    """
    os.makedirs(save_dir, exist_ok=True)
    models = list(all_results.keys())

    # 图 1：累计奖励箱线图 (隐藏离群点以恢复可视性)
    plt.figure(figsize=(10, 6))
    reward_data = [all_results[m]['raw_rewards'] for m in models]
    # showfliers=False 是关键！隐藏因为严重作弊被重罚导致的夸张极值，让箱体清晰可见
    plt.boxplot(reward_data, labels=models, showmeans=True, showfliers=False)
    plt.title('自动驾驶规控策略演进对比 (Cumulative Reward Distribution)', fontsize=14)
    plt.ylabel('Episode Reward (Outliers Hidden)', fontsize=12)
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reward_boxplot.png'), dpi=300)
    plt.close()

    # 图 2：存活率柱状图
    plt.figure(figsize=(10, 6))
    survival_rates = [all_results[m]['survival_rate'] for m in models]
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    bars = plt.bar(models, survival_rates, color=colors)
    plt.title('规控策略存活率对比 (Survival Rate)', fontsize=14)
    plt.ylabel('Survival Rate (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.xticks(rotation=15)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'survival_rate_bar.png'), dpi=300)
    plt.close()

    print(f"📈 学术级对比图表已保存至: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    # ==========================================
    # 实验配置区
    # ==========================================
    models_to_evaluate = {
        # v1.0: 没有任何约束，表现为“原地发癫”和“极度胆小”
        "v1.0_Unconstrained": "outputs/models/highway-v0_SAC_20260329_150543/sac_highway_final.pth",

        # v2.0: 强化了速度奖励但没关草地，表现为“草地飙车党”
        "v2.0_Offroad_Hacker": "outputs/models/highway-v0_SAC_20260329_185751/sac_highway_final.pth",

        # v3.0: 开启了草地死刑和LQR惩罚，但因局数制导致“严重欠拟合/早产”
        "v3.0_LQR_Underfit": "outputs/models/highway-v0_SAC_20260330_010914/sac_highway_final.pth",

        # v4.0: 改为步数制练够了4万步，但钻了规则空子学会了“倒车苟活”
        "v4.0_Reverse_Hacker": "outputs/models/highway-v0_SAC_20260330_013424/sac_highway_final.pth",

        # v5.0: 锁死了倒车和草地，并练够12万步，预期为“法规级专家”
        "v5.0_AV_Regulated": "outputs/models/highway-v0_SAC_20260330_135449/sac_highway_final.pth"
    }

    NUM_EVAL_EPISODES = 30  # 建议论文取值 50-100，平时测试用 30 足够

    # ---------------------------------------------------------
    # 🎯 动态实验归档系统构建
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_tags = [name.split('_')[0] for name in models_to_evaluate.keys()]
    versions_str = "_".join(version_tags)

    eval_run_name = f"eval_[{versions_str}]_{timestamp}"
    eval_run_dir = os.path.join("outputs", "eval_results", eval_run_name)
    print(f"📁 本次评估的独立实验档案袋已创建: {eval_run_dir}")

    # 定义子文件夹
    plot_save_dir = os.path.join(eval_run_dir, "plots")
    data_save_dir = os.path.join(eval_run_dir, "data")  # 新增数据存放目录

    # 执行大循环评估
    all_results = {}
    for model_name, path in models_to_evaluate.items():
        if os.path.exists(path):
            res = evaluate_single_model(
                model_name=model_name,
                model_path=path,
                env_name='highway-v0',
                eval_run_dir=eval_run_dir,
                num_episodes=NUM_EVAL_EPISODES
            )
            if res:
                all_results[model_name] = res
        else:
            print(f"⚠️ 找不到权重文件，跳过评估: {path}")

    # 数据落盘与绘图
    if len(all_results) > 0:
        save_metrics_to_csv(all_results, save_dir=data_save_dir)
        plot_comparisons(all_results, save_dir=plot_save_dir)