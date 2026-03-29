import os
import glob
from datetime import datetime
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
from algorithms.sac.sac_agent import SACAgent
from envs.highway_wrapper import create_highway_env


def get_latest_model_path(base_path="outputs/models"):
    """
    [自动寻路系统]
    在复杂的按时间戳生成的目录中，自动寻找最新的实验文件夹和最终的权重文件。
    避免手动复制粘贴路径引发的人为错误。
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"找不到基础模型目录: {base_path}")

    # 1. 寻找最新修改的实验子文件夹
    subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs:
        raise FileNotFoundError(f"在 {base_path} 下没有找到任何实验文件夹！")

    latest_dir = max(subdirs, key=os.path.getmtime)

    # 2. 优先寻找 'sac_highway_final.pth'
    final_model = os.path.join(latest_dir, "sac_highway_final.pth")
    if os.path.exists(final_model):
        return final_model

    # 3. 退而求其次，寻找该目录下最新的 checkpoint
    pths = glob.glob(os.path.join(latest_dir, "*.pth"))
    if not pths:
        raise FileNotFoundError(f"在 {latest_dir} 下没有找到任何 .pth 权重文件！")

    latest_pth = max(pths, key=os.path.getmtime)
    return latest_pth


def main():
    env_name = 'highway-v0'
    algo_name = 'SAC'

    # 1. 创建底层环境
    env = create_highway_env(env_name)

    # ---------------------------------------------------------
    # [架构优化] 动态视频存储路径对齐
    # 自动生成包含：环境名、算法名、时间戳的专属视频文件夹
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_run_name = f"{env_name}_{algo_name}_eval_{timestamp}"
    video_dir = os.path.join("outputs", "videos", video_run_name)
    os.makedirs(video_dir, exist_ok=True)

    # 录制包装器
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True,  # 记录每一局
        name_prefix="eval_video"  # 简化前缀，因为外层文件夹已经包含了足够的信息
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 2. 实例化基建大脑
    agent = SACAgent(state_dim, action_dim, action_scale=max_action)

    # 3. 自动加载最新权重
    try:
        model_path = get_latest_model_path()
        print(f"🎯 自动定位到最新权重文件: {model_path}")
        agent.load_model(model_path)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 4. 开始验收 (Evaluation)
    episodes = 3  # 我们录制 3 局足以看出性能
    print(f"\n🎬 摄像机已就绪，开始后台录制 {episodes} 局实战画面...")

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            # 实战期必须关闭重参数化的正态分布采样，直接输出 Actor 均值
            action = agent.select_action(state, evaluate=True)

            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                status = "💥 撞车 (Terminated)" if terminated else "🏁 完赛 (Truncated)"
                print(f"Episode {ep + 1}: 存活 {steps} 步 | 总回报: {episode_reward:.2f} | 结局: {status}")
                break

    env.close()
    print(f"\n✅ 验收完毕！实战录像已按规范保存至: {os.path.abspath(video_dir)}")


if __name__ == "__main__":
    main()