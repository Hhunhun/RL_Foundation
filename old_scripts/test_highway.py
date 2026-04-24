import os
import glob
from datetime import datetime
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
from algorithms.sac.sac_agent import SACAgent
from envs.highway_wrapper import create_highway_env


def get_latest_model_path(base_path="../outputs/highway-v0/models"):
    """[自动寻路系统] 保留原注释，寻找最新权重。"""
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"找不到基础模型目录: {base_path}")
    subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs:
        raise FileNotFoundError(f"在 {base_path} 下没有找到任何实验文件夹！")
    latest_dir = max(subdirs, key=os.path.getmtime)
    final_model = os.path.join(latest_dir, "sac_highway_final.pth")
    if os.path.exists(final_model):
        return final_model
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

    # 动态视频存储路径对齐 (保留原规范)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_run_name = f"{env_name}_{algo_name}_eval_{timestamp}"
    video_dir = os.path.join("../outputs", env_name, "videos", video_run_name)
    os.makedirs(video_dir, exist_ok=True)

    # 录制包装器
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True,  # 记录每一局
        name_prefix="eval_video_v5"  # [v5.0 标识]
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 2. 实例化基建大脑
    agent = SACAgent(state_dim, action_dim, action_scale=max_action)

    # 3. 自动加载最新权重
    try:
        model_path = get_latest_model_path()
        print(f"🎯 [test v5.0] 自动定位到最新权重文件: {model_path}")
        agent.load_model(model_path)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 4. 开始验收 (Evaluation)
    episodes = 3
    print(f"\n🎬 摄像机已就绪(开启show_trajectories轨迹显示)，开始录制 {episodes} 局实战画面...")
    print(f"🕵️ [专家诊断点] 请盯紧终端实时打印的 '自车速度 vx'。")

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            # 实战期必须关闭采样，输出均值
            action = agent.select_action(state, evaluate=True)

            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            # ----------------------------------------------------
            # 🕵️ [要求B之工业替代方案] 每一帧终端实时打印自车速 (m/s)
            # ----------------------------------------------------
            ego_speed = info.get("ego_speed_vx", 0.0)  # 读取 Wrapper 注入的探针
            print(
                f"\r├─ Ep {ep + 1} | Step {steps:3d} | 自车速度 vx: {ego_speed:5.2f} m/s | LQR塑形后单步奖励: {reward:.3f}",
                end="")

            if terminated or truncated:
                status = "💥 撞车或越野 (Terminated)" if terminated else "🏁 完赛 (Truncated)"
                print(f"\n└─ Episode {ep + 1}: 存活 {steps} 步 | 总回报: {episode_reward:.2f} | 结局: {status}\n")
                break

    env.close()
    print(f"\n✅ 验收完毕！\n📁 视频已保存(内含轨迹显示): {os.path.abspath(video_dir)}")


if __name__ == "__main__":
    main()