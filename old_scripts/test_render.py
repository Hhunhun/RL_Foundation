import gymnasium as gym
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from algorithms.sac.sac_agent import SACAgent


def test_agent_matplotlib():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SACAgent(state_dim, action_dim, action_scale=max_action)
    model_path = "../outputs/models/sac_pendulum.pth"
    agent.load_model(model_path)

    # 开启交互模式
    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.canvas.manager.set_window_title('SAC Pendulum (High Performance Render)')

    # --- 性能优化 1：在循环外预先设置好静态的画布属性 ---
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- 性能优化 2：预先画好中心点和“占位符”摆杆对象 ---
    ax.plot(0, 0, marker='o', markersize=10, color='black')  # 中心圆点永远不动
    line, = ax.plot([], [], linewidth=6, color='firebrick')  # 拿住这根线的引用
    title = ax.set_title("")

    episodes = 5
    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            cos_th, sin_th, _ = state

            # --- 核心提速：只更新线段的坐标，绝对不 clear 画布 ---
            line.set_data([0, sin_th], [0, cos_th])
            title.set_text(f"Episode: {ep + 1} | Action Torque: {action[0]:.2f}")

            # 强制立即刷新 GUI，不使用缓慢的 plt.pause
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Pendulum 环境的物理步长(dt)默认是 0.05 秒，我们让画面完美同步物理时间 (20 FPS)
            time.sleep(0.05)

        print(f"✅ 第 {ep + 1} 局结束，总得分: {episode_reward:.2f}")

    plt.ioff()
    plt.close()
    env.close()


if __name__ == "__main__":
    test_agent_matplotlib()