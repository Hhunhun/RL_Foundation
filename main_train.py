import os
import gymnasium as gym
import numpy as np
import torch
from core.replay_buffer import ReplayBuffer
from algorithms.sac.sac_agent import SACAgent
from utils.logger import Logger

def main():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"[{env_name}] State dim: {state_dim} | Action dim: {action_dim} | Max action: {max_action}")

    # 1. 实例化核心组件 (统一使用 outputs/ 路径前缀)
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=100000)
    agent = SACAgent(state_dim, action_dim, action_scale=max_action, lr=3e-4)
    logger = Logger(log_dir="outputs/logs", env_name=env_name)  # <-- 调整：统一为 outputs/

    max_episodes = 100
    batch_size = 256
    start_steps = 1000
    total_steps = 0

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            # 动作选择
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)

            # 环境交互
            next_state, reward, terminated, truncated, _ = env.step(action)
            done_bool = float(terminated)

            # 存入经验池
            replay_buffer.add(state, action, reward, next_state, done_bool)

            state = next_state
            episode_reward += reward
            total_steps += 1

            # 网络更新
            if replay_buffer.size > batch_size:
                # 获取 update 函数返回的 loss 字典
                loss_dict = agent.update(replay_buffer, batch_size)

                # 将 Loss 数据推送到 TensorBoard (每步记录)
                logger.log_scalar("Loss/Critic", loss_dict["critic_loss"], total_steps)
                logger.log_scalar("Loss/Actor", loss_dict["actor_loss"], total_steps)

            if terminated or truncated:
                break

        # 记录每一局的回报 (每局记录)
        logger.log_scalar("Reward/Episode_Reward", episode_reward, episode)

        print(f"Episode: {episode + 1}/{max_episodes} | Total Steps: {total_steps} | Reward: {episode_reward:.2f}")

    # <-- 调整：成功退格！让保存模型逻辑在 100 局循环彻底结束后才执行一次
    os.makedirs("outputs/models", exist_ok=True)
    agent.save_model("outputs/models/sac_pendulum.pth")

    env.close()
    logger.close()
    print("训练结束！基础测试通过。")

if __name__ == "__main__":
    main()