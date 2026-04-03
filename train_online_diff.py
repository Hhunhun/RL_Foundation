import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 导入你的核心组件
from envs.highway_wrapper import create_highway_env
from core.replay_buffer import ReplayBuffer
from algorithms.diffusion_sac.diff_sac_agent import DiffSACAgent


def train_online_diffusion(pretrained_actor_path, max_episodes=200, batch_size=256):
    print("=" * 60)
    print("🚀 [阶段三] Diffusion-RL 在线微调 (Online Fine-tuning)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 运行设备: {device}")

    env = create_highway_env("highway-v0", is_eval=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DiffSACAgent(state_dim, action_dim, device=device)
    agent.load_pretrained_actor(pretrained_actor_path)

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=100000)

    # 日志与归档准备
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("outputs", "models", f"highway_DiffSAC_{timestamp}")
    log_dir = os.path.join("outputs", "logs", f"highway_DiffSAC_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # 初始化 TensorBoard Writer
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📊 TensorBoard 日志将保存在: {log_dir}")

    total_steps = 0
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0

        c_loss_list, a_loss_list = [], []

        while True:
            # a. 极速采样 (DDIM 5步)
            action = agent.select_action(state, sample_steps=5)

            # b. 环境步进
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # c. 数据存入经验池
            replay_buffer.add(state, action, reward, next_state, terminated)

            # d. 网络更新
            if replay_buffer.size > batch_size * 5:
                c_loss, a_loss = agent.update(replay_buffer, batch_size, sample_steps=5)
                c_loss_list.append(c_loss)
                a_loss_list.append(a_loss)

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if done:
                break

        # 回合结束，计算平均 Loss
        avg_c_loss = np.mean(c_loss_list) if c_loss_list else 0.0
        avg_a_loss = np.mean(a_loss_list) if a_loss_list else 0.0

        # 写入 TensorBoard
        writer.add_scalar("Train/Reward", episode_reward, episode)
        writer.add_scalar("Train/Steps", episode_steps, episode)
        if c_loss_list:
            writer.add_scalar("Loss/Critic", avg_c_loss, episode)
            writer.add_scalar("Loss/Actor", avg_a_loss, episode)

        print(f"🏁 Episode {episode + 1:03d} | Reward: {episode_reward:5.1f} | "
              f"Steps: {episode_steps:3d} | C_Loss: {avg_c_loss:.3f} | A_Loss: {avg_a_loss:.3f}")

        # 每 100 回合保存一次权重
        if (episode + 1) % 100 == 0:
            save_path = os.path.join(save_dir, f"diff_sac_ep{episode + 1}.pth")
            torch.save(agent.actor.state_dict(), save_path)
            print(f"💾 模型已存档: {save_path}")

    env.close()
    writer.close()
    print("🎉 在线微调训练彻底完成！")


if __name__ == "__main__":
    # ⚠️ 确保这里的路径是你刚刚跑出来的专家数据预训练模型！
    PRETRAINED_PATH = "outputs/models/diffusion_bc_20260403_015017/diffusion_actor_bc.pth"

    # 为了保险起见，建议第一次跑设为 200 局观察情况
    train_online_diffusion(pretrained_actor_path=PRETRAINED_PATH, max_episodes=300)