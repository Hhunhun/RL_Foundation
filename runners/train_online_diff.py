"""
阶段三：Diffusion-RL 在线微调流水线 (Online Fine-tuning with Demo Augmentation)

此模块是整个架构的最后一步，也是最关键的强化学习 (RL) 阶段。
它将第二阶段 (BC) 训练出的“准专家”放入真实的自动驾驶仿真环境中进行实车交互。
核心工作流：
1. 环境交互：收集带有真实物理反馈 (Reward) 的新数据。
2. 尺度转换：在物理环境（真实量纲）与神经网络（归一化量纲）之间无缝切换状态和动作。
3. 混合训练：调用 DiffSACAgent，利用经验池中的专家掩码，让智能体在“不忘本 (BC)”的前提下，
   追求更高的环境回报 (Q-learning)。
"""

import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 导入底层组件
from envs.highway_wrapper import create_highway_env
from core.offline_buffer import MixedReplayBuffer
from algorithms.diffusion_sac.diff_sac_agent import DiffSACAgent


def train_online_diffusion(pretrained_actor_path, expert_data_path, max_episodes=250, batch_size=256, q_weight=0.05, lr=3e-4):
    print("=" * 60)
    print("🚀 [阶段三] Diffusion-RL 在线微调 (Demo Augmented RL 专家混合增强版)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 运行设备: {device}")

    # 创建平滑版的课程学习环境 (algo="diff" 禁用了导致 Critic 崩溃的极端悬崖惩罚)
    env = create_highway_env("highway-v0", is_eval=False, algo="diff")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 1. 初始化代理并加载预训练权重
    # 将终端配置的 q_weight 和 lr 透传给底层智能体
    agent = DiffSACAgent(state_dim, action_dim, device=device, q_weight=q_weight, lr=lr)
    agent.load_pretrained_actor(pretrained_actor_path) # 挂载第二阶段的 BC 权重

    # 2. 初始化双区混合经验池
    # 内部会自动读取专家数据，并建立全局唯一的数据归一化基准 (Normalizers)
    replay_buffer = MixedReplayBuffer(expert_data_path=expert_data_path, max_online_size=100000, device=device)

    # --- 日志与存档准备 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("../outputs", "models", f"highway_DiffSAC_{timestamp}")
    log_dir = os.path.join("../outputs", "logs", f"highway_DiffSAC_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    total_steps = 0
    # 3. 核心在线交互与训练循环
    for episode in range(max_episodes):
        # env.reset() 返回的是带有物理量纲的原始状态 (例如速度为 20m/s，坐标为 150m)
        raw_state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0

        c_loss_list, a_loss_list, q_val_list = [], [], []

        while True:
            # a. 状态正向归一化：将物理尺度的状态压缩到 [-1, 1] 附近的分布，喂给神经网络
            norm_state = replay_buffer.state_normalizer.normalize(raw_state)

            # b. 策略采样：利用 Diffusion DDIM 推理出归一化尺度下的动作 (sample_steps=5 代表加速采样)
            norm_action = agent.select_action(norm_state, sample_steps=5, explore=True)

            # 🚨 c. 逆向去归一化，并施加绝对物理保护
            # 将网络吐出的归一化动作，还原成环境能听懂的物理动作 (如真实的转向角和加速度)
            raw_action = replay_buffer.action_normalizer.unnormalize(norm_action)
            # 物理引擎安全锁：防止极端动作导致仿真器崩溃
            clipped_raw_action = np.clip(raw_action, -1.0, 1.0)

            # d. 真实环境步进：执行动作，获取环境的客观反馈 (下一个状态和奖励)
            raw_next_state, reward, terminated, truncated, _ = env.step(clipped_raw_action)
            done = terminated or truncated

            # e. 存入经验池
            # 注意：此处存入的是物理尺度的原始数据。Buffer 内部的 add() 方法会自动调用 Normalizer 进行转换
            replay_buffer.add(raw_state, clipped_raw_action, reward, raw_next_state, terminated)

            # f. 网络更新 (当经验池里的数据足够塞满一个 Batch 时开始训练)
            if replay_buffer.online_size > batch_size or replay_buffer.expert_size > batch_size:
                # agent.update 内部完成了：Critic防爆更新 + Actor专家掩码克隆 + Q值引导更新
                c_loss, a_loss, q_val = agent.update(replay_buffer, batch_size, sample_steps=5)
                c_loss_list.append(c_loss)
                a_loss_list.append(a_loss)
                q_val_list.append(q_val)

            # 推进状态，累计统计数据
            raw_state = raw_next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if done:
                break

        # --- 指标统计与日志记录 ---
        avg_c_loss = np.mean(c_loss_list) if c_loss_list else 0.0
        avg_a_loss = np.mean(a_loss_list) if a_loss_list else 0.0
        avg_q_val = np.mean(q_val_list) if q_val_list else 0.0

        writer.add_scalar("Train/Reward", episode_reward, episode)
        writer.add_scalar("Train/Steps", episode_steps, episode)
        if c_loss_list:
            writer.add_scalar("Loss/Critic", avg_c_loss, episode)
            writer.add_scalar("Loss/Actor", avg_a_loss, episode)
            writer.add_scalar("Metric/Q_Value", avg_q_val, episode)

        # 终端实时打印进度
        print(f"🏁 Episode {episode + 1:03d} | Reward: {episode_reward:5.1f} | "
              f"Steps: {episode_steps:3d} | Q_val: {avg_q_val:5.2f} | C_Loss: {avg_c_loss:.3f} | A_Loss: {avg_a_loss:.3f}")

        # --- 阶段存档 ---
        if (episode + 1) % 50 == 0:
            save_path = os.path.join(save_dir, f"diff_sac_ep{episode + 1}.pth")
            # 只存档主 Actor 网络，后续在评估脚本中调用 load_pretrained_actor 时，它会自动同步给 EMA 网络
            torch.save(agent.actor.state_dict(), save_path)
            print(f"💾 模型已存档: {save_path}")

    env.close()
    writer.close()
    print("🎉 混合增强微调彻底完成！")

if __name__ == "__main__":
    # 本地独立测试时的入口路径 (通常通过 02_train_pipeline 调度，极少直接运行此文件)
    PRETRAINED_ACTOR_PATH = "../outputs/models/diffusion_bc_20260403_015017/diffusion_actor_bc.pth"
    EXPERT_DATA_PATH = "../data/expert_data/dataset_v5_20260403_014037/expert_transitions.npz"

    train_online_diffusion(
        pretrained_actor_path=PRETRAINED_ACTOR_PATH,
        expert_data_path=EXPERT_DATA_PATH,
        max_episodes=250
    )