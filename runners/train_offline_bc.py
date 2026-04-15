"""
阶段二：离线行为克隆预训练流水线 (Offline Behavior Cloning)

此模块负责 Diffusion-SAC 架构的“筑基”阶段。
它读取离线收集的真实专家驾驶数据，使用行为克隆 (BC) 的方式预训练 Diffusion Actor。
核心思想是：把专家的完美动作故意打乱（加噪），然后逼迫神经网络学着把它还原（去噪）。
通过这种方式，我们在不与环境交互的情况下，就能得到一个掌握基本驾驶技能的初始网络，
极大加速后续在线强化学习的收敛速度。
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

# 导入底层组件
from core.offline_buffer import ExpertDataset
from algorithms.diffusion_sac.diffusion_model import ConditionalActor, NoiseScheduler


def train_diffusion_bc(data_path, env_name="highway-v0", num_epochs=50, batch_size=256, learning_rate=3e-4):
    print("=" * 60)
    print("🚀 [阶段二] 开始 Diffusion Actor 离线预训练 (Behavior Cloning)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 训练设备: {device}")

    # 1. 实例化定制的 ExpertDataset (内部已自动完成全局归一化)
    # 将打包好的死数据变成可以按批次 (Batch) 投喂给神经网络的数据流
    dataset = ExpertDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    # 2. 初始化扩散模型组件
    noise_scheduler = NoiseScheduler(num_train_timesteps=100)
    actor_net = ConditionalActor(state_dim=state_dim, action_dim=action_dim).to(device)

    # 设置优化器 (Adam) 和损失函数 (均方误差 MSE)
    optimizer = torch.optim.Adam(actor_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # 锁定项目根目录 (此脚本在 runners/ 下，向上回退两级到达 RL_Foundation)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 准备模型保存路径 (使用绝对路径构建)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PROJECT_ROOT, "outputs", env_name, "models", f"diffusion_bc_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # 开启训练模式 (启用 Dropout/BatchNorm 等训练期特有行为)
    actor_net.train()

    # 3. 核心训练循环
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for states, actions in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            current_batch_size = states.shape[0]

            # --- 扩散模型训练的核心四步 ---

            # a. 随机采样时间步 t：决定我们要给这批动作加多重的噪声 (0 是极轻微，99 是完全破坏)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (current_batch_size,),
                                      device=device).long()

            # b. 采样纯噪声：从正态分布中抓取一把标准的白噪声
            noise = torch.randn_like(actions, device=device)

            # c. 前向加噪：把白噪声按照时间步 t 的强度，混入专家的完美动作中，得到“脏动作”
            noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

            # d. 预测噪声：把当前路况 (states) 和脏动作 (noisy_actions) 给网络，让它猜刚刚加进去的噪声是什么
            predicted_noise = actor_net(states, noisy_actions, timesteps)

            # e. 计算去噪损失：对比网络猜的噪声和实际加进去的真实噪声，差异越小越好
            loss = loss_fn(predicted_noise, noise)

            # f. 反向传播更新：根据误差调整网络权重，并加上梯度裁剪防止步子迈得太大扯到网络
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # 打印当前 Epoch 的平均损失，观察收敛情况
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"🔄 Epoch [{epoch + 1:03d}/{num_epochs:03d}] | MSE Loss: {avg_loss:.6f}")

    # 4. 存档
    model_save_path = os.path.join(save_dir, "diffusion_actor_bc.pth")
    torch.save(actor_net.state_dict(), model_save_path)

    print("=" * 60)
    print("🎉 离线预训练完成！")
    print(f"💾 模型权重已保存至: {model_save_path}")
    print("=" * 60)

    # 👇 加上这一行 👇
    # 返回预训练好的模型路径，以便后续的主控脚本 (02_train_pipeline) 直接将其移交给在线 RL 阶段
    return model_save_path


if __name__ == "__main__":
    # 本地独立测试时的入口
    EXPERT_DATA_PATH = "../data/expert_data/dataset_v5_20260403_014037/expert_transitions.npz"

    train_diffusion_bc(
        data_path=EXPERT_DATA_PATH,
        num_epochs=50,
        batch_size=256,
        learning_rate=3e-4
    )