import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime

# 导入我们刚刚写好的核心组件
from algorithms.diffusion_sac.diffusion_model import ConditionalActor, NoiseScheduler


# ==========================================
# 1. 构建 PyTorch 离线数据集加载器
# ==========================================
class ExpertDataset(Dataset):
    def __init__(self, data_path):
        """
        一次性将 .npz 专家数据加载到内存，并转换为 PyTorch 张量
        """
        print(f"正在加载专家数据集: {data_path} ...")
        data = np.load(data_path)

        # 我们只需要路况状态 (observations) 和专家动作 (actions)
        self.states = torch.tensor(data['observations'], dtype=torch.float32)
        self.actions = torch.tensor(data['actions'], dtype=torch.float32)

        self.num_samples = len(self.states)
        print(f"✅ 成功加载 {self.num_samples} 条过渡数据 (Transitions)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


# ==========================================
# 2. 核心训练主循环
# ==========================================
def train_diffusion_bc(data_path, num_epochs=50, batch_size=256, learning_rate=3e-4):
    print("=" * 60)
    print("🚀 [阶段二] 开始 Diffusion Actor 离线预训练 (Behavior Cloning)")
    print("=" * 60)

    # 1. 硬件配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 训练设备: {device} (如果你用的是 3050，这里必须显示 cuda)")

    # 2. 准备数据
    dataset = ExpertDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 动态获取维度信息
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    # 3. 初始化扩散模型组件
    # 注意：这里的 100 步必须和后续在线强化学习时的步数保持一致
    noise_scheduler = NoiseScheduler(num_train_timesteps=100, device=device)
    actor_net = ConditionalActor(state_dim=state_dim, action_dim=action_dim).to(device)

    # 优化器与损失函数
    optimizer = torch.optim.Adam(actor_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # 创建保存权重的文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("outputs", "models", f"diffusion_bc_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # 4. 开启训练大循环
    actor_net.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_idx, (states, actions) in enumerate(dataloader):
            # 将数据推入显存
            states = states.to(device)
            actions = actions.to(device)
            current_batch_size = states.shape[0]

            # ---------------------------------------------------
            # 扩散模型核心训练逻辑
            # ---------------------------------------------------
            # a. 为 batch 中的每一个样本随机采样一个时间步 t (范围: 0 到 T-1)
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps,
                (current_batch_size,), device=device
            ).long()

            # b. 采样与动作维度一致的纯噪声 epsilon ~ N(0, I)
            noise = torch.randn_like(actions, device=device)

            # c. 使用调度器的魔法公式，一步计算出带噪动作 a_t
            noisy_actions = noise_scheduler.add_noise(
                original_action=actions,
                noise=noise,
                timesteps=timesteps
            )

            # d. 让神经网络看着状态 s 和时间步 t，去猜加入的纯噪声
            predicted_noise = actor_net(state=states, action=noisy_actions, time=timesteps)

            # e. 计算 MSE Loss (猜的噪声 vs 真实加入的噪声)
            loss = loss_fn(predicted_noise, noise)

            # f. 反向传播更新权重
            optimizer.zero_grad()
            loss.backward()
            # 加入梯度裁剪，防止梯度爆炸，这在扩散模型训练中是个好习惯
            torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)
            optimizer.step()
            # ---------------------------------------------------

            epoch_loss += loss.item()

        # 打印当前 Epoch 的平均 Loss
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"🔄 Epoch [{epoch + 1:03d}/{num_epochs:03d}] | MSE Loss: {avg_loss:.6f}")

    # 5. 保存最终模型权重
    model_save_path = os.path.join(save_dir, "diffusion_actor_bc.pth")
    torch.save(actor_net.state_dict(), model_save_path)

    print("=" * 60)
    print("🎉 离线预训练完成！")
    print(f"💾 模型权重已保存至: {model_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    # ⚠️ 请将下面这行替换为你刚刚采集到的数据集的实际路径！
    # 比如： "data/expert_data/dataset_v5_20260403_014037/expert_transitions.npz"
    EXPERT_DATA_PATH = "data/expert_data/dataset_v5_20260403_014037/expert_transitions.npz"

    # 50 个 Epoch 通常需要 5 到 10 分钟
    train_diffusion_bc(
        data_path=EXPERT_DATA_PATH,
        num_epochs=50,
        batch_size=256,
        learning_rate=3e-4
    )