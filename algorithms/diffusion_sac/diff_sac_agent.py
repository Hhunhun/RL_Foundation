"""
Diffusion-SAC 智能体核心控制模块 (Diffusion-SAC Agent)

此模块是 Offline-to-Online 强化学习架构的“指挥中枢”，负责协调 Actor 和 Critic 的训练。
核心创新点在于：
1. 混合策略更新 (Hybrid Policy Update)：Actor 不仅要预测噪声（模仿专家），还要最大化 Q 值（追求高回报）。
2. 非对称掩码 (Asymmetric Masking)：Actor 的模仿行为被严格限制在专家数据上，但 Critic 的惩罚/奖励信号对所有数据开放。
3. 工业级防爆设计：通过奖励截断 (Reward Clipping) 和平滑损失函数 (Huber Loss)，确保 Critic 在严苛自动驾驶环境中不会发生梯度爆炸。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from algorithms.diffusion_sac.diffusion_model import ConditionalActor, NoiseScheduler, EMAModel


class DoubleQCritic(nn.Module):
    """
    双 Q 值评估网络 (Twin Delayed Critic)。
    强化学习中，Critic 网络很容易对未见过的动作产生“盲目乐观”（过估计，Overestimation）。
    通过同时训练两个独立的 Q 网络，并在计算目标 Q 值时取两者的最小值，
    可以极大地压制这种危险的过估计倾向，这对于自动驾驶的安全至关重要。
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1 网络
        self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.Mish(),
                                nn.Linear(hidden_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, 1))
        # Q2 网络 (结构与 Q1 完全相同，但初始化权重不同)
        self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.Mish(),
                                nn.Linear(hidden_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        # 将状态和动作拼接在一起，同时送入两个网络评估其价值
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


class DiffSACAgent:
    """
    Diffusion-SAC 智能体封装类。
    包含了网络实例化、预训练加载、动作采样推理以及最核心的 online 梯度更新流水线。
    """

    def __init__(self, state_dim, action_dim, device="cuda", lr=3e-4, gamma=0.99, tau=0.005, q_weight=0.05):
        self.device = device
        self.gamma = gamma  # 折扣因子：决定智能体是看重眼前利益还是长远利益
        self.tau = tau  # 软更新系数：控制目标网络追踪当前网络的平滑程度
        self.action_dim = action_dim
        self.q_weight = q_weight  # Q 值引导权重：决定 Actor 有多“听从” Critic 的指挥

        # --- 实例化 Actor (扩散模型侧) ---
        self.actor = ConditionalActor(state_dim, action_dim).to(self.device)
        self.scheduler = NoiseScheduler(num_train_timesteps=100)
        self.ema_actor = EMAModel(self.actor, decay=0.995)  # EMA 影子网络，用于稳定推理
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # --- 实例化 Critic (价值评估侧) ---
        self.critic = DoubleQCritic(state_dim, action_dim).to(self.device)
        self.critic_target = DoubleQCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())  # 初始时目标网络与当前网络对齐
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def load_pretrained_actor(self, path):
        """加载离线 BC 阶段训练好的预训练权重，为在线微调提供一个高起点的“准专家”"""
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.ema_actor.update(self.actor)
        print(f"✅ [DiffSACAgent] 成功挂载并同步预训练 Diffusion Actor: {path}")

    def select_action(self, state, sample_steps=5, explore=True):
        """
        环境交互推理函数。
        将当前环境观测转换为张量，并通过 DDIM 反向采样出具体的驾驶动作。
        explore=True 时引入 eta=0.1 的随机性用于训练探索；explore=False 时用于纯净评估。
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        eta = 0.1 if explore else 0.0
        # 始终使用平滑的 ema_actor 进行推理，保证驾驶动作的稳定性
        action_tensor = self.scheduler.sample(self.ema_actor, state_tensor, self.action_dim, sample_steps=sample_steps,
                                              eta=eta)
        return action_tensor[0].cpu().numpy()

    def update(self, replay_buffer, batch_size=256, sample_steps=5):
        """
        核心训练更新流水线 (Offline-to-Online Fine-tuning)。
        包含 Critic 防爆更新和 Actor 的非对称掩码更新。
        """
        # 🚨 接收第 6 个返回值：专家掩码 (is_expert)，1.0 代表专家数据，0.0 代表在线探索的菜鸟数据
        states, actions, rewards, next_states, dones, is_expert = replay_buffer.sample(batch_size)

        # ---------------------------------------------------
        # 1. 更新 Critic 网络 (引入防爆装甲)
        # ---------------------------------------------------
        with torch.no_grad():
            # 获取下一个状态对应的动作。加入目标策略平滑噪声 (Target Policy Smoothing)
            # 防止 Critic 对某个极端的尖峰动作产生过拟合
            next_actions = self.scheduler.sample(self.ema_actor, next_states, self.action_dim,
                                                 sample_steps=sample_steps, eta=0.0)
            noise = (torch.randn_like(next_actions) * 0.1).clamp(-0.25, 0.25)
            next_actions = torch.clamp(next_actions + noise, -5.0, 5.0)

            # 计算目标 Q 值，取双 Q 网络的最小值以对抗过估计
            target_q1, target_q2 = self.critic_target(next_states, next_actions)

            # 🚨 防爆一：强制截断环境传来的极端惩罚，防止 Critic 产生核弹级梯度而绝望崩溃
            safe_rewards = torch.clamp(rewards, min=-10.0, max=10.0)
            target_q = safe_rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(states, actions)

        # 🚨 防爆二：使用 Huber Loss (smooth_l1_loss) 替换常规的 MSE 均方误差。
        # 当 TD-error 过大时，Huber 会从平方惩罚退化为线性惩罚，防止极端 Q 误差撕裂网络权重
        critic_loss = F.smooth_l1_loss(current_q1, target_q) + F.smooth_l1_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # 梯度裁剪，上物理锁
        self.critic_optimizer.step()

        # ---------------------------------------------------
        # 2. 更新 Actor 网络 (引入不对称掩码的核心手术)
        # ---------------------------------------------------
        # 随机抽取一批扩散时间步，并给动作加上对应强度的真实白噪声
        t = torch.randint(0, self.scheduler.num_train_timesteps, (batch_size,), device=self.device).long()
        noise = torch.randn_like(actions)

        noisy_actions = self.scheduler.add_noise(actions, noise, t)
        pred_noise = self.actor(states, noisy_actions, t)  # 让 Actor 猜猜加了多少噪声

        # 🚨 核心掩码逻辑：
        # 计算未被平均的 BC Loss (每个样本独立计算预测误差)
        unreduced_bc_loss = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=1, keepdim=True)

        # 仅对 is_expert == 1 的样本计算克隆损失。
        # 坚决禁止 Actor 模仿自己在线探索时造成的撞车/违规动作！
        bc_loss = (unreduced_bc_loss * is_expert).sum() / (is_expert.sum() + 1e-8)

        # --- Q 引导项计算 ---
        # 利用 Actor 预测的噪声，倒推出干净的初始动作 a0
        alpha_bar_t = self.scheduler.alphas_cumprod.to(self.device)[t].unsqueeze(1)
        pred_a0 = (noisy_actions - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        pred_a0_clipped = torch.clamp(pred_a0, -5.0, 5.0)

        # 冻结 Critic 的梯度，拿着 Actor 刚想出来的动作去问 Critic 能拿多少分
        for param in self.critic.parameters():
            param.requires_grad = False

        q1, q2 = self.critic(states, pred_a0_clipped)
        q_value = torch.min(q1, q2)
        # 强化学习的目标是最大化 Q 值，在梯度下降框架中等同于最小化负的 Q 值
        q_loss = -q_value.mean()

        for param in self.critic.parameters():
            param.requires_grad = True

        # 最终损失合成：只克隆专家的优秀姿势，但接受 Critic 对全局经验的批判与指导
        actor_loss = bc_loss + self.q_weight * q_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ---------------------------------------------------
        # 3. 目标网络软更新 (Soft Update)
        # ---------------------------------------------------
        # 让目标 Critic 网络平滑地追踪当前 Critic 网络的最新知识
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 让 EMA 影子 Actor 吸收主干 Actor 的最新知识
        self.ema_actor.update(self.actor)

        return critic_loss.item(), actor_loss.item(), q_value.mean().item()