import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 导入扩散核心组件
from algorithms.diffusion_sac.diffusion_model import ConditionalActor, NoiseScheduler


# ==========================================
# 标准的 SAC 双 Q 网络 (Critic)
# ==========================================
class DoubleQCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1 网络
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2 网络 (用于缓解 Q 值过估计)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


# ==========================================
# 混合强化学习代理 (Diffusion-SAC Agent)
# ==========================================
class DiffSACAgent:
    def __init__(self, state_dim, action_dim, device="cuda", lr=3e-4, gamma=0.99, tau=0.005):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        # 1. 实例化扩散 Actor 和调度器
        self.actor = ConditionalActor(state_dim, action_dim).to(self.device)
        self.scheduler = NoiseScheduler(num_train_timesteps=100, device=self.device)

        # 🚨 [关键修复]: 单独压低 Actor 的学习率，防止遗忘专家经验
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-5)

        # 2. 实例化双 Q Critic 及其目标网络
        self.critic = DoubleQCritic(state_dim, action_dim).to(self.device)
        self.critic_target = DoubleQCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 🚨 [关键修复]: 保持 Critic 学习率不变，且必须在 Critic 实例化后进行优化器绑定
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def load_pretrained_actor(self, path):
        """加载阶段二离线预训练好的 Behavior Cloning 权重"""
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✅ 成功挂载预训练 Diffusion Actor: {path}")

    def select_action(self, state, sample_steps=5):
        """在线交互时的动作选择 (利用 DDIM 极速采样，无需梯度)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # 调用 scheduler.sample，内部自带 @torch.no_grad()
        action_tensor = self.scheduler.sample(self.actor, state_tensor, self.action_dim, sample_steps=sample_steps)
        return action_tensor[0].cpu().numpy()

    def update(self, replay_buffer, batch_size=256, sample_steps=5):
        """从经验池中采样并更新 Critic 和 Diffusion Actor"""

        # 你的 ReplayBuffer.sample() 已经直接返回了正确的 Tensor，直接解包
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # 确保它们在当前 Agent 的计算设备上
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ---------------------------------------------------
        # 1. 更新 Critic 网络 (价值评估)
        # ---------------------------------------------------
        with torch.no_grad():
            # 使用扩散模型预测下一步的动作 (无梯度采样)
            next_actions = self.scheduler.sample(self.actor, next_states, self.action_dim, sample_steps=sample_steps)
            # 计算目标 Q 值
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # [防爆盾]: 保护 Critic 不被异常奖励摧毁
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---------------------------------------------------
        # 2. 更新 Diffusion Actor 网络 (Q值引导的梯度反传)
        # ---------------------------------------------------
        # 我们在这里手写一个轻量级的可微 DDIM 采样循环，保留梯度图让 Critic 打分
        pred_action = torch.randn((batch_size, self.action_dim), device=self.device)
        step_indices = torch.linspace(self.scheduler.num_train_timesteps - 1, 0, sample_steps, dtype=torch.long,
                                      device=self.device)

        for i in range(sample_steps):
            t = step_indices[i]
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            pred_noise = self.actor(states, pred_action, t_batch)

            alpha_bar_t = self.scheduler.alphas_cumprod[t]
            alpha_bar_t_prev = self.scheduler.alphas_cumprod[
                step_indices[i + 1]] if i < sample_steps - 1 else torch.tensor(1.0, device=self.device)

            # 可微去噪公式
            pred_original_action = (pred_action - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

            # 🚨【绝密防爆补丁：必须在这里把预测出的原始动作物理截断】🚨
            pred_original_action = torch.clamp(pred_original_action, -1.0, 1.0)

            dir_xt = torch.sqrt(1.0 - alpha_bar_t_prev) * pred_noise
            pred_action = torch.sqrt(alpha_bar_t_prev) * pred_original_action + dir_xt

        # 🚨【保险起见：循环结束后对最终产生的动作再 Clamp 一次】🚨
        pred_action = torch.clamp(pred_action, -1.0, 1.0)

        # 此时得到的 pred_action 带有完整梯度流，让 Critic 评估它
        q_values, _ = self.critic(states, pred_action)

        # 🚨 [抢救补丁 1+2：延迟更新 + Q值缩放] 🚨
        # 只有当经验池里的数据量足够多时（Critic 眼睛治好了），才允许更新 Actor
        if replay_buffer.size > 10000:
            # 引入缩放系数 0.01，强行把巨大梯度压缩到 Diffusion 习惯的微调级别！
            actor_loss = -q_values.mean() * 0.01

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
        else:
            # 如果 Critic 还在热身，Actor 就不更新，loss 记为 0
            actor_loss = torch.tensor(0.0)

        # ---------------------------------------------------
        # 3. 目标网络软更新 (Soft Update)
        # ---------------------------------------------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()