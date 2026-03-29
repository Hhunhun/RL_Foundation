import torch
import torch.nn.functional as F
import numpy as np

# 从我们刚才写的base中引入契约和网络
from core.base_agent import BaseAgent
from algorithms.sac.sac_nets import Actor, Critic


class SACAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, action_scale: float = 1.0,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 target_entropy: float = None):  # [修改点1] 移除了固定的 alpha
        super().__init__(state_dim, action_dim)

        self.gamma = gamma
        self.tau = tau

        # 自动检测是否有 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SAC Agent initialized on {self.device}")

        # 1. 实例化核心网络
        self.actor = Actor(state_dim, action_dim, action_scale=action_scale).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        # 2. 目标网络初始化 (强制将主网络的权重硬拷贝给 Target)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 3. 设置优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # ----------------------------------------------------
        # [修改点2] 新增：自适应温度系数 (Auto-Alpha)
        # ----------------------------------------------------
        # 目标熵通常设定为 -dim(A)，表示我们期望策略保留的最少随机性
        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = target_entropy

        # 我们优化的是 log_alpha，而不是 alpha 本身。
        # 为什么？因为网络更新可能产生负数，而 exp(log_alpha) 可以保证真实的 alpha 永远大于 0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        # 为 alpha 单独配备一个优化器
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        根据当前状态选择动作 (实现 BaseAgent 接口)
        """
        # 将 numpy 数组转为 torch 张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 在实际环境中执行时，不需要计算梯度
        with torch.no_grad():
            if evaluate:
                # 如果是测试评估模式，我们不探索，直接输出均值动作 (最确定的动作)
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean) * self.actor.action_scale
            else:
                # 训练模式，从分布中采样 (带有重参数化和随机性)
                action, _ = self.actor.sample(state_tensor)

        # 将结果从 GPU 拉回 CPU，并转回 numpy 数组，方便和物理环境交互
        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size: int):
        """
        核心训练逻辑 (实现 BaseAgent 接口)
        """
        # 1. 从经验池中抽取一批数据
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # 获取当前的 alpha 值 (从计算图中剥离，用于 Actor 和 Critic 的 Loss 计算)
        alpha = self.log_alpha.exp().detach()

        # ------------------------
        # Step 1: 更新 Critic (价值网络)
        # ------------------------
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            # [修改点3] 使用自适应的 alpha
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------------
        # Step 2: 更新 Actor (策略网络)
        # ------------------------
        # [修改点4 - 极度重要] 冻结 Critic 的梯度，防止算力泄露
        # 因为求 min_q_new 时必须经过 Critic 的网络层，冻结后就不会白白计算 Critic 权重的梯度了
        for param in self.critic.parameters():
            param.requires_grad = False

        new_action, log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)

        # [修改点5] 使用自适应的 alpha
        actor_loss = (alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # [修改点4 - 延续] 释放 Critic 的梯度，准备下一次 Update 循环
        for param in self.critic.parameters():
            param.requires_grad = True

        # ------------------------
        # Step 3: 更新 Alpha (自适应温度调节)
        # ------------------------
        # [修改点6] 我们希望 log_prob 的均值趋近于 target_entropy
        # 注意这里 log_prob 必须 detach()，我们只是调整 alpha 适应它，绝不能让梯度传回 Actor
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ------------------------
        # Step 4: 软更新目标网络 (Soft Update)
        # ------------------------
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # [修改点7] 将 alpha 的状态也返回，极其建议在 Tensorboard 中监控它！
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha.item()
        }

    def save_model(self, path: str):
        """保存模型权重"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha  # [修改点8] 把 alpha 状态也保存下来
        }, path)
        print(f"模型已成功保存至: {path}")

    def load_model(self, path: str):
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])

        # [修改点9] 读取 alpha 状态
        if 'log_alpha' in checkpoint:
            with torch.no_grad():
                self.log_alpha.copy_(checkpoint['log_alpha'])

        print(f"模型权重已从 {path} 加载")