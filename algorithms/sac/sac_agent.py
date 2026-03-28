import torch
import torch.nn.functional as F
import numpy as np

# 从我们刚才写的基建中引入契约和网络
from core.base_agent import BaseAgent
from algorithms.sac.sac_nets import Actor, Critic


class SACAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, action_scale: float = 1.0,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005, alpha: float = 0.2):
        super().__init__(state_dim, action_dim)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # 也就是我们在推导里说的“温度系数”，控制探索力度

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

        # ------------------------
        # Step 1: 更新 Critic (价值网络)
        # ------------------------
        with torch.no_grad():
            # 获取下一个状态的动作和对数概率 (当前 Actor 的实时反应)
            next_action, next_log_prob = self.actor.sample(next_state)
            # 获取目标网络的两个 Q 值预测
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            # 取最小值防止高估，并减去熵的惩罚项
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            # 计算最终的贝尔曼目标值 (注意这里的 1 - done 逻辑)
            target_q = reward + (1 - done) * self.gamma * target_q

        # 获取当前网络对 Q 值的预测
        current_q1, current_q2 = self.critic(state, action)

        # 计算 Critic 的均方误差
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # 优化 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------------
        # Step 2: 更新 Actor (策略网络)
        # ------------------------
        # 根据当前状态，重新生成动作及其对数概率 (因为网络刚刚可能变了)
        new_action, log_prob = self.actor.sample(state)

        # 看看更聪明的 Critic 给新动作打多少分
        q1_new, q2_new = self.critic(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)

        # Actor 想要最大化 Q 值，同时最大化熵 (也就是最小化它们的负值)
        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        # 优化 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------
        # Step 3: 软更新目标网络 (Soft Update)
        # ------------------------
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # 返回日志数据用于绘图监控
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }

    def save_model(self, path: str):
        """保存模型权重"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
        }, path)
        print(f"✅ 模型已成功保存至: {path}")

    def load_model(self, path: str):
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        print(f"🔄 模型权重已从 {path} 加载")