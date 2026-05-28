"""
Module: Soft Actor-Critic (SAC) Agent Base
Description:
    本模块实现了基于最大熵强化学习 (Maximum Entropy RL) 的 Soft Actor-Critic 算法。
    在本项目中，SAC 扮演双重核心角色：
    1. 作为自动驾驶规控任务的强大连续控制基线 (Baseline)。
    2. 作为 Diffusion-SAC 架构的“离线专家”，通过环境交互收集高质量先验轨迹数据。

Key Features:
    - Maximum Entropy Objective: 在最大化累计收益的同时最大化策略熵，鼓励宽泛探索。
    - Twin Delayed Critic: 采用双重 Q 网络取极小值，消除价值过估计 (Overestimation Bias)。
    - Adaptive Temperature (Alpha): 动态调节熵正则化系数，平衡探索 (Exploration) 与利用 (Exploitation)。
"""

import torch
import torch.nn.functional as F
import numpy as np

# 从我们刚才写的base中引入契约和网络
from core.base_agent import BaseAgent
from algorithms.sac.sac_nets import Actor, Critic


class SACAgent(BaseAgent):
    """
    SAC 核心智能体类。
    基于连续动作空间的 Actor-Critic 拓扑架构设计。
    """
    def __init__(self, state_dim: int, action_dim: int, action_scale: float = 1.0,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 target_entropy: float = None):  # [修改点1] 移除了固定的 alpha
        super().__init__(state_dim, action_dim)

        self.gamma = gamma
        self.tau = tau

        # 自动检测是否有 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SAC Agent initialized on {self.device}")

        # 1. 实例化核心网络拓扑 (Actor & Twin Critic)
        self.actor = Actor(state_dim, action_dim, action_scale=action_scale).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        # 2. 目标网络初始化 (确保目标价值评估的初始状态对齐)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 3. 设置优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # ----------------------------------------------------
        # 自适应温度系数 (Adaptive Temperature / Auto-Alpha)
        # ----------------------------------------------------
        # 物理意义：在自动驾驶连续控制中，固定的温度超参数极易导致策略收敛过早或过度随机。
        # 目标熵 (Target Entropy) 默认设定为 \mathcal{H} = -dim(\mathcal{A})，代表策略分布应保留的最小随机性底线。
        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = target_entropy

        # 数学技巧：优化对数温度 \log \alpha 而非 \alpha 本身。
        # 保证在对偶梯度下降期间，实际的温度系数 \alpha = \exp(\log \alpha) 严格恒正，维持有效的熵正则化。
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        策略前向推理逻辑 (Forward Inference)。
        
        Args:
            state: 当前环境的全维观测状态 S_t
            evaluate: 推理模式标志。
                      - True: 执行最大后验概率估计 (MAP)，直接输出高斯分布的均值，提供稳定可靠的安全驾驶控制。
                      - False: 启用重参数化技巧 (Reparameterization Trick) 的随机采样，为经验池注入兼具探索度与平滑性的轨迹。
                      
        Returns:
            action: 映射至物理执行器边界的控制指令向量 A_t
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 阻断计算图，前向推理期间不保留梯度缓存
        with torch.no_grad():
            if evaluate:
                # 纯净评估模式：剥离随机性，输出确定性贪婪策略 \mu_\theta(s)
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean) * self.actor.action_scale
            else:
                # 探索采样模式：输出 a_t \sim \pi_\theta(\cdot|s_t)
                action, _ = self.actor.sample(state_tensor)

        # 降维并桥接底盘执行器接口
        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size: int):
        """
        核心动力学优化中枢 (Core Dynamics Optimization).
        执行 Soft Actor-Critic 的交替算子更新，包含基于软贝尔曼方程的价值评估与基于重参数化策略梯度的策略提升。
        
        Args:
            replay_buffer: 经验回放池，提供独立同分布 (i.i.d) 的马尔可夫转移序列
            batch_size: 微批次样本容量
            
        Returns:
            包含各项损失函数与温度系数标量的字典，供日志系统挂载监控
        """
        # 从经验池抽取转移元组 (Transition Tuples)
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # 强制将底层经验池数据推送至计算设备 (GPU/CPU) 并确保张量维度对齐
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        if reward.dim() == 1: reward = reward.unsqueeze(1)
        if done.dim() == 1: done = done.unsqueeze(1)

        # 提取动态温度标量 \alpha，剥离计算图以切断交叉反向传播
        alpha = self.log_alpha.exp().detach()

        # ----------------------------------------------------
        # 步骤 1: 价值网络参数更新 (Critic Evaluation Step)
        # ----------------------------------------------------
        # 基于带熵正则化的软贝尔曼残差方程 (Soft Bellman Residual Equation):
        # 目标价值 y(s,a) = r + \gamma (1 - d) \mathbb{E}_{s'} [ \min_{i=1,2} Q_{\phi_i'}(s', a') - \alpha \log \pi_\theta(a'|s') ]
        # 其中极小值截断机制有效消除了连续控制场景中极易诱发震荡的价值过估计 (Overestimation Bias) 风险。
        with torch.no_grad():
            # 目标策略平滑采样
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------------------------------------
        # 步骤 2: 策略网络参数更新 (Actor Improvement Step)
        # ----------------------------------------------------
        # [工程极客优化：梯度冻结 Gradient Freezing]
        # 优化策略网络时，需要依靠 \nabla_a Q(s,a) \nabla_\theta \pi(s) 使得动作梯度穿透 Critic 网络。
        # 此时显式阻断 Critic 层的参数梯度累积 (requires_grad = False)，
        # 不仅避免了对其权重产生无效扰动，更能极大缩减反向传播的算力开销，显著提升系统吞吐量。
        for param in self.critic.parameters():
            param.requires_grad = False

        # 基于当前最新策略进行重参数化采样 (Reparameterization Trick)
        new_action, log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)

        # 最小化 KL 散度目标，等价于最大化熵增强的预期回报：
        # \mathcal{J}_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}} [ \alpha \log \pi_\theta(f_\theta(\epsilon; s)|s) - Q_\phi(s, f_\theta(\epsilon; s)) ]
        actor_loss = (alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 释放 Critic 参数锁，恢复下一次循环的正常价值评估网络更新
        for param in self.critic.parameters():
            param.requires_grad = True

        # ----------------------------------------------------
        # 步骤 3: 自适应温度系数调节 (Temperature Alpha Tuning)
        # ----------------------------------------------------
        # 基于对偶梯度下降 (Dual Gradient Descent) 求解带约束的最优化问题：
        # \mathcal{J}(\alpha) = \mathbb{E}_{a \sim \pi_t} [ -\alpha (\log \pi_t(a|s) + \bar{\mathcal{H}}) ]
        # 动态驱动策略的当前信息熵收敛于预设的先验目标底线 (Target Entropy)。
        # 注意：需切断 log_prob 梯度，仅让 \alpha 单独负责调节任务。
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ----------------------------------------------------
        # 步骤 4: 目标网络平滑追踪 (Target Network Soft Update)
        # ----------------------------------------------------
        # 引入 Polyak 平均 (Polyak Averaging) 衰减更新公式：\phi' \leftarrow \tau \phi + (1 - \tau) \phi'
        # 在确保目标价值估计网络不断吸收新知识的同时，强力抑制其早期的剧烈震荡。
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # 输出优化状态全景监控数据
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha.item()
        }

    def save_model(self, path: str):
        """
        模型序列化接口。
        持久化挂载 Actor策略网络、Twin Critic价值网络及其 Target影子网络 的底层张量字典，
        同时存留自适应温度系数参数，以确保 Offline-to-Online 环境中的严格时序可复现性。
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha
        }, path)
        print(f"模型已成功保存至: {path}")

    def load_model(self, path: str):
        """
        模型反序列化接口。
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])

        if 'log_alpha' in checkpoint:
            with torch.no_grad():
                self.log_alpha.copy_(checkpoint['log_alpha'])

        print(f"模型权重已从 {path} 加载")