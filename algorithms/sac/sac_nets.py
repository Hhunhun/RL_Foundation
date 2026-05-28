"""
Module: Soft Actor-Critic Neural Architectures (SAC Nets)
Description:
    本模块构建了 Soft Actor-Critic 算法底层的深层神经网络拓扑，包含基于重参数化采样
    的挤压高斯策略网络 (Squashed Gaussian Policy) 与双重动作价值评估网络 (Twin Q-Networks)。
    这些结构为连续空间的自动驾驶规控提供了兼具平滑性、探索度与稳健性的数学表征基础。

Key Components:
    - Xavier/Glorot Initialization: 维持深层特征传递过程中的激活方差一致性，防止梯度弥散/爆炸。
    - Twin Critic: 独立的双通路 Q 评估计算图，为极小值截断抑制过估计提供物理载体。
    - Reparameterized Actor: 利用 \tanh 映射将无界的高斯潜变量平滑挤压至闭区间物理动作域，
      并严格执行基于雅可比行列式 (Jacobian Determinant) 的概率密度修正。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def weights_init_(m):
    """
    网络层权重正交/均匀初始化函数。
    物理意义：依据输入输出维度缩放初始权重方差，保证前向传播与反向梯度的能量守恒，
    使得极其容易发散的强化学习网络在早期训练阶段维持稳定的梯度流。
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Critic(nn.Module):
    """
    独立同分布的双路动作价值网络 (Twin Action-Value Networks)。
    基于 Clipped Double Q-Learning 范式设计，构建两套独立初始化的计算图以解耦状态-动作空间的高维特征映射。
    目标：精准逼近真实的软期望回报 Q(s, a)，为策略提升提供无过估计偏差的梯度引导。
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # 计算支路 Q1: \phi_1 网络拓扑
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # 计算支路 Q2: \phi_2 网络拓扑
        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        """
        价值前向评估。
        将状态向量 S_t 与动作向量 A_t 拼接为联合特征，同时并行推送至双支路进行独立打分。
        """
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.linear1(sa))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        q2 = F.relu(self.linear4(sa))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)

        return q1, q2


class Actor(nn.Module):
    """
    挤压高斯策略网络 (Squashed Gaussian Policy Network)。
    以当前环境状态 S_t 为条件输入，预测连续动作分布的潜变量参数 (均值 \mu 与对数标准差 \log \sigma)。
    全面支持重参数化采样 (Reparameterization Trick) 以建立可微的端到端梯度流。
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, action_scale=1.0):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # 对角高斯分布参数流：均值 \mu_\theta(s)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        # 对角高斯分布参数流：对数标准差 \log \sigma_\theta(s)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

        # 物理边界挂载 (Action Scale Bound)
        # 作为不可求导的 Buffer 层常驻内存，负责将非线性挤压函数 (\tanh) 的 [-1, 1] 输出投射到实际的车辆底层控制边界。
        self.register_buffer('action_scale', torch.tensor(action_scale, dtype=torch.float32))

        # 截断对数方差域。物理意义：极端的极小方差会导致对数概率运算产生 NaN (梯度黑洞)；
        # 极大的方差会导致数值溢出。施加硬限幅护栏保护网络存活。
        self.LOG_STD_MAX = 2.0
        self.LOG_STD_MIN = -20.0

    def forward(self, state):
        """提取隐层特征，输出潜变量分布的数字特征"""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, state):
        """
        可微概率采样模块 (Differentiable Probability Sampler)。
        执行动作生成及对应的对数概率 \log \pi(a|s) 计算，用于计算 KL 散度和熵正则化目标。
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)

        # 重参数化 (Reparameterization Trick): 
        # x_t = \mu + \sigma \cdot \epsilon, where \epsilon \sim \mathcal{N}(0, I)
        # 将随机节点 \epsilon 剥离于计算图外，确保 \mu 和 \sigma 的梯度完全可传导。
        x_t = normal.rsample()

        # 动作挤压 (Squashing): y_t = \tanh(x_t)
        # 将无界的正态变量映射至紧致的开闭域，符合真实环境控制信号 (如油门/刹车)。
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale

        # ----------------------------------------------------
        # 雅可比概率修正 (Jacobian Log-Probability Correction)
        # ----------------------------------------------------
        # 因为我们对随机变量施加了可逆的非线性变换 \tanh，根据概率论中的随机变量变量替换积分定理，
        # 必须利用转换矩阵的雅可比行列式来修正分布密度：
        # \log \pi(a|s) = \log p(x|s) - \sum \log \Big( 1 - \tanh^2(x) \Big)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob