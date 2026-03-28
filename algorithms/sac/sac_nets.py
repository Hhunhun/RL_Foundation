import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# 权重初始化工具函数
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Critic(nn.Module):
    """
    双重 Q 网络 (Twin Q-Network)
    用于评估 Q(s, a)，并取最小值以抑制 Q 值高估
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # Q1 网络架构
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 网络架构
        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        # 拼接状态和动作
        sa = torch.cat([state, action], dim=1)

        # Q1 前向传播
        q1 = F.relu(self.linear1(sa))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        # Q2 前向传播
        q2 = F.relu(self.linear4(sa))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)

        return q1, q2


class Actor(nn.Module):
    """
    高斯策略网络 (Gaussian Policy Network)
    输出动作的正态分布参数 (均值和对数标准差)，并支持重参数化采样
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, action_scale=1.0):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # 均值输出层
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        # 对数标准差输出层
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

        # 动作缩放比例 (将 tanh 的 [-1, 1] 映射到环境真实边界)
        self.action_scale = torch.tensor(action_scale)
        # 限制 log_std 的范围，防止计算出极其夸张的方差导致 NaN
        self.LOG_STD_MAX = 2.0
        self.LOG_STD_MIN = -20.0

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, state):
        """
        根据状态采样动作，并计算动作的对数概率 (用于计算熵)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # 构造正态分布
        normal = Normal(mean, std)

        # 重参数化采样 (等价于 mean + std * noise)
        x_t = normal.rsample()

        # 使用 tanh 压缩动作到 [-1, 1]，再乘以 scale
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale

        # 计算经过 tanh 变换后的对数概率 log_pi
        # (这里涉及一个雅可比行列式的修正，是 SAC 官方实现的标准做法)
        log_prob = normal.log_prob(x_t)
        # 修正项: log(1 - tanh^2(x))，加上 1e-6 防止 log(0)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob