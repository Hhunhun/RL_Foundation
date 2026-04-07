"""
数据流转与经验池核心模块 (Data Management & Replay Buffers)

此模块是 Offline-to-Online 强化学习架构的数据底座，主要包含三个核心组件：
1. DataNormalizer: 数据归一化器，保证神经网络输入的稳定性。
2. ExpertDataset: 静态数据集，专供第一阶段的纯离线行为克隆 (BC) 预训练使用。
3. MixedReplayBuffer: 混合经验池，专供第二阶段的在线微调使用。它将静态的专家数据与
   动态的在线探索数据混合，并输出特征掩码 (is_expert)，以支持非对称的 Actor-Critic 更新。
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class DataNormalizer:
    """
    数据归一化工具类。
    用于将状态 (states) 和动作 (actions) 缩放到均值为 0、方差为 1 的标准区间，
    这对于提高神经网络（尤其是 Diffusion 模型）的训练稳定性和收敛速度至关重要。
    """

    def __init__(self, data: np.ndarray, eps: float = 1e-4):
        # 初始化时，根据传入的一批基准数据（通常是专家数据）计算均值和标准差
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + eps  # 加上微小的 eps 防止除以 0

    def normalize(self, data):
        # 将原始数据转化为归一化数据，兼容 PyTorch 张量和 NumPy 数组
        if isinstance(data, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=torch.float32, device=data.device)
            std = torch.tensor(self.std, dtype=torch.float32, device=data.device)
            return (data - mean) / std
        return (data - self.mean) / self.std

    def unnormalize(self, data):
        # 逆操作：将神经网络输出的归一化数据还原为真实物理环境中的数值
        if isinstance(data, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=torch.float32, device=data.device)
            std = torch.tensor(self.std, dtype=torch.float32, device=data.device)
            return data * std + mean
        return data * self.std + self.mean


class ExpertDataset(Dataset):
    """
    专家数据集包装器 (基于 PyTorch Dataset)。
    仅用于第一阶段：将离线采集的专家数据打包，供 DataLoader 批量读取，
    用来对 Diffusion Actor 进行行为克隆 (BC) 预训练。
    """

    def __init__(self, data_path: str):
        # 加载本地 .npz 格式的专家数据
        data = np.load(data_path)
        raw_states, raw_actions = data['observations'], data['actions']

        # 实例化并固化数据的归一化标准
        self.state_normalizer = DataNormalizer(raw_states)
        self.action_normalizer = DataNormalizer(raw_actions)

        # 提前将所有数据进行归一化并转为 Tensor，加速后续训练时的读取
        self.states = torch.tensor(self.state_normalizer.normalize(raw_states), dtype=torch.float32)
        self.actions = torch.tensor(self.action_normalizer.normalize(raw_actions), dtype=torch.float32)
        self.num_samples = len(self.states)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class MixedReplayBuffer:
    """
    混合经验回放池 (核心组件)。
    用于第二阶段在线 RL 微调。它在内存中同时维护两份数据：
    1. 固定的专家数据 (Expert Data): 不可变，永久保存。
    2. 动态滚动的在线数据 (Online Data): 环形缓冲区，使用 FIFO 规则覆盖。
    在采样时按固定比例混合这两者，并为每个样本打上标签 (is_expert)。
    """

    def __init__(self, expert_data_path: str, max_online_size: int = int(1e5), device: str = "cpu"):
        self.device = torch.device(device)

        # --- 1. 加载并处理静态专家数据 ---
        expert_data = np.load(expert_data_path)
        raw_expert_states = expert_data['observations']
        raw_expert_actions = expert_data['actions'].astype(np.float32)

        # 兼容不同命名习惯的数据集字段
        self.expert_rewards = expert_data.get('rewards',
                                              expert_data.get('reward', np.zeros((len(raw_expert_states), 1))))
        self.expert_next_states = expert_data.get('next_observations',
                                                  expert_data.get('next_states', raw_expert_states))
        self.expert_dones = expert_data.get('terminals',
                                            expert_data.get('dones', np.zeros((len(raw_expert_states), 1))))

        # 确保奖励和完成标志的维度是列向量 (N, 1)
        if self.expert_rewards.ndim == 1: self.expert_rewards = self.expert_rewards.reshape(-1, 1)
        if self.expert_dones.ndim == 1: self.expert_dones = self.expert_dones.reshape(-1, 1)

        self.expert_size = len(raw_expert_states)
        self.state_dim, self.action_dim = raw_expert_states.shape[1], raw_expert_actions.shape[1]

        # 核心逻辑：整个系统的归一化基准完全由专家数据决定
        self.state_normalizer = DataNormalizer(raw_expert_states)
        self.action_normalizer = DataNormalizer(raw_expert_actions)

        # 将专家数据归一化后长期存放在内存中
        self.expert_states = self.state_normalizer.normalize(raw_expert_states).astype(np.float32)
        self.expert_actions = self.action_normalizer.normalize(raw_expert_actions).astype(np.float32)
        self.expert_next_states = self.state_normalizer.normalize(self.expert_next_states).astype(np.float32)

        # --- 2. 初始化动态在线数据存储区 ---
        self.max_online_size = max_online_size
        self.online_ptr, self.online_size = 0, 0  # 指针与当前容量

        # 预先分配内存以提高运行效率
        self.online_states = np.zeros((max_online_size, self.state_dim), dtype=np.float32)
        self.online_actions = np.zeros((max_online_size, self.action_dim), dtype=np.float32)
        self.online_rewards = np.zeros((max_online_size, 1), dtype=np.float32)
        self.online_next_states = np.zeros((max_online_size, self.state_dim), dtype=np.float32)
        self.online_dones = np.zeros((max_online_size, 1), dtype=np.float32)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: float):
        """
        向经验池添加环境交互产生的新数据。
        注意：存入前必须使用专家数据的 Normalizer 进行归一化，保持数据尺度一致。
        """
        self.online_states[self.online_ptr] = self.state_normalizer.normalize(state)
        self.online_actions[self.online_ptr] = self.action_normalizer.normalize(action)
        self.online_rewards[self.online_ptr] = reward
        self.online_next_states[self.online_ptr] = self.state_normalizer.normalize(next_state)
        self.online_dones[self.online_ptr] = done

        # 环形缓冲区指针逻辑：满了就覆盖最旧的数据
        self.online_ptr = (self.online_ptr + 1) % self.max_online_size
        self.online_size = min(self.online_size + 1, self.max_online_size)

    def sample(self, batch_size: int, expert_ratio: float = 0.5):
        """
        核心采样逻辑：按比例抽取专家数据和在线数据。
        返回的第 6 个元素 'is_expert' 是一个掩码，对非对称网络更新至关重要。
        """
        # 计算分别需要采样多少个专家样本和在线样本
        expert_batch = int(batch_size * expert_ratio)
        online_batch = batch_size - expert_batch

        # 处理训练刚开始，在线数据还不够的情况
        if self.online_size == 0:
            expert_batch, online_batch = batch_size, 0
        elif self.online_size < online_batch:
            online_batch, expert_batch = self.online_size, batch_size - self.online_size

        # --- 抽取专家数据 ---
        exp_idx = np.random.randint(0, self.expert_size, size=expert_batch)
        s = self.expert_states[exp_idx]
        a = self.expert_actions[exp_idx]
        r = self.expert_rewards[exp_idx]
        ns = self.expert_next_states[exp_idx]
        d = self.expert_dones[exp_idx]

        # 🚨 生成专家数据掩码数组：1.0 代表此样本是专家动作，0.0 代表是在线探索动作
        # 前面 expert_batch 个位置填 1.0，后面默认是 0.0
        is_expert = np.zeros((expert_batch + online_batch, 1), dtype=np.float32)
        is_expert[:expert_batch] = 1.0

        # --- 抽取在线数据并拼接 ---
        if online_batch > 0:
            on_idx = np.random.randint(0, self.online_size, size=online_batch)
            s = np.concatenate([s, self.online_states[on_idx]], axis=0)
            a = np.concatenate([a, self.online_actions[on_idx]], axis=0)
            r = np.concatenate([r, self.online_rewards[on_idx]], axis=0)
            ns = np.concatenate([ns, self.online_next_states[on_idx]], axis=0)
            d = np.concatenate([d, self.online_dones[on_idx]], axis=0)

        # 将所有 Numpy 数组转为目标设备上的 PyTorch Tensor
        return (torch.FloatTensor(s).to(self.device),
                torch.FloatTensor(a).to(self.device),
                torch.FloatTensor(r).to(self.device),
                torch.FloatTensor(ns).to(self.device),
                torch.FloatTensor(d).to(self.device),
                torch.FloatTensor(is_expert).to(self.device))