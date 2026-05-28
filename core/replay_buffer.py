"""
Module: Standard Experience Replay Buffer (标准经验回放池)
Description:
    本模块实现了异策略强化学习 (Off-policy RL) 的核心数据底座。
    通过构建定长的环形缓冲区 (Ring Buffer)，以马尔可夫转移元组 (Markov Transition Tuples) 
    的形式持久化智能体与物理环境的交互序列。

Key Features:
    - I.I.D. Guarantee: 通过对历史轨迹的均匀随机采样，打破序列决策中强烈的时序自相关性 
      (Temporal Autocorrelation)，满足深度神经网络随机梯度下降对独立同分布 (i.i.d.) 的假设契约。
    - Memory Pre-allocation: 采用底层连续内存块预分配机制 (NumPy Zero-initialization)，
      彻底规避高频交互探索期间由动态数组扩容引发的内存碎片化与算力抖动。
    - Zero-copy Device Transfer: 在采样算子端实现批次级张量装载与异构计算设备 (GPU) 路由，优化总线带宽。
"""

import numpy as np
import torch

class ReplayBuffer:
    """
    定长环形经验回放池。
    基于预分配的 NumPy 矩阵构建，维持 O(1) 的状态写入复杂度，同时约束系统内存的最高水位线。
    """
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6), device: torch.device = torch.device('cpu')):
        self.max_size = max_size
        self.ptr = 0  # 环形写入游标 (Write Pointer)
        self.size = 0 # 物理内存当前驻留量 (Current Occupancy)
        self.device = device

        # =====================================================================
        # 内存预分配 (Memory Pre-allocation)
        # 物理意义：拒绝 Python List 原生的动态 Append 操作。在内存中预先开辟一整块连续空间，
        # 构建 [Max_Size, Dim] 的矩阵，确保指针覆写时绝对的 O(1) 效率与零内存抖动。
        # =====================================================================
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: float):
        """
        装载单步马尔可夫转移元组 (Markov Transition Tuple): (S_t, A_t, R_t, S_{t+1}, D_t)。
        利用指针覆盖机制进行内存块的原地复写。
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        # 环形拓扑算子 (Ring Buffer Topology Operator)
        # 当游标越过最大容量边界时，自动对齐至起始地址 0，对久远的历史数据进行 FIFO 淘汰。
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        """
        经验池均匀抽样算子 (Uniform Minibatch Sampler)。
        基于蒙特卡洛积分的离散实现，从历史经验中抽取出无时序相关性的 Mini-batch 批次，
        用于近似计算目标函数在数据分布 \mathcal{D} 上的数学期望 \mathbb{E}_{(s,a) \sim \mathcal{D}}。
        """
        # 生成独立同分布的随机索引
        ind = np.random.randint(0, self.size, size=batch_size)

        # 批次装载、浮点数精度对齐与异构设备流转
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )