import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6), device: torch.device = torch.device('cpu')):
        self.max_size = max_size
        self.ptr = 0  # 写入指针
        self.size = 0 # 当前已存入的数据量
        self.device = device

        # 使用 NumPy 预先分配连续内存，大幅提高读写速度
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: float):
        """将一次真实环境交互数据存入池中"""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        # 环形缓冲区逻辑：写满后从头开始覆盖最老的数据
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        """随机采样一个 Batch 的数据，并直接转移到计算设备(CPU/GPU)上的张量"""
        # 生成 batch_size 个随机索引
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )