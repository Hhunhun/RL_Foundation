import abc
import numpy as np


class BaseAgent(abc.ABC):
    """
    强化学习算法的顶级抽象基类。
    强制所有算法实现统一的调用接口。
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abc.abstractmethod
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        pass

    @abc.abstractmethod
    def update(self, replay_buffer, batch_size: int):
        pass

    @abc.abstractmethod
    def save_model(self, path: str):
        pass

    @abc.abstractmethod
    def load_model(self, path: str):
        pass