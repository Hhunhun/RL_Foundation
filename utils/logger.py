"""
Module: TensorBoard Logging Utility (TensorBoard 日志记录工具)
Description:
    本模块提供了一个标准化的 TensorBoard 日志记录接口。
    通过为每次独立的训练运行创建唯一的、带时间戳的日志目录，确保了实验的可追溯性与结果的可复现性。
    核心功能是封装 PyTorch 的 SummaryWriter，并修复其在特定操作系统 (如 Windows) 下的实时刷新问题。

Key Features:
    - Unique Run Identification: 自动生成基于环境名、算法名与时间戳的日志文件夹，避免实验数据交叉污染。
    - Real-time Flushing: 在每次标量记录后强制执行 I/O 刷新，确保在任何环境下都能实时观察到训练曲线，
      解决了 TensorBoard 在 Windows 上常见的“只有一个点”或“曲线不更新”的顽固问题。
    - Simplified API: 提供极简的 `log_scalar` 接口，解耦上层训练逻辑与底层日志系统的实现细节。
"""

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    TensorBoard 日志记录器封装类。
    为每一次独立的实验运行 (Run) 实例化一个 SummaryWriter，并管理其生命周期。
    """
    def __init__(self, log_dir="outputs/logs", env_name="Unknown"):
        """
        初始化日志记录器。

        Args:
            log_dir (str): 存放所有日志的根目录。
            env_name (str): 当前实验的环境/算法名称，用于构建可读的子目录。
        """
        # 利用当前时间戳生成唯一的运行名称，确保实验结果不被覆盖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 🚨 [核心修复] 之前的 env_name 包含了 SAC_ 前缀，这里直接使用，不再重复添加
        self.run_dir = os.path.join(log_dir, f"{env_name}_SAC_{timestamp}")

        # 实例化 TensorBoard 底层写入器
        self.writer = SummaryWriter(log_dir=self.run_dir)
        print(f"📊 TensorBoard 日志系统已就绪，数据保存在: {self.run_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        """
        记录单一的标量数据 (如 Reward, Loss) 到 TensorBoard。

        Args:
            tag (str): 数据标签，如 'Reward/Episode_Reward'。
            value (float): 要记录的数值。
            step (int): 对应的训练步数或回合数 (X轴坐标)。
        """
        self.writer.add_scalar(tag, value, step)
        # 强制将内存中的日志缓冲区数据刷入硬盘文件！
        # 这一操作是解决 Windows 环境下 TensorBoard 曲线不实时更新或只有一个点的关键。
        self.writer.flush()

    def close(self):
        """
        安全关闭文件写入流。
        在训练主循环结束后调用，确保所有缓存的日志数据都被完整写入磁盘。
        """
        self.writer.close()