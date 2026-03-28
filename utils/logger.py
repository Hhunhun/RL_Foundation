import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir="output/logs", env_name="Unknown"):
        # 利用当前时间戳生成唯一的运行名称，防止不同实验的数据混在一起
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"{env_name}_SAC_{timestamp}")

        # 实例化 TensorBoard 记录器
        self.writer = SummaryWriter(log_dir=self.run_dir)
        print(f"📊 TensorBoard 日志系统已就绪，数据保存在: {self.run_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        """记录单一的标量数据 (如 Reward, Loss)"""
        self.writer.add_scalar(tag, value, step)

    def close(self):
        """训练结束后安全关闭文件流"""
        self.writer.close()