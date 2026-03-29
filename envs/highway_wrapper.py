import gymnasium as gym
import highway_env  # <-- 核心修复 1：必须显式导入，触发底层的环境注册机制
import numpy as np


class HighwayFlattenWrapper(gym.ObservationWrapper):
    """
    针对 HighwayEnv 矩阵状态的降维包装器。
    将原始的 (V, F) 矩阵展平为 (V * F,) 的一维向量，以便 SAC 的全连接层 (MLP) 摄入。
    """

    def __init__(self, env):
        super().__init__(env)
        # 获取原始的观测空间
        obs_space = self.env.observation_space

        # 严谨的检查：确保我们包装的是 Box 空间
        assert isinstance(obs_space, gym.spaces.Box), "原始观测空间必须是 Box 类型"

        # 计算展平后的新维度 (例如 5 * 5 = 25)
        self.flat_dim = np.prod(obs_space.shape)

        # 重新定义 observation_space
        self.observation_space = gym.spaces.Box(
            low=np.min(obs_space.low),
            high=np.max(obs_space.high),
            shape=(self.flat_dim,),
            dtype=np.float32
        )
        # print(f"🔧 [Wrapper] 观测空间已降维重塑: {obs_space.shape} -> {self.observation_space.shape}")

    def observation(self, obs):
        """
        在每次 env.reset() 和 env.step() 被调用时，这个函数会自动拦截并处理原始观测值
        """
        # 展平矩阵，并确保数据类型是神经网络友好的 float32
        return np.array(obs, dtype=np.float32).flatten()


def create_highway_env(env_name="highway-v0"):  # <-- 核心修复 2：使用标准基础环境名
    """
    工业级环境创建工厂：不仅创建环境，还锁死关键配置。
    防止不同机器或版本的默认值跳跃导致训练无法复现。
    """
    env = gym.make(env_name, render_mode="rgb_array")

    # [核心防御逻辑] 强制注入严谨的物理配置
    env.unwrapped.configure({
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,  # 1 辆主车 + 4 辆最邻近车辆
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,  # 极其重要：使用相对坐标！以主车为原点，网络才具备空间平移泛化能力
            "normalize": True  # 极其重要：强制归一化到 [-1, 1] 附近，防止 Actor/Critic 梯度爆炸
        },
        "action": {
            "type": "ContinuousAction"  # 核心修复 2 补充：在这里指定连续控制，完美适配 SAC
        },
        "simulation_frequency": 15,  # 物理引擎的仿真频率 (Hz)
        "policy_frequency": 5,  # 策略网络的决策频率 (Hz) - 相当于每 0.2 秒做一次动作
    })

    # 必须调用 reset 才能让 configure 生效
    env.reset()

    # 套上我们的降维装甲
    env = HighwayFlattenWrapper(env)

    return env


if __name__ == "__main__":
    # === 探针测试：合理性检验 (Sanity Check) ===
    print("开始 HighwayEnv 环境探针测试...")
    env = create_highway_env()

    obs, info = env.reset()
    print(f"\n✅ 成功获取展平后的观测空间，Shape: {obs.shape}")
    print(f"✅ 动作空间，Shape: {env.action_space.shape}")
    print(f"🔍 初始观测数据快照 (前 10 个维度):\n{obs[:10]}")

    # 测试一步随机交互
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = env.step(action)
    print(f"\n🚗 随机执行动作: {action}")
    print(f"💰 获得单步奖励: {reward:.4f} | 结束状态: {terminated}")

    env.close()