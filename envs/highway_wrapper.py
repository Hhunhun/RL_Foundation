import gymnasium as gym
import highway_env
import numpy as np


class HighwayFlattenWrapper(gym.ObservationWrapper):
    """[保留原注释] 针对矩阵状态的降维包装器。"""

    def __init__(self, env):
        super().__init__(env)
        obs_space = self.env.observation_space
        assert isinstance(obs_space, gym.spaces.Box), "原始观测空间必须是 Box 类型"
        self.flat_dim = np.prod(obs_space.shape)
        self.observation_space = gym.spaces.Box(
            low=np.min(obs_space.low),
            high=np.max(obs_space.high),
            shape=(self.flat_dim,),
            dtype=np.float32
        )

    def observation(self, obs):
        return np.array(obs, dtype=np.float32).flatten()


# ----------------------------------------------------
# 🎯 [v5.0 升级] 工业级 AV Control 包装器
# 彻底根治“倒车苟活” Bug，并允许包容平滑变道。
# ----------------------------------------------------
class HighwayAVControlWrapper(gym.Wrapper):
    """
    [v5.0] AV Control 包装器。
    1. LQR 二次型 Jerk 惩罚：防止抖动摆头。
    2. Velocity Constraint：禁止纵向倒车(vx < 0)和龟速跟车。
    """

    def __init__(self, env, jerk_weight=1.0, reverse_penalty_coeff=5.0):
        super().__init__(env)
        self.last_action = np.zeros(self.env.action_space.shape)
        # jerk_weight: 核心惩罚转向变化率 (禁止抖动)
        self.jerk_weight = jerk_weight
        # reverse_penalty_coeff: 惩罚倒车的系数 (要重重地惩罚！)
        self.reverse_penalty_coeff = reverse_penalty_coeff
        print(
            f"🔧 [Wrapper v5.0] AV Control 专家套件已部署 | Jerk权重: {jerk_weight} | 倒车惩罚系数: {reverse_penalty_coeff}")

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. 提取动作：只需要惩罚方向盘的抖动 jerk
        # action[0]是油门，不惩罚。 action[1]是转向。
        steering_jerk = abs(action[1] - self.last_action[1])

        # [v5.0] 专家核心手术：速度约束 (Velocity Constraint)
        # 我们直接访问物理引擎的 unwrapped 对象，获取最精准的绝对纵向速度 (m/s)
        ego_speed_vx = self.env.unwrapped.vehicle.speed  # 自车物理速度

        penalty = 0.0
        if not terminated:
            # a) LQR Jerk 惩罚 (二次型): 0.1^2=0.01; 0.8^2=0.64
            # 允许平滑超车，严打发癫抖动
            jerk_penalty = self.jerk_weight * (steering_jerk ** 2)

            # b) [v5.0 新增] 致命禁止：纵向倒车惩罚
            # 在高速上倒着开是绝对红线。如果 vx < 0，给予毁灭性的持续惩罚。
            reverse_penalty = 0.0
            if ego_speed_vx < 0:
                # 倒得越快扣得越多，惩罚是和速度幅度的二次方成正比
                reverse_penalty = self.reverse_penalty_coeff * (ego_speed_vx ** 2)
            elif ego_speed_vx < 10.0:
                # 在高速上龟速跟车(10m/s=36km/h)也要小惩罚，逼迫它加速拿high_speed奖励
                reverse_penalty = 0.5 * (10.0 - ego_speed_vx)

            penalty = jerk_penalty + reverse_penalty
            reward -= penalty

        self.last_action = action.copy()

        # [v5.0 探针] 将实时速度注入 info 字典，方便 test 脚本读取显示
        info["ego_speed_vx"] = ego_speed_vx

        return next_obs, reward, terminated, truncated, info


def create_highway_env(env_name="highway-v0"):
    """
    工业级环境创建工厂：不仅创建环境，还锁死关键配置。
    """
    env = gym.make(env_name, render_mode="rgb_array")

    # [核心防御逻辑] 强制注入 v5.0 严谨AV法规物理配置
    env.unwrapped.configure({
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True
        },
        "action": {
            "type": "ContinuousAction"
        },
        "simulation_frequency": 15,
        "policy_frequency": 5,

        # ----------------------------------------------------
        # 🎯 [专家手术点 1] v5.0 法规级KPI配置
        # 治愈草地作弊，解决无意义连续变道
        # ----------------------------------------------------
        "offroad_terminal": True,  # [致命红线] 出轨即死刑，杜绝草地作弊！
        "collision_reward": -10.0,  # [致命红线] 撞车重罚
        "high_speed_reward": 5.0,  # [提速渴望] 高额提速奖金池
        "reward_speed_range": [20, 30],

        # [v5.0 优化] 重新引入微小的原生变道惩罚。
        # 不会阻止它为了超车而变道，但会阻止它无意义地横跨两条车道去加速。
        "lane_change_reward": -0.02,

        # ----------------------------------------------------
        # 🎯 [要求B之工业替代方案] 开启轨迹显示
        # 在渲染阶段，车辆下方会显示一条平滑的轨迹线。
        # 轨迹越平滑，说明 LQR 效果越好；轨迹剧烈弯曲，说明动作抖动严重。
        # 这是诊断 jerk 和是否倒车的最佳直观可视化手段！
        # ----------------------------------------------------
        "show_trajectories": True,
    })

    env.reset()

    # 1. 套上降维装甲
    env = HighwayFlattenWrapper(env)

    # ----------------------------------------------------
    # 2. [专家手术点 2] 套上 v5.0 AV Control 专家套件
    # 用于根治倒车苟活和 wobbling
    # ----------------------------------------------------
    env = HighwayAVControlWrapper(env, jerk_weight=1.0, reverse_penalty_coeff=10.0)

    return env


if __name__ == "__main__":
    # === 探针测试 ===
    print("开始 HighwayEnv v5.0 AV 法规版探针测试...")
    env = create_highway_env()
    obs, info = env.reset()

    # 测试一步倒车动作：检查 info 探针和奖励
    reverse_action = np.array([-1.0, 0.0])  # 踩死刹车，直行倒车
    for _ in range(5):
        _, reward, _, _, info = env.step(reverse_action)
        print(f"🚗 物理倒车 vx: {info['ego_speed_vx']:.2f} m/s | 💰 塑形后奖励: {reward:.4f} (应该吃到毁灭性惩罚)")

    env.close()