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
# 🎯 [v6.0 升级] 工业级 AV Control 包装器
# 引入横向绝对约束与更激进的纵向速度重塑
# ----------------------------------------------------
class HighwayAVControlWrapper(gym.Wrapper):
    """
    [v6.0] AV Control 包装器。
    1. LQR 二次型 Jerk 惩罚：防止抖动摆头。
    2. [新增] 绝对转向惩罚：防止持续维持固定转角导致的连续变道侧翻。
    3. Velocity Constraint：禁止纵向倒车，并严厉惩罚 20m/s 以下的龟速行为。
    """

    def __init__(self, env, jerk_weight=1.0, steering_weight=0.5, reverse_penalty_coeff=5.0):
        super().__init__(env)
        self.last_action = np.zeros(self.env.action_space.shape)

        self.jerk_weight = jerk_weight
        # [v6.0] steering_weight: 核心惩罚绝对转向幅度
        self.steering_weight = steering_weight
        self.reverse_penalty_coeff = reverse_penalty_coeff

        # 动态日志：打印当前是训练态(有权重)还是评估态(0权重)
        mode_str = "训练模式" if jerk_weight > 0 else "评估模式(纯净探针)"
        print(
            f"🔧 [Wrapper v6.0 - {mode_str}] AV Control 专家套件已部署 | Jerk权重: {jerk_weight} | 转向幅度权重: {steering_weight} | 倒车惩罚系数: {reverse_penalty_coeff}")

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. 提取动作特征
        steering_jerk = abs(action[1] - self.last_action[1])
        steering_mag = abs(action[1])  # [v6.0] 获取绝对转向幅度

        ego_speed_vx = self.env.unwrapped.vehicle.speed  # 自车物理速度

        penalty = 0.0
        if not terminated:
            # a) 横向稳定性约束 (Lateral Stability Constraints)
            jerk_penalty = self.jerk_weight * (steering_jerk ** 2)
            # [v6.0] 附加惩罚：方向盘只要不回正，就会按照二次型扣分，逼迫变道后立即回正
            mag_penalty = self.steering_weight * (steering_mag ** 2)

            # b) [v6.0 升级] 纵向速度重塑 (Longitudinal Speed Shaping)
            reverse_penalty = 0.0
            if ego_speed_vx < 0:
                reverse_penalty = self.reverse_penalty_coeff * (ego_speed_vx ** 2)
            elif ego_speed_vx < 20.0:
                # [v6.0 修正] 严厉惩罚龟速，打破 21m/s 的舒适区，阈值从 10 抬升至 20
                reverse_penalty = self.reverse_penalty_coeff * 0.1 * (20.0 - ego_speed_vx)  # 联动系数，当惩罚系数为0时也为0

            penalty = jerk_penalty + mag_penalty + reverse_penalty
            reward -= penalty

        self.last_action = action.copy()

        # [v5.0 探针] 将实时速度注入 info 字典，方便 test 脚本读取显示
        info["ego_speed_vx"] = ego_speed_vx

        return next_obs, reward, terminated, truncated, info


def create_highway_env(env_name="highway-v0", is_eval=False):
    """
    工业级环境创建工厂：不仅创建环境，还锁死关键配置。
    [新特性] 支持 is_eval 物理隔离，确保评估标尺的绝对纯净。
    """
    env = gym.make(env_name, render_mode="rgb_array")

    if not is_eval:
        # [核心防御逻辑] 强制注入 v6.0 严谨AV法规物理配置 (训练考场)
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
            # 🎯 [专家手术点 1] v6.0 法规级KPI配置
            # ----------------------------------------------------
            "offroad_terminal": True,  # [致命红线] 出轨即死刑
            "collision_reward": -10.0,  # [致命红线] 撞车重罚
            "high_speed_reward": 5.0,  # [提速渴望] 高额提速奖金池

            # [v6.0 修正] 将期望速度下限提高到 24，逼迫其实施超车行为
            "reward_speed_range": [24, 30],

            # [v6.0 修正] 加重原生变道惩罚，结合 Wrapper 中的转向幅度惩罚，彻底杜绝无意义变道
            "lane_change_reward": -0.05,

            # ----------------------------------------------------
            # 🎯 [要求B之工业替代方案] 开启轨迹显示
            # ----------------------------------------------------
            "show_trajectories": True,
        })
    else:
        # ❄️ [核心冻结逻辑] 绝对纯净的评估考场：剥离所有人工塑形
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
            "offroad_terminal": True,  # 物理底线保留
            "collision_reward": -1.0,  # 纯净客观的碰撞标记
            "high_speed_reward": 1.0,  # 纯净客观的提速标记
            "reward_speed_range": [20, 30],  # 恢复基准速度域
            "lane_change_reward": 0.0,  # 取消人为干预变道惩罚
            "show_trajectories": True,
        })

    env.reset()

    # 1. 套上降维装甲
    env = HighwayFlattenWrapper(env)

    # ----------------------------------------------------
    # 2. [专家手术点 2] 套上 v6.0 AV Control 专家套件
    # ----------------------------------------------------
    if not is_eval:
        # 🏋️ 训练模式挂载所有严酷惩罚
        env = HighwayAVControlWrapper(env, jerk_weight=1.0, steering_weight=0.5, reverse_penalty_coeff=10.0)
    else:
        # ❄️ 评估模式彻底关停惩罚权重，仅利用 Wrapper 充当速度探针
        env = HighwayAVControlWrapper(env, jerk_weight=0.0, steering_weight=0.0, reverse_penalty_coeff=0.0)

    return env


if __name__ == "__main__":
    # === 探针测试 ===
    print("开始 HighwayEnv v6.0 AV 法规版探针测试 (训练模式)...")
    env = create_highway_env(is_eval=False)
    obs, info = env.reset()

    # 测试一步倒车动作：检查 info 探针和奖励
    reverse_action = np.array([-1.0, 0.0])  # 踩死刹车，直行倒车
    for _ in range(3):
        _, reward, _, _, info = env.step(reverse_action)
        print(f"🚗 训练场倒车 vx: {info['ego_speed_vx']:.2f} m/s | 💰 塑形后奖励: {reward:.4f} (应扣分)")
    env.close()

    print("\n开始 HighwayEnv 冻结考场探针测试 (评估模式)...")
    env_eval = create_highway_env(is_eval=True)
    obs, info = env_eval.reset()
    for _ in range(3):
        _, reward, _, _, info = env_eval.step(reverse_action)
        print(f"❄️ 评估场倒车 vx: {info['ego_speed_vx']:.2f} m/s | 💰 纯净物理奖励: {reward:.4f} (无额外扣分)")
    env_eval.close()