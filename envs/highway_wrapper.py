import gymnasium as gym
import highway_env  # 必须导入以触发环境注册机制
import numpy as np


class HighwayFlattenWrapper(gym.ObservationWrapper):
    """
    针对 HighwayEnv 矩阵状态的降维包装器。
    """

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
# 🎯 [专家手术点 1] 新增：动作约束包装器
# 用于根治连续动作空间下的剧烈摆头(wobbling)和压线
# ----------------------------------------------------
class HighwayActionWobbleWrapper(gym.Wrapper):
    """
    动作约束包装器。
    拦截 step() 函数，根据动作的大小和变化率(Jerk)在环境奖励上扣除惩罚分。
    """

    def __init__(self, env, steering_penalty_coeff=0.5):
        super().__init__(env)
        # 记录上一个动作，用于计算动作变化率 (jerk)
        self.last_action = np.zeros(self.env.action_space.shape)
        # 惩罚系数：告诉小车打方向盘的力度和抖动幅度扣分有多严重
        self.steering_penalty_coeff = steering_penalty_coeff
        print(f"🔧 [Wrapper] 动作约束机制已点火，惩罚系数: {steering_penalty_coeff}")

    def step(self, action):
        """
        拦截动作，计算惩罚，并修改奖励。
        """
        # 1. 执行原环境动作
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # 2. 专家级奖励塑形 (Reward Shaping) - 动作惩罚
        # action[1] 是连续转向动作 [-1, 1]。0.0 表示直行。
        # 我们要惩罚两个方面：

        # a) 转向幅度惩罚：打死方向盘（-1或1）绝对不鼓励，轻微打盘（0.1）惩罚微小。
        steering_magnitude = abs(action[1])

        # b) 转向抖动惩罚 (Jerk Penalty)：
        # 如果从直行(0.0)瞬间变成打死(1.0)，这是一个剧烈的抖动。
        # 如果从轻微左转(0.1)变成轻微右转(-0.1)，这是一个动作换向的抖动。
        # 这些剧烈的动作抖动必须重罚。
        steering_change = abs(action[1] - self.last_action[1])

        # 组合惩罚项 (惩罚必须是负数)
        # 如果这一局 terminated（撞车）了，就不扣这个动作分了，因为碰撞惩罚足够重。
        wobble_penalty = 0.0
        if not terminated:
            # 我们重点惩罚转向幅度，因为 user 反馈主要是左右摆头压线
            wobble_penalty = self.steering_penalty_coeff * (steering_magnitude + steering_change)
            reward -= wobble_penalty  # 将惩罚减去

        # 3. 更新上一个动作状态
        self.last_action = action.copy()

        return next_obs, reward, terminated, truncated, info


def create_highway_env(env_name="highway-v0"):
    """
    工业级环境创建工厂。
    """
    env = gym.make(env_name, render_mode="rgb_array")

    # [核心防御逻辑] 强制注入严谨的物理配置与KPI
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
        # [专家手术点 2] 极度Aggressive的环境KPI配置
        # 治愈消极怠工
        # ----------------------------------------------------
        "collision_reward": -10.0,  # [重罚] 撞车重罚：这依然是绝对红线
        "high_speed_reward": 5.0,  # [大修] 疯狂加倍！重赏之下必有勇夫。彻底覆盖存活低保收益。
        "reward_speed_range": [20, 30],  # 定义 KPI：速度低于 20m/s (72km/h) 别想拿大钱。
        "lane_change_reward": -0.2,  # [惩罚] 严禁消极变道跳Bug：鼓励变道超车，但禁止没意义的横跳。
        "right_lane_reward": 0.0,  # [移除] 移除隐含奖励以消除变道回来的Bug行为，依靠高速奖励和动作惩罚来驱动决策。
    })

    env.reset()

    # 1. 套上降维装甲
    env = HighwayFlattenWrapper(env)

    # ----------------------------------------------------
    # 2. [专家手术点 3] 物理套壳：套上动作约束套件
    # 用于根治 wobbling 和 lane-hopping
    # ----------------------------------------------------
    env = HighwayActionWobbleWrapper(env, steering_penalty_coeff=0.5)

    return env


if __name__ == "__main__":
    # === 探针测试：合理性检验 (Sanity Check) ===
    print("开始新版 HighwayEnv 数据探针测试...")
    env = create_highway_env()

    obs, info = env.reset()
    print(f"\n✅ 成功获取展平后的观测空间，Shape: {obs.shape}")
    print(f"✅ 动作空间，Shape: {env.action_space.shape}")

    # 测试一步随机交互，检查奖励是否合理
    random_action = np.array([1.0, 1.0])  # 疯狂踩油门打方向盘的发癫动作
    next_obs, reward, terminated, truncated, _ = env.step(random_action)
    print(f"\n🚗 发癫随机执行动作: {random_action}")
    print(f"💰 获得塑形后奖励: {reward:.4f} (如果环境物理引擎计算小车没加速成功，这里应该是一个由于动作过大导致的负分)")

    env.close()