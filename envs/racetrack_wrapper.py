import gymnasium as gym
import highway_env
import numpy as np

# ----------------------------------------------------
# 🎯 基础观测展平包装器
# ----------------------------------------------------
class RacetrackFlattenWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = self.env.observation_space
        self.flat_dim = np.prod(obs_space.shape)
        self.observation_space = gym.spaces.Box(
            low=np.min(obs_space.low), high=np.max(obs_space.high),
            shape=(self.flat_dim,), dtype=np.float32
        )
    def observation(self, obs):
        return np.array(obs, dtype=np.float32).flatten()

# ----------------------------------------------------
# 🎯 SAC 专属 AV Control 包装器
# ----------------------------------------------------
class RacetrackAVControlWrapper(gym.Wrapper):
    def __init__(self, env, jerk_weight=1.0, steering_weight=0.5):
        super().__init__(env)
        self.last_action = np.zeros(self.env.action_space.shape)
        self.jerk_weight = jerk_weight
        self.steering_weight = steering_weight
        mode_str = "训练模式" if jerk_weight > 0 else "评估模式(纯净探针)"
        print(f"🏎️ [Wrapper 原生 Racetrack SAC - {mode_str}] 部署 | Jerk: {jerk_weight}")

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        ego = self.env.unwrapped.vehicle
        ego_speed_vx = ego.speed
        
        # 🚫 赛道专属物理裁判：赛道允许大角度航向(过弯)，因此主要惩罚冲出赛道和倒车
        is_out_of_road = info.get("is_out_of_road", False)
        is_not_on_road = not getattr(ego, "on_road", True)
        is_reverse = ego_speed_vx < -1.0

        if not terminated:
            # 1. 触发死刑条件
            if is_out_of_road or is_not_on_road or is_reverse:
                terminated = True
                reward -= 50.0 # 严惩出轨与倒车
                info["crashed"] = True # 统一标记为失败
            else:
                # 2. 存活期间的平滑约束 (防画龙)
                steering_jerk = abs(action[1] - self.last_action[1])
                steering_mag = abs(action[1])
                # 赛道必须转弯，因此 steering_mag 的惩罚权重应当比直道小，重点惩罚 jerk
                reward -= (self.jerk_weight * (steering_jerk ** 2) + self.steering_weight * (steering_mag ** 2))
                
                # 赛道最低限速保底：防止学会龟速蠕动过弯
                if ego_speed_vx < 10.0:
                    reward -= (10.0 - ego_speed_vx) * 0.5 

        self.last_action = action.copy()
        info["ego_speed_vx"] = ego_speed_vx
        return next_obs, reward, terminated, truncated, info

# ----------------------------------------------------
# 🎯 Diffusion 专属宽容包装器
# ----------------------------------------------------
class DiffRacetrackAVControlWrapper(gym.Wrapper):
    def __init__(self, env, is_eval=False):
        super().__init__(env)
        self.is_eval = is_eval
        print(f"🏎️ [Wrapper 原生 Racetrack Diff] 部署")

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        ego = self.env.unwrapped.vehicle
        ego_speed_vx = ego.speed
        
        crashed = getattr(ego, "crashed", False)
        is_out_of_road = info.get("is_out_of_road", False)
        is_not_on_road = not getattr(ego, "on_road", True)
        is_reverse = ego_speed_vx < -1.0

        # 物理死刑
        if crashed or is_out_of_road or is_not_on_road or is_reverse:
            terminated = True
            if not self.is_eval:
                reward = -10.0
        else:
            if not self.is_eval:
                base_reward = 1.0
                # 赛道刷圈均速奖励
                speed_reward = min((ego_speed_vx - 10.0) / 10.0, 1.0) if ego_speed_vx >= 10.0 else -0.1
                reward = max(min(base_reward + speed_reward, 4.0), -10.0)

        info["ego_speed_vx"] = ego_speed_vx
        return next_obs, reward, terminated, truncated, info

# ----------------------------------------------------
# 🎯 核心工厂函数
# ----------------------------------------------------
def create_racetrack_env(env_name="racetrack-v0", render_mode="rgb_array", is_eval=False, algo="sac", wrapper_config=None, env_config=None):
    env = gym.make(env_name, render_mode=render_mode)
    unwrapped_env = env.unwrapped

    # 🐛 [BUG FIX 保留] 修复 highway_env 连续动作空间的底层源码缺陷
    original_rewards_fn = unwrapped_env._rewards
    def patched_rewards(action):
        if isinstance(action, np.ndarray): return original_rewards_fn(1)
        return original_rewards_fn(action)
    unwrapped_env._rewards = patched_rewards

    # 🔧 赛道专属基础配置
    base_config = {
        "observation": {"type": "Kinematics", "features": ["presence", "x", "y", "vx", "vy"], "absolute": False, "normalize": True},
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 15, "policy_frequency": 5,
        "controlled_vehicles": 1,
        "duration": 50, # 赛道可能需要更长时间跑完一圈
        "offroad_terminal": True,
        "collision_reward": -10.0 if not is_eval else -1.0,
        "high_speed_reward": 5.0 if not is_eval else 1.0,
        "reward_speed_range": [15, 30], # 赛道速度范围
        "show_trajectories": True
    }

    if env_config: base_config.update(env_config)
    env.unwrapped.configure(base_config)
    env.reset() 

    env = RacetrackFlattenWrapper(env)

    if algo == "diff":
        env = DiffRacetrackAVControlWrapper(env, is_eval=is_eval)
    else:
        jerk = wrapper_config.get("jerk_weight", 1.0) if wrapper_config else 1.0
        # 赛道环境中，转向幅度是不可避免的，因此调低默认 steering_weight
        steering = wrapper_config.get("steering_weight", 0.1) if wrapper_config else 0.1
        env = RacetrackAVControlWrapper(env, jerk_weight=jerk if not is_eval else 0.0, steering_weight=steering if not is_eval else 0.0)

    return env