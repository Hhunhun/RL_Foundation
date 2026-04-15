import gymnasium as gym
import highway_env
import numpy as np


class MergeFlattenWrapper(gym.ObservationWrapper):
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
# 🎯 SAC 专属 AV Control 包装器 (侧重平滑性约束)
# ----------------------------------------------------
class MergeAVControlWrapper(gym.Wrapper):
    def __init__(self, env, jerk_weight=1.0, steering_weight=0.5):
        super().__init__(env)
        self.last_action = np.zeros(self.env.action_space.shape)
        self.jerk_weight = jerk_weight
        self.steering_weight = steering_weight

        mode_str = "训练模式" if jerk_weight > 0 else "评估模式(纯净探针)"
        print(f"🔧 [Wrapper Merge SAC - {mode_str}] AV Control 专家套件已部署 | Jerk: {jerk_weight}")

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        steering_jerk = abs(action[1] - self.last_action[1])
        steering_mag = abs(action[1])
        ego_speed_vx = self.env.unwrapped.vehicle.speed

        penalty = 0.0
        if not terminated:
            jerk_penalty = self.jerk_weight * (steering_jerk ** 2)
            mag_penalty = self.steering_weight * (steering_mag ** 2)
            penalty = jerk_penalty + mag_penalty
            reward -= penalty

        self.last_action = action.copy()
        info["ego_speed_vx"] = ego_speed_vx

        return next_obs, reward, terminated, truncated, info


# ----------------------------------------------------
# 🎯 Diffusion 专属平滑包装器 (Curriculum Stage)
# ----------------------------------------------------
class DiffMergeAVControlWrapper(gym.Wrapper):
    def __init__(self, env, is_eval=False):
        super().__init__(env)
        self.is_eval = is_eval
        mode_str = "评估模式(纯净探针)" if is_eval else "训练模式(宽松存活期)"
        print(f"🔧 [Wrapper Merge Diff - {mode_str}] 宽容级平滑套件已部署")

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        ego_speed_vx = self.env.unwrapped.vehicle.speed
        crashed = getattr(self.env.unwrapped.vehicle, "crashed", False)

        if not self.is_eval:
            if crashed:
                reward = -10.0
                terminated = True
            else:
                # 给予基础存活奖励和适当的速度引导
                base_reward = 1.0
                speed_reward = min((ego_speed_vx - 15.0) / 10.0, 1.0) if ego_speed_vx >= 15.0 else -0.1
                reward = max(min(base_reward + speed_reward, 3.0), -10.0)

        info["ego_speed_vx"] = ego_speed_vx
        return next_obs, reward, terminated, truncated, info


def create_merge_env(env_name="merge-v0", is_eval=False, algo="sac", wrapper_config=None):
    env = gym.make(env_name, render_mode="rgb_array")

    # -------------------------------------------------------------------
    # 🐛 [BUG FIX] 修复 highway_env 库在 MergeEnv 下连续动作空间的底层源码缺陷
    # -------------------------------------------------------------------
    unwrapped_env = env.unwrapped
    original_rewards_fn = unwrapped_env._rewards
    
    def patched_rewards(action):
        # 如果 action 是连续动作的 numpy 数组，则向底层传入一个虚拟的离散动作 `1` (代表保持直行)
        # 从而安全地绕过底层代码里 `action in [0, 2]` 的强制数组真值检测。
        # 我们的连续动作平滑惩罚已经交由外层的 Wrapper 处理了，所以这里覆盖掉毫无影响。
        if isinstance(action, np.ndarray):
            return original_rewards_fn(1)
        return original_rewards_fn(action)
        
    unwrapped_env._rewards = patched_rewards
    # -------------------------------------------------------------------

    env.unwrapped.configure({
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True
        },
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "collision_reward": -10.0 if not is_eval else -1.0,
        "right_lane_reward": 0.1,  
        "high_speed_reward": 5.0 if not is_eval else 1.0,
        "reward_speed_range": [20, 30],
        "lane_change_reward": 0.0, # 避免与连续动作空间冲突，直接设为0
        "show_trajectories": True,
    })

    env.reset()
    env = MergeFlattenWrapper(env)

    if algo == "diff":
        env = DiffMergeAVControlWrapper(env, is_eval=is_eval)
    else:
        # 如果外部传入了 wrapper_config，则使用它；否则使用默认值
        jerk = wrapper_config.get("jerk_weight", 1.0) if wrapper_config else 1.0
        steering = wrapper_config.get("steering_weight", 0.5) if wrapper_config else 0.5

        if not is_eval:
            env = MergeAVControlWrapper(env, jerk_weight=jerk, steering_weight=steering)
        else:
            # 评估模式下，强制关闭所有平滑惩罚，确保公平
            env = MergeAVControlWrapper(env, jerk_weight=0.0, steering_weight=0.0)

    return env
