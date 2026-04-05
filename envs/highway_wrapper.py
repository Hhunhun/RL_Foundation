import gymnasium as gym
import highway_env
import numpy as np


class HighwayFlattenWrapper(gym.ObservationWrapper):
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
# 🎯 [保留] v6.0 工业级 AV Control 包装器 (专供 SAC 基线使用)
# ----------------------------------------------------
class HighwayAVControlWrapper(gym.Wrapper):
    def __init__(self, env, jerk_weight=1.0, steering_weight=0.5, reverse_penalty_coeff=5.0):
        super().__init__(env)
        self.last_action = np.zeros(self.env.action_space.shape)
        self.jerk_weight = jerk_weight
        self.steering_weight = steering_weight
        self.reverse_penalty_coeff = reverse_penalty_coeff

        mode_str = "训练模式" if jerk_weight > 0 else "评估模式(纯净探针)"
        print(
            f"🔧 [Wrapper SAC-v6.0 - {mode_str}] AV Control 专家套件已部署 | Jerk: {jerk_weight} | 转向: {steering_weight}")

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        steering_jerk = abs(action[1] - self.last_action[1])
        steering_mag = abs(action[1])
        ego_speed_vx = self.env.unwrapped.vehicle.speed

        penalty = 0.0
        if not terminated:
            jerk_penalty = self.jerk_weight * (steering_jerk ** 2)
            mag_penalty = self.steering_weight * (steering_mag ** 2)

            reverse_penalty = 0.0
            if ego_speed_vx < 0:
                reverse_penalty = self.reverse_penalty_coeff * (ego_speed_vx ** 2)
            elif ego_speed_vx < 20.0:
                reverse_penalty = self.reverse_penalty_coeff * 0.1 * (20.0 - ego_speed_vx)

            penalty = jerk_penalty + mag_penalty + reverse_penalty
            reward -= penalty

        self.last_action = action.copy()
        info["ego_speed_vx"] = ego_speed_vx

        return next_obs, reward, terminated, truncated, info


# ----------------------------------------------------
# 🎯 [修改] Diffusion 专属平滑包装器 (Curriculum Stage 1/2)
# ----------------------------------------------------
class DiffHighwayAVControlWrapper(gym.Wrapper):
    def __init__(self, env, is_eval=False):
        super().__init__(env)
        self.is_eval = is_eval
        mode_str = "评估模式(纯净探针)" if is_eval else "训练模式(课程学习:生存期)"
        print(f"🔧 [Wrapper Diff-v1.0 - {mode_str}] 宽容级平滑套件已部署")

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        ego_speed_vx = self.env.unwrapped.vehicle.speed
        crashed = getattr(self.env.unwrapped.vehicle, "crashed", False)

        if not self.is_eval:
            if crashed:
                reward = -10.0
                terminated = True
            elif ego_speed_vx < 0:
                reward = -5.0
                terminated = True
            else:
                base_reward = 1.0
                # 极其微弱的线性引导，避免悬崖惩罚
                if ego_speed_vx < 20.0:
                    speed_reward = -0.1
                else:
                    speed_reward = min((ego_speed_vx - 20.0) / 5.0, 1.0)

                reward = base_reward + speed_reward
                reward = max(min(reward, 3.0), -10.0)

        info["ego_speed_vx"] = ego_speed_vx
        return next_obs, reward, terminated, truncated, info


def create_highway_env(env_name="highway-v0", is_eval=False, algo="sac"):
    env = gym.make(env_name, render_mode="rgb_array")

    if not is_eval:
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
            "offroad_terminal": True,
            "collision_reward": -10.0,
            "high_speed_reward": 5.0,
            "reward_speed_range": [24, 30],
            "lane_change_reward": -0.05 if algo == "sac" else 0.0, # 训练初期取消变道惩罚
            "show_trajectories": True,
        })
    else:
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
            "offroad_terminal": True,
            "collision_reward": -1.0,
            "high_speed_reward": 1.0,
            "reward_speed_range": [20, 30],
            "lane_change_reward": 0.0,
            "show_trajectories": True,
        })

    env.reset()
    env = HighwayFlattenWrapper(env)

    if algo == "diff":
        env = DiffHighwayAVControlWrapper(env, is_eval=is_eval)
    else:
        if not is_eval:
            env = HighwayAVControlWrapper(env, jerk_weight=1.0, steering_weight=0.5, reverse_penalty_coeff=10.0)
        else:
            env = HighwayAVControlWrapper(env, jerk_weight=0.0, steering_weight=0.0, reverse_penalty_coeff=0.0)

    return env


if __name__ == "__main__":
    print("=== 开始 SAC 环境探针测试 ===")
    env_sac = create_highway_env(is_eval=False, algo="sac")
    obs, info = env_sac.reset()
    env_sac.close()

    print("\n=== 开始 Diffusion 环境探针测试 ===")
    env_diff = create_highway_env(is_eval=False, algo="diff")
    obs, info = env_diff.reset()
    env_diff.close()