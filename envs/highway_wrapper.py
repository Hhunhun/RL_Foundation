"""
Module: Highway Environment Wrappers (高速巡航场景封装器)
Description:
    本模块构建了强化学习智能体与底层物理仿真环境 (highway-env) 之间的标准交互桥梁。
    通过引入状态表征转换与定制化的奖励整形 (Reward Shaping) 机制，为连续控制任务提供符合
    车辆运动学约束与安全底线的马尔可夫决策过程 (MDP) 建模。

Key Features:
    - State Vectorization: 将二维车辆运动学矩阵展平为一维张量，适配多层感知机 (MLP) 的输入流。
    - Kinematic Regularization: 针对原生 SAC 基准注入动作变化率 (Jerk) 与横向控制量的二次方惩罚，强制保证策略平顺性。
    - Tolerant Reward Landscape: 针对极易因奖励剧烈波动而导致梯度爆炸的生成式扩散策略 (Diff-SAC)，
      专门设计了基于“生存与基础寻迹”的宽容度评估方案，将平顺性学习压力前置转移至离线行为克隆阶段。
    - Evaluation Probe: 严格解耦训练态与评估态，测试期间屏蔽一切辅助约束惩罚，提供绝对公平的物理量化探针。
"""

import gymnasium as gym
import highway_env
import numpy as np


class HighwayFlattenWrapper(gym.ObservationWrapper):
    """
    状态空间向量化算子 (State Space Vectorization Operator)。
    底层环境返回的原始观测通常为二维特征矩阵 (如 [车辆数, 运动学特征数])。
    该包装器将其展平为一维连续向量，以严格匹配全连接深度神经网络 (Actor/Critic) 的张量维数契约。
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
# 🎯 [保留] v6.0 工业级 AV Control 包装器 (专供 SAC 基线使用)
# ----------------------------------------------------
class HighwayAVControlWrapper(gym.Wrapper):
    """
    基于运动学约束的奖励整形层 (Kinematics-Constrained Reward Shaping Layer)。
    作为 SAC 算法的基础训练套件，向原生环境奖励中叠加一阶导数惩罚 (Jerk Penalty) 
    与反向动力学惩罚，旨在引导智能体避开“高频抖动”与“消极倒车”的病态局部最优解。
    """
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

        # 提取上一时间步与当前时间步的横向控制量，评估动作抖动烈度
        steering_jerk = abs(action[1] - self.last_action[1])
        steering_mag = abs(action[1])
        ego_speed_vx = self.env.unwrapped.vehicle.speed

        penalty = 0.0
        if not terminated:
            # L2 范数惩罚项 (L2 Norm Penalty): 利用二次函数特性，轻微波动惩罚小，极端抖动惩罚呈指数放大
            jerk_penalty = self.jerk_weight * (steering_jerk ** 2)
            mag_penalty = self.steering_weight * (steering_mag ** 2)

            reverse_penalty = 0.0
            if ego_speed_vx < 0:
                # 绝对逆行惩罚：对违背高速公路物理单向性的行为施加重罚
                reverse_penalty = self.reverse_penalty_coeff * (ego_speed_vx ** 2)
            elif ego_speed_vx < 20.0:
                # 消极怠工惩罚：防止模型收敛于“原地停车以绝对避免碰撞”的琐碎解 (Trivial Solution)
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
    """
    生成式策略宽容约束套件 (Tolerant Constraint Wrapper for Generative Policies)。
    物理意义：Diffusion 去噪过程对 Q 值的非平稳地貌 (Non-stationary Landscape) 极其敏感。
    为防止训练前期复杂的物理组合惩罚导致价值评估网络 (Critic) 崩溃，本模块大幅简化了奖励函数，
    仅提供基于安全性 (不碰撞) 与基础驱动力 (车速引导) 的稠密平滑奖励。
    """
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
                # 宽容型线性速度激励 (Tolerant Linear Speed Incentive)
                base_reward = 1.0
                # 构建轻量级梯度引导面，避免形成令生成式模型发散的陡峭“悬崖”惩罚区
                if ego_speed_vx < 20.0:
                    speed_reward = -0.1
                else:
                    speed_reward = min((ego_speed_vx - 20.0) / 5.0, 1.0)

                reward = base_reward + speed_reward
                reward = max(min(reward, 3.0), -10.0)

        info["ego_speed_vx"] = ego_speed_vx
        return next_obs, reward, terminated, truncated, info


def create_highway_env(env_name="highway-v0", is_eval=False, algo="sac"):
    """
    高速巡航场景构造工厂 (Highway Environment Factory)。
    
    Args:
        env_name: 底层 Gymnasium 注册环境名
        is_eval: 模式控制开关。True 时关闭一切训练期辅助惩罚，激活客观物理评测探针。
        algo: 指定外层挂载算法 ('sac' 或 'diff')，据此动态路由对应的运动学约束模块。
    Returns:
        env: 配置完毕且符合强化学习标准 API 契约的封装环境
    """
    env = gym.make(env_name, render_mode="rgb_array")

    if not is_eval:
        # 训练态配置 (Training Configurations)
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
        # 纯净评估态配置 (Pure Evaluation Configurations)
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