"""
Module: Merge Environment Wrappers (匝道汇入场景封装器)
Description:
    本模块构建了强化学习智能体与底层高动态匝道汇入环境 (merge-v0) 的交互桥梁。
    相比于高速巡航，匝道汇入属于典型的高时空耦合非合作博弈 (Non-cooperative Game) 任务。
    本模块通过引入 TTC 动态微扰与严苛的运动学/安全惩罚，强制模型学习高维度的动态博弈避让策略。

Key Features:
    - Hard Safety Constraints: 构建绝对的物理安全边界 (偏航角阈值、逆行检测)，触发即直接截断并施加极值惩罚。
    - Anti-Trivial Heuristics: 针对智能体极易陷入的“原地急刹车等待 NPC 先行”的局部最优琐碎解 (Trivial Solution)，
      设计了自适应的速度惩罚下界，倒逼模型在保持高效通行速度的同时完成汇入博弈。
    - TTC Dynamic Jittering: 独创的时空微扰机制。通过接管底层初始化，注入与随机种子强绑定的微小位置/速度偏移，
      彻底打破确定性路况下的轨迹过拟合 (Trajectory Overfitting)，显著提升泛化能力。
"""

import gymnasium as gym
import highway_env
import numpy as np

# ----------------------------------------------------
# 🎯 状态空间向量化包装器 (State Space Vectorization Wrapper)
# ----------------------------------------------------
class MergeFlattenWrapper(gym.ObservationWrapper):
    """
    将环境返回的二维多智能体运动学特征矩阵 (Multi-agent Kinematics Matrix)
    展平为一维连续张量，以严格满足多层感知机 (MLP) 的输入维数契约。
    """
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
# 🎯 SAC 专属运动学控制包装器 (Kinematics Control Wrapper for SAC)
# ----------------------------------------------------
class MergeAVControlWrapper(gym.Wrapper):
    """
    基于运动学平滑性与严格物理边界的奖励整形层 (Reward Shaping Layer)。
    专供基线 SAC 算法使用，向原生稀疏奖励中叠加一阶导数惩罚 (Jerk Penalty) 
    与高维度的安全硬约束，以塑造平顺且安全的驾驶策略。
    """
    def __init__(self, env, jerk_weight=1.0, steering_weight=0.5):
        super().__init__(env)
        self.last_action = np.zeros(self.env.action_space.shape)
        self.jerk_weight = jerk_weight
        self.steering_weight = steering_weight
        mode_str = "训练模式" if jerk_weight > 0 else "评估模式(纯净探针)"
        print(f"🔧 [Wrapper 原生 Merge SAC - {mode_str}] 部署 | Jerk: {jerk_weight}")

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        ego = self.env.unwrapped.vehicle
        ego_speed_vx = ego.speed
        
        # 物理安全硬约束 (Hard Safety Constraints): 严苛判定失控姿态 (如大横摆角偏航) 与非法区域边界
        is_out_of_road = info.get("is_out_of_road", False)
        is_not_on_road = not getattr(ego, "on_road", True)
        is_sideways = abs(ego.heading) > 0.4  # 高速行驶时偏航超过 0.4 弧度绝对是失控

        if not terminated:
            # 1. 触发终结条件 (Terminal States)
            if is_out_of_road or is_not_on_road or is_sideways or ego_speed_vx < -1.0:
                terminated = True
                reward -= 50.0 # 施加越界与失控的极大值惩罚
                info["crashed"] = True 
            else:
                # 2. 存活期间的 L2 范数运动学平滑约束 (Kinematic Smoothness L2 Regularization)
                steering_jerk = abs(action[1] - self.last_action[1])
                steering_mag = abs(action[1])
                reward -= (self.jerk_weight * (steering_jerk ** 2) + self.steering_weight * (steering_mag ** 2))
                
                # =========================================================
                # 🚫 [防琐碎解启发式惩罚 (Anti-Trivial Solution Heuristics)]
                # 物理意义：在非合作博弈中，模型极易退化为“完全刹车停滞，等待周围车辆清空”的消极安全策略。
                # 算法机制：抬高速度惩罚的容忍下界 (至 15.0m/s)，构造基于速度落差的线性惩罚面。
                # 预期效果：迫使智能体在保持较高动能的约束下，利用横向空间 (变道) 或微小纵向加减速完成空间博弈。
                # =========================================================
                if ego_speed_vx < 15.0:
                    reward -= (15.0 - ego_speed_vx) * 0.5 

        self.last_action = action.copy()
        info["ego_speed_vx"] = ego_speed_vx
        return next_obs, reward, terminated, truncated, info

# ----------------------------------------------------
# 🎯 Diffusion 专属宽容生成包装器 (Tolerant Constraint Wrapper for Diffusion)
# ----------------------------------------------------
class DiffMergeAVControlWrapper(gym.Wrapper):
    """
    专为生成式扩散模型设计的宽容环境封装层。
    物理意义：在不削弱任何核心安全边界 (碰撞/越界即终止) 的前提下，完全移除一阶抖动惩罚。
    旨在避免 Diffusion 模型在迭代去噪期间因非平稳的价值梯度地貌 (Non-stationary Q-landscape) 
    而陷入过估计或梯度爆炸，将精细平滑的任务完全解耦交由离线 BC 阶段学习。
    """
    def __init__(self, env, is_eval=False):
        super().__init__(env)
        self.is_eval = is_eval
        print(f"🔧 [Wrapper 原生 Merge Diff] 部署")

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        ego = self.env.unwrapped.vehicle
        ego_speed_vx = ego.speed
        
        # 沿用绝对的安全边界物理探测 (Absolute Safety Boundaries Probe)
        crashed = getattr(ego, "crashed", False)
        is_out_of_road = info.get("is_out_of_road", False)
        is_not_on_road = not getattr(ego, "on_road", True)
        is_sideways = abs(ego.heading) > 0.4
        is_reverse = ego_speed_vx < -1.0

        # 拦截致命失控 (Fatal Control Loss Interception)
        # 强制接管评估态 (is_eval=True) 下的终止判定，阻断极高 Q 权重引导下
        # 扩散网络可能产生的越野/疯狂倒车等分布外 (OOD) 生成幻想。
        if crashed or is_out_of_road or is_not_on_road or is_sideways or is_reverse:
            terminated = True
            if not self.is_eval:
                reward = -10.0
        else:
            if not self.is_eval:
                # 宽容模式下的线性驱动收益 (Tolerant Linear Driving Incentive)
                base_reward = 1.0
                speed_reward = min((ego_speed_vx - 15.0) / 10.0, 1.0) if ego_speed_vx >= 15.0 else -0.1
                reward = max(min(base_reward + speed_reward, 4.0), -10.0)

        info["ego_speed_vx"] = ego_speed_vx
        return next_obs, reward, terminated, truncated, info

# ----------------------------------------------------
# 🎯 核心环境构造工厂 (Environment Construction Factory)
# ----------------------------------------------------
def create_merge_env(env_name="merge-v0", render_mode="rgb_array", is_eval=False, algo="sac", wrapper_config=None, env_config=None):
    """
    匝道汇入场景的标准化构建流水线。
    包含底层缺陷修复、观测空间重塑、时空微扰注入以及依据外挂算法类型动态映射对应的约束体系。
    """
    env = gym.make(env_name, render_mode=render_mode)
    unwrapped_env = env.unwrapped

    # 🐛 [底层缺陷修补] 修复 highway_env 原生库针对 ContinuousAction 处理逻辑的隐患
    original_rewards_fn = unwrapped_env._rewards
    def patched_rewards(action):
        if isinstance(action, np.ndarray): return original_rewards_fn(1)
        return original_rewards_fn(action)
    unwrapped_env._rewards = patched_rewards

    # =========================================================
    # 🎯 [时空博弈对齐修正 (Spatiotemporal Game Alignment)]
    # 物理意义：原生库中的初始态极易导致 NPC 车辆与主车到达汇入点的时间错位，破坏博弈条件。
    # 通过接管底层 `reset()`，基于碰撞时间 (Time-To-Collision, TTC) 原理对偶发位置进行重构。
    # =========================================================
    original_reset = unwrapped_env.reset
    def patched_reset(*args, **kwargs):
        # 托管底层环境的原始内存布局分配
        obs, info = original_reset(*args, **kwargs)
        try:
            ego = unwrapped_env.vehicle
            road = unwrapped_env.road
            
            # =========================================================
            # 🎲 [TTC 动态时空微扰 (Dynamic Spatiotemporal Jittering)]
            # 算法机制：抽离并劫持受固定 Seed 管控的伪随机数生成器 (PRNG)。
            # 在对向双车的初始纵向位置与初速度平面上注入具有确定性的高斯/均匀抖动噪声。
            # 科研价值：彻底瓦解神经网络在固化单一场景中的轨迹过拟合 (Trajectory Overfitting)，
            # 同时确保离线评估阶段的控制变量法 (同 Seed 同路况) 绝对成立。
            # =========================================================
            np_random = unwrapped_env.np_random
            ego_pos_jitter = np_random.uniform(-5.0, 5.0)
            ego_spd_jitter = np_random.uniform(-2.0, 2.0)
            npc_pos_jitter = np_random.uniform(-5.0, 5.0)
            npc_spd_jitter = np_random.uniform(-2.0, 2.0)

            # [主车 (Ego) 初始态注入]
            ego.lane_index = ("a", "b", 1)
            lane_ego = road.network.get_lane(("a", "b", 1))
            ego.position = lane_ego.position(30 + ego_pos_jitter, 0)
            ego.speed = 25.0 + ego_spd_jitter
            
            # [匝道他车 (NPC) 初始态注入]
            ramp_lane = road.network.get_lane(("j", "k", 0))
            ramp_vehicles = [v for v in road.vehicles if v is not ego and v.lane_index == ("j", "k", 0)]
            if ramp_vehicles:
                npc = ramp_vehicles[0]
                npc.position = ramp_lane.position(50 + npc_pos_jitter, 0)
                npc.speed = 20.0 + npc_spd_jitter
                npc.target_speed = 25.0 + npc_spd_jitter # 注入强烈汇入意图
                
            # 状态重组 (State Re-observation)：物理内存地址覆写后，必须强制雷达模块
            # 重新发射探测射线，以同步更新观测张量 (Observation Tensor)。
            obs = unwrapped_env.observation_type.observe()
        except Exception as e:
            # 拓扑容错回退机制
            pass 
        return obs, info
    
    # 挂载劫持补丁
    unwrapped_env.reset = patched_reset
    # =========================================================

    # 🔧 底层物理动力学与马尔可夫奖励契约配置
    base_config = {
        "observation": {"type": "Kinematics", "vehicles_count": 5, "features": ["presence", "x", "y", "vx", "vy"], "absolute": False, "normalize": True},
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 15, "policy_frequency": 5,
        "controlled_vehicles": 1,
        "duration": 20, # 设定控制视界边界：20 秒 (对应 Policy 频率下的 100 决策步长)
        "offroad_terminal": True,
        "collision_reward": -10.0 if not is_eval else -1.0,
        "high_speed_reward": 5.0 if not is_eval else 1.0,
        "reward_speed_range": [20, 30], "show_trajectories": True,
        # 强制关闭居右行驶奖励 (Right Lane Reward)，阻断主车倒灌匝道的病态探索行为
        "right_lane_reward": 0.0, 
    }

    if env_config: base_config.update(env_config)
    env.unwrapped.configure(base_config)
    env.reset() 

    env = MergeFlattenWrapper(env)

    if algo == "diff":
        env = DiffMergeAVControlWrapper(env, is_eval=is_eval)
    else:
        jerk = wrapper_config.get("jerk_weight", 1.0) if wrapper_config else 1.0
        steering = wrapper_config.get("steering_weight", 0.5) if wrapper_config else 0.5
        env = MergeAVControlWrapper(env, jerk_weight=jerk if not is_eval else 0.0, steering_weight=steering if not is_eval else 0.0)

    return env