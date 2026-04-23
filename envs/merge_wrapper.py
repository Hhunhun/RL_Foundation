import gymnasium as gym
import highway_env
import numpy as np

# ----------------------------------------------------
# 🎯 基础观测展平包装器
# ----------------------------------------------------
class MergeFlattenWrapper(gym.ObservationWrapper):
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
class MergeAVControlWrapper(gym.Wrapper):
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
        
        # 🚫 铁腕物理裁判：防打滑、防草地、防倒车
        is_out_of_road = info.get("is_out_of_road", False)
        is_not_on_road = not getattr(ego, "on_road", True)
        is_sideways = abs(ego.heading) > 0.4  # 高速行驶时偏航超过 0.4 弧度绝对是失控

        if not terminated:
            # 1. 触发死刑条件
            if is_out_of_road or is_not_on_road or is_sideways or ego_speed_vx < -1.0:
                terminated = True
                reward -= 50.0 # 严惩死亡
                info["crashed"] = True # 统一标记为失败
            else:
                # 2. 存活期间的平滑约束
                steering_jerk = abs(action[1] - self.last_action[1])
                steering_mag = abs(action[1])
                reward -= (self.jerk_weight * (steering_jerk ** 2) + self.steering_weight * (steering_mag ** 2))
                
                # =========================================================
                # 🚫 [核心优化：封杀消极苟活漏洞]
                # 目的：防止主车学会“一脚刹车踩死，等NPC先走”的作弊让行策略。
                # 机制：将龟速惩罚阈值从 10.0m/s 提高到 15.0m/s (约 54km/h)。
                # 效果：逼迫模型学会在保持较高车速的情况下，通过“微调降速(如18m/s)”
                #       或者“向左变道”来完成高动态的避让博弈。
                # =========================================================
                if ego_speed_vx < 15.0:
                    reward -= (15.0 - ego_speed_vx) * 0.5 

        self.last_action = action.copy()
        info["ego_speed_vx"] = ego_speed_vx
        return next_obs, reward, terminated, truncated, info

# ----------------------------------------------------
# 🎯 Diffusion 专属宽容包装器
# ----------------------------------------------------
class DiffMergeAVControlWrapper(gym.Wrapper):
    def __init__(self, env, is_eval=False):
        super().__init__(env)
        self.is_eval = is_eval
        print(f"🔧 [Wrapper 原生 Merge Diff] 部署")

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        ego = self.env.unwrapped.vehicle
        ego_speed_vx = ego.speed
        
        # 🚫 同步部署铁腕裁判
        crashed = getattr(ego, "crashed", False)
        is_out_of_road = info.get("is_out_of_road", False)
        is_not_on_road = not getattr(ego, "on_road", True)
        is_sideways = abs(ego.heading) > 0.4
        is_reverse = ego_speed_vx < -1.0

        # 🚨 [核心修复] 物理死刑（如倒车、横摆失控）必须在任何模式下绝对生效！
        # 防止高 Q 权重发癫的模型在评估时拿到“免死金牌”从而出现全程倒车的灵异现象。
        if crashed or is_out_of_road or is_not_on_road or is_sideways or is_reverse:
            terminated = True
            if not self.is_eval:
                reward = -10.0
        else:
            if not self.is_eval:
                # 原生任务中，主要依靠速度奖励和存活奖励
                base_reward = 1.0
                speed_reward = min((ego_speed_vx - 15.0) / 10.0, 1.0) if ego_speed_vx >= 15.0 else -0.1
                reward = max(min(base_reward + speed_reward, 4.0), -10.0)

        info["ego_speed_vx"] = ego_speed_vx
        return next_obs, reward, terminated, truncated, info

# ----------------------------------------------------
# 🎯 核心工厂函数
# ----------------------------------------------------
def create_merge_env(env_name="merge-v0", render_mode="rgb_array", is_eval=False, algo="sac", wrapper_config=None, env_config=None):
    env = gym.make(env_name, render_mode=render_mode)
    unwrapped_env = env.unwrapped

    # 🐛 [BUG FIX 保留] 修复 highway_env 连续动作空间的底层源码缺陷
    original_rewards_fn = unwrapped_env._rewards
    def patched_rewards(action):
        if isinstance(action, np.ndarray): return original_rewards_fn(1)
        return original_rewards_fn(action)
    unwrapped_env._rewards = patched_rewards

    # =========================================================
    # 🎯 [核心优化：注入 TTC 同步传送魔法]
    # 目的：修复默认环境中“匝道NPC过早到达汇入口”的物理错位问题。
    # 机制：接管底层 reset() 函数，通过物理学公式(距离=速度×时间)强行重置双车位置。
    # =========================================================
    original_reset = unwrapped_env.reset
    def patched_reset(*args, **kwargs):
        # 先让底层完成原生的初始化
        obs, info = original_reset(*args, **kwargs)
        try:
            ego = unwrapped_env.vehicle
            road = unwrapped_env.road
            
            # =========================================================
            # 🎲 [TTC 动态抖动 (Jittering)]
            # 提取被当前环境 Seed 严格控制的底层随机数生成器。
            # 引入位置和速度的微小偏移，彻底粉碎 1v1 纯净决斗的同质化过拟合，
            # 同时保证同一个 Seed 下的评估考题永远一模一样。
            # =========================================================
            np_random = unwrapped_env.np_random
            ego_pos_jitter = np_random.uniform(-5.0, 5.0)
            ego_spd_jitter = np_random.uniform(-2.0, 2.0)
            npc_pos_jitter = np_random.uniform(-5.0, 5.0)
            npc_spd_jitter = np_random.uniform(-2.0, 2.0)

            # [主车设定]
            ego.lane_index = ("a", "b", 1)
            lane_ego = road.network.get_lane(("a", "b", 1))
            ego.position = lane_ego.position(30 + ego_pos_jitter, 0)
            ego.speed = 25.0 + ego_spd_jitter
            
            # [匝道 NPC 设定]
            ramp_lane = road.network.get_lane(("j", "k", 0))
            ramp_vehicles = [v for v in road.vehicles if v is not ego and v.lane_index == ("j", "k", 0)]
            if ramp_vehicles:
                npc = ramp_vehicles[0]
                npc.position = ramp_lane.position(50 + npc_pos_jitter, 0)
                npc.speed = 20.0 + npc_spd_jitter
                npc.target_speed = 25.0 + npc_spd_jitter # 保持持续加速意图
                
            # [极度关键]：物理位置被我们强行篡改后，原有的 obs 张量就作废了
            # 必须调用环境底层方法，重新发射雷达射线获取最新的状态张量
            obs = unwrapped_env.observation_type.observe()
        except Exception as e:
            # 容错：如果未来 highway-env 更新改了底层车道 ID ("a","b","j","k")
            # 捕获异常并静默退回默认生成位置，防止程序直接崩溃
            pass 
        return obs, info
    
    # 用我们打好补丁的 reset 替换环境底层的 reset
    unwrapped_env.reset = patched_reset
    # =========================================================

    # 🔧 基础配置：回归官方默认逻辑
    base_config = {
        "observation": {"type": "Kinematics", "vehicles_count": 5, "features": ["presence", "x", "y", "vx", "vy"], "absolute": False, "normalize": True},
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 15, "policy_frequency": 5,
        "controlled_vehicles": 1,
        "duration": 20, # 设定为 20 秒 (即 policy_frequency=5 下的 100 步)，与外层训练的 100 步完赛标准完美对齐
        "offroad_terminal": True,
        "collision_reward": -10.0 if not is_eval else -1.0,
        "high_speed_reward": 5.0 if not is_eval else 1.0,
        "reward_speed_range": [20, 30], "show_trajectories": True,
        # 显式关闭 right_lane_reward，防止自车被强行“吸”到右侧匝道去
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