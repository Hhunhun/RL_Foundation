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
        # 🌟 [核心升级]：除了原生展平外，额外扩展 2 个维度：横向偏差、航向偏差
        self.flat_dim = np.prod(obs_space.shape) + 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, # 放宽边界，容纳新增的标准化偏差值
            shape=(self.flat_dim,), dtype=np.float32
        )
        
    def observation(self, obs):
        flat_obs = np.array(obs, dtype=np.float32).flatten()
        
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        road = getattr(unwrapped, "road", None)
        
        lat_error, heading_error = 0.0, 0.0
        # 🌟 动态计算相对于当前赛道的偏差特征，赋予模型“感知赛道走向”的能力
        if road and ego:
            try:
                lane_index = road.network.get_closest_lane_index(ego.position, ego.heading)
                lane = road.network.get_lane(lane_index)
                lon, lat_error = lane.local_coordinates(ego.position)
                target_heading = lane.heading_at(lon)
                # 将角度差规范化到 [-pi, pi] 之间
                heading_error = (ego.heading - target_heading + np.pi) % (2 * np.pi) - np.pi
            except Exception:
                pass # 容错处理：环境初始化瞬间可能还没有完全构建路网
                
        # 标准化后拼接（假设赛道半宽约 5m，航向差不超过 pi）
        return np.concatenate([flat_obs, [np.clip(lat_error/5.0, -1, 1), np.clip(heading_error/np.pi, -1, 1)]], dtype=np.float32)

# ----------------------------------------------------
# 🎯 SAC 专属 AV Control 包装器
# ----------------------------------------------------
class RacetrackAVControlWrapper(gym.Wrapper):
    def __init__(self, env, jerk_weight=1.0, steering_weight=0.5):
        super().__init__(env)
        self.last_action = np.zeros(self.env.action_space.shape)
        self.jerk_weight = jerk_weight
        self.steering_weight = steering_weight
        mode_str = "训练模式" if jerk_weight > 0 else "评估模式"
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

                # 🌟 [新增] 密集对齐引导：告诉模型如何顺着赛道开
                try:
                    road = self.env.unwrapped.road
                    lane_index = road.network.get_closest_lane_index(ego.position, ego.heading)
                    lane = road.network.get_lane(lane_index)
                    lon, lat = lane.local_coordinates(ego.position)
                    target_heading = lane.heading_at(lon)
                    heading_diff = (ego.heading - target_heading + np.pi) % (2 * np.pi) - np.pi
                    
                    # 车头越歪、偏离中心线越远，惩罚越重。这替代了配置里的 use_lane_centering
                    lat_penalty = abs(lat) * 0.1 
                    heading_penalty = abs(heading_diff) * 1.5
                    reward -= (lat_penalty + heading_penalty)
                except Exception:
                    pass

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
# 🎯 赛道随机起点包装器 (修复版)
# ----------------------------------------------------
class RacetrackRandomSpawnWrapper(gym.Wrapper):
    def __init__(self, env, is_eval=False):
        super().__init__(env)
        self.is_eval = is_eval
        self.reset_count = 0
        if not is_eval:
            print("🎲 [Wrapper] 随机起点模式已激活 - 覆盖全赛道范围")

    def reset(self, **kwargs):
        # 1. 执行原生重置
        obs, info = self.env.reset(**kwargs)
        
        if self.is_eval:
            return obs, info

        unwrapped = self.env.unwrapped
        road = unwrapped.road
        vehicle = unwrapped.vehicle
        
        # 2. 随机挑选车道 (过滤掉长度过短的连接段)
        lanes = road.network.lanes_list()
        valid_lanes = [l for l in lanes if l.length > 5.0]
        if not valid_lanes: valid_lanes = lanes
        
        random_lane = unwrapped.np_random.choice(valid_lanes)
        # 在车道 10% 到 90% 的位置随机采样，避免刷在衔接处导致瞬间判死
        random_s = unwrapped.np_random.uniform(0.1 * random_lane.length, 0.9 * random_lane.length)
        
        # 3. 强制物理状态同步
        new_pos = random_lane.position(random_s, 0)
        new_heading = random_lane.heading_at(random_s)
        
        vehicle.position = np.array(new_pos, dtype=float)
        vehicle.heading = float(new_heading)
        # 赋予一个初始速度，防止初始阶段因为速度过低被惩罚
        vehicle.speed = float(unwrapped.np_random.uniform(10, 20))
        vehicle.target_speed = vehicle.speed
        
        # 🛠️ [核心修复] 更新车辆在路网中的位置索引
        # 否则 Kinematics 观测器可能会保留旧车道的信息
        vehicle.lane_index = road.network.get_closest_lane_index(vehicle.position, vehicle.heading)
        vehicle.lane = road.network.get_lane(vehicle.lane_index)
        
        # 4. 重新获取观测
        obs = unwrapped.observation_type.observe()
        
        self.reset_count += 1
        if self.reset_count % 100 == 0:
            print(f"📍 [Debug] Episode {self.reset_count} 随机生成位置: {vehicle.position.round(1)}, 航向: {round(vehicle.heading, 2)}")
             
        return obs, info

# ----------------------------------------------------
# 🎯 修改工厂函数以集成随机起点
# ----------------------------------------------------
def create_racetrack_env(env_name="racetrack-v0", render_mode="rgb_array", is_eval=False, algo="sac", wrapper_config=None, env_config=None):
    env = gym.make(env_name, render_mode=render_mode)
    unwrapped_env = env.unwrapped

    # 修复 highway_env 连续动作空间的底层源码缺陷
    original_rewards_fn = unwrapped_env._rewards
    def patched_rewards(action):
        if isinstance(action, np.ndarray): return original_rewards_fn(1)
        return original_rewards_fn(action)
    unwrapped_env._rewards = patched_rewards

    # 🔧 赛道专属基础配置
    base_config = {
        # 🌟 [核心修复]：加入 cos_h 和 sin_h，让模型感知自己的车头朝向！
        # 结合上述我们在 wrapper 里注入的横纵向偏差，模型彻底告别“盲目记忆”
        "observation": {"type": "Kinematics", "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"], "absolute": False, "normalize": True},
        # 动作空间：使用连续控制，适应赛道转向与油门控制需求
        "action": {"type": "ContinuousAction"},
        # 时间尺度：仿真频率(物理步长)与决策频率(RL采样频率)，保持在 3:1 以获得平滑控制曲线
        "simulation_frequency": 15,
        "policy_frequency": 5,
        # 车辆控制：指定由 RL 智能体操控的车辆数量
        "controlled_vehicles": 1,
        # 任务约束：单回合最大时长，单位为秒
        "duration": 50,
        # 物理限制：驶出路网边界即终止，作为硬性约束防止模型探索无效空间
        "offroad_terminal": True,
        # 奖励整形：根据是否在评估模式动态调整惩罚力度，引导模型在训练时规避碰撞
        "collision_reward": -10.0 if not is_eval else -1.0,
        # 速度奖励：鼓励模型在目标速度区间(15m/s - 30m/s)内高速过弯
        "high_speed_reward": 5.0 if not is_eval else 1.0,
        "reward_speed_range": [15, 30],
        # 可视化：启用轨迹显示，便于评估与调试过程中的车辆姿态跟踪
        "show_trajectories": True
    }

    if env_config: base_config.update(env_config)
    env.unwrapped.configure(base_config)
    env.reset() 
    
    # --- 包装链 ---
    # 1. 先做随机化（因为它需要访问未展平的 road 对象）
    env = RacetrackRandomSpawnWrapper(env, is_eval=is_eval)
    
    # 2. 展平观测
    env = RacetrackFlattenWrapper(env)

    # 3. 算法特定控制
    if algo == "diff":
        env = DiffRacetrackAVControlWrapper(env, is_eval=is_eval)
    else:
        jerk = wrapper_config.get("jerk_weight", 1.0) if wrapper_config else 1.0
        steering = wrapper_config.get("steering_weight", 0.1) if wrapper_config else 0.1
        env = RacetrackAVControlWrapper(env, jerk_weight=jerk if not is_eval else 0.0, steering_weight=steering if not is_eval else 0.0)

    return env