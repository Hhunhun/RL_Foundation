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
    def __init__(self, env, jerk_weight=1.0, steering_weight=0.5, is_eval=False):
        super().__init__(env)
        self.last_action = np.zeros(self.env.action_space.shape)
        self.jerk_weight = jerk_weight
        self.steering_weight = steering_weight
        self.is_eval = is_eval
        mode_str = "训练模式" if jerk_weight > 0 else "评估模式"
        print(f"🏎️ [Wrapper 原生 Racetrack SAC - {mode_str}] 部署 | Jerk: {jerk_weight}")

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        ego = self.env.unwrapped.vehicle
        ego_speed_vx = ego.speed
        
        # 🚫 赛道专属物理裁判：必须捕捉到撞前车的行为
        crashed = getattr(ego, "crashed", False)
        is_out_of_road = info.get("is_out_of_road", False)
        is_not_on_road = not getattr(ego, "on_road", True)
        is_reverse = ego_speed_vx < -1.0

        # 1. 触发死刑条件 (强行接管底层的 terminated 判定)
        if crashed or is_out_of_road or is_not_on_road or is_reverse:
            terminated = True
            if not self.is_eval:
                reward -= 50.0 # 严惩一切死亡行为（包括追尾周车！）
            info["crashed"] = True # 统一标记为失败
        else:
            if not self.is_eval:
                # 2. 存活期间的平滑约束 (防画龙)
                steering_jerk = abs(action[1] - self.last_action[1])
                steering_mag = abs(action[1])
                reward -= (self.jerk_weight * (steering_jerk ** 2) + self.steering_weight * (steering_mag ** 2))
                
                # 🌟 [温和但坚定的催促] 惩罚低速，但控制力度防止自杀
                # 阈值提高到 15.0 逼迫超车，但权重设为 0.5。
                # 10m/s 跟车时每步扣 2.5 分，跟车 10 步扣 25 分，痛苦但好过 -50 分撞车死刑。
                if ego_speed_vx < 15.0:
                    reward -= (15.0 - ego_speed_vx) * 0.5 

                # 🌟 二次方对齐惩罚：防画龙的同时鼓励变道
                try:
                    road = self.env.unwrapped.road
                    # 注意：get_closest_lane_index 会动态返回离当前车辆最近的车道。变道后自动对齐新车道！
                    lane_index = road.network.get_closest_lane_index(ego.position, ego.heading)
                    lane = road.network.get_lane(lane_index)
                    lon, lat = lane.local_coordinates(ego.position)
                    target_heading = lane.heading_at(lon)
                    heading_diff = (ego.heading - target_heading + np.pi) % (2 * np.pi) - np.pi
                    
                    # [二次方惩罚]：走直线(lat=0)不扣分，稍微偏离罚得很轻，极端跨线才会有明显痛感
                    lat_penalty = (lat ** 2) * 0.05 
                    heading_penalty = abs(heading_diff) * 0.2 # 降低航向惩罚，鼓励敢打方向盘
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

    def _get_ahead_position(self, road, start_lane_index, start_s, ahead_distance):
        """沿着路网拓扑向前推演指定距离，确保车辆始终落在物理赛道内，避免弯道计算越界"""
        s = start_s + ahead_distance
        lane_index = start_lane_index
        lane = road.network.get_lane(lane_index)
        
        # 赛道由多段拼接而成，沿着 Graph 往下找，把 ahead_distance 消耗完
        for _ in range(10): # 限制查找层数防止死循环
            if s <= lane.length:
                break
            s -= lane.length
            next_from = lane_index[1]
            if next_from in road.network.graph and road.network.graph[next_from]:
                next_to = list(road.network.graph[next_from].keys())[0]
                lanes_list = road.network.graph[next_from][next_to]
                next_id = lane_index[2] if lane_index[2] < len(lanes_list) else 0
                lane_index = (next_from, next_to, next_id)
                lane = road.network.get_lane(lane_index)
            else:
                s = lane.length - 0.1 # 到底了，卡在路段末尾
                break
                
        return lane_index, lane, s

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
        # 赋予适中的初始速度，既能加速探索，又能防止开局由于速度过快反应不及而秒撞
        vehicle.speed = float(unwrapped.np_random.uniform(12.0, 18.0))
        vehicle.target_speed = vehicle.speed
        
        # 🛠️ [核心修复] 更新车辆在路网中的位置索引
        # 否则 Kinematics 观测器可能会保留旧车道的信息
        vehicle.lane_index = road.network.get_closest_lane_index(vehicle.position, vehicle.heading)
        vehicle.lane = road.network.get_lane(vehicle.lane_index)
        
        # 🚗 [核心改进] 概率性强制相遇机制 (Probabilistic Encounter)
        # 50% 概率遇到前方慢车（学习变道避让），50% 概率畅通无阻（学习极速切弯）
        other_vehicles = [v for v in road.vehicles if v is not vehicle]
        if other_vehicles:
            dummy = other_vehicles[0] # 锁定当前环境中的唯一一辆周车
            is_encounter = unwrapped.np_random.uniform(0, 1) < 0.5
            
            if is_encounter:
                # [近距相遇局]：距离 30~50 米。100% 挡道（同车道），纯粹训练初学者的紧急避让和打方向盘能力。
                ahead_dist = unwrapped.np_random.uniform(30.0, 50.0)
                start_lane_idx = vehicle.lane_index
                dummy_lane_idx, dummy_lane, dummy_s = self._get_ahead_position(road, start_lane_idx, random_s, ahead_dist)
            else:
                # [远距追击局 (爽跑局升级)]：放置在远方 100~150 米的随机车道。
                # 让自车先爽跑提速，随后以 25m/s+ 的真实高速完成追尾变道博弈！
                ahead_dist = unwrapped.np_random.uniform(100.0, 150.0)
                from_node, to_node, lane_id = vehicle.lane_index
                
                # 🛠️ [转移] 打破高速状态下“见车必躲”的条件反射
                # 70% 概率同车道刷新（必须变道避障），30% 概率旁边车道刷新（无需避让，保持定力直接通过）
                if unwrapped.np_random.uniform(0, 1) < 0.7:
                    start_lane_idx = vehicle.lane_index
                else:
                    lanes_list = road.network.graph.get(from_node, {}).get(to_node, [])
                    lanes_count = len(lanes_list) if len(lanes_list) > 0 else 1
                    other_lane_id = 1 - lane_id if lanes_count > 1 else lane_id
                    start_lane_idx = (from_node, to_node, other_lane_id)
                    
                dummy_lane_idx, dummy_lane, dummy_s = self._get_ahead_position(road, start_lane_idx, random_s, ahead_dist)
                
            # 周车统一设为龟速移动路障 (8~12m/s)，确保自车无论远近，迟早都会追上它！
            dummy.speed = float(unwrapped.np_random.uniform(8.0, 12.0))
            
            try:
                dummy_lane = road.network.get_lane(dummy_lane_idx)
            except Exception:
                dummy_lane_idx = vehicle.lane_index
                dummy_lane = vehicle.lane
                    
            dummy_pos = dummy_lane.position(dummy_s, 0)
            dummy.position = np.array(dummy_pos, dtype=float)
            dummy.heading = float(dummy_lane.heading_at(dummy_s))
            dummy.target_speed = dummy.speed
            
            # 🛠️ [核心修复] 彻底同步 IDM 周车的内部导航大脑
            # 必须重置其目标车道与路径，否则周车会为了回到原始出生点而猛打方向盘冲进草地！
            dummy.lane_index = road.network.get_closest_lane_index(dummy.position, dummy.heading)
            dummy.lane = road.network.get_lane(dummy.lane_index)
            
            # 洗脑：让周车认为自己本就该呆在这个新车道上
            if hasattr(dummy, "target_lane_index"):
                dummy.target_lane_index = dummy.lane_index
            if hasattr(dummy, "route"):
                dummy.route = [dummy.lane_index]

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
        # 🚗 显式声明仅产生 1 辆周车，配合概率性相遇机制进行单车博弈
        "vehicles_count": 1,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
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
        env = RacetrackAVControlWrapper(env, jerk_weight=jerk if not is_eval else 0.0, steering_weight=steering if not is_eval else 0.0, is_eval=is_eval)

    return env