"""
Module: Racetrack Environment Wrappers (赛道竞速场景封装器)
Description:
    本模块构建了强化学习智能体与高动态极限赛道环境 (racetrack-v0) 之间的交互桥梁。
    针对赛道场景特有的连续大曲率过弯与高速超车需求，本模块引入了拓扑感知状态增强与
    程序化域随机化 (Domain Randomization) 机制，强制模型在极限动态下学习稳健的通用规控策略。

Key Features:
    - Topology-Aware State Augmentation: 将底层笛卡尔坐标转换为局部的 Frenet 坐标系特征 (横向/航向偏差)，
      赋予神经网络对赛道几何走向的显式感知能力。
    - Procedural Encounter Generation: 独创的程序化随机生成引擎。通过劫持路网拓扑并在不同时空尺度
      (近距紧急避障 vs. 远距高速追击) 注入 NPC 车辆，彻底粉碎单点过拟合 (Trajectory Overfitting)。
    - Quadratic Tracking Regularization: 针对 SAC 算法设计了基于 L2 范数的车道保持惩罚，
      在抑制“画龙 (Weaving)”现象的同时，保留了模型超车变道的机动性自由度。
"""

import gymnasium as gym
import highway_env
import numpy as np

# ----------------------------------------------------
# 🎯 拓扑感知状态向量化包装器 (Topology-Aware State Vectorization Wrapper)
# ----------------------------------------------------
class RacetrackFlattenWrapper(gym.ObservationWrapper):
    """
    状态重构与展平算子。
    在原生多智能体运动学矩阵的基础上，显式注入基于 Frenet 坐标系的局部几何误差 
    (Lateral Error & Heading Error)，消除模型在连续弯道中“盲目死记硬背”的弊端。
    """
    def __init__(self, env):
        super().__init__(env)
        obs_space = self.env.observation_space
        # 扩展状态空间维数以容纳 Frenet 坐标系投影特征
        self.flat_dim = np.prod(obs_space.shape) + 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.flat_dim,), dtype=np.float32
        )
        
    def observation(self, obs):
        flat_obs = np.array(obs, dtype=np.float32).flatten()
        
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        road = getattr(unwrapped, "road", None)
        
        lat_error, heading_error = 0.0, 0.0
        # 动态解析拓扑图 (Topology Graph)，计算主车相对于当前最优参考线的空间姿态偏差
        if road and ego:
            try:
                lane_index = road.network.get_closest_lane_index(ego.position, ego.heading)
                lane = road.network.get_lane(lane_index)
                lon, lat_error = lane.local_coordinates(ego.position)
                target_heading = lane.heading_at(lon)
                # 将角度偏差规范化映射至 [-pi, pi] 主值区间
                heading_error = (ego.heading - target_heading + np.pi) % (2 * np.pi) - np.pi
            except Exception:
                pass # 拓扑容错机制：屏蔽环境初始化瞬间尚未建立路网抛出的异常
                
        # 执行特征无量纲化 (假设赛道标准半宽约为 5.0m，最大合理航向差为 pi)
        return np.concatenate([flat_obs, [np.clip(lat_error/5.0, -1, 1), np.clip(heading_error/np.pi, -1, 1)]], dtype=np.float32)

# ----------------------------------------------------
# 🎯 SAC 专属运动学寻迹控制包装器 (Kinematics & Tracking Control Wrapper for SAC)
# ----------------------------------------------------
class RacetrackAVControlWrapper(gym.Wrapper):
    """
    面向高速竞速任务的综合奖励整形层 (Comprehensive Reward Shaping Layer)。
    叠加一阶动力学约束、低速启发式惩罚与二次方寻迹误差惩罚 (Quadratic Tracking Error Penalty)，
    以塑造兼具激进超车能力与稳健过弯性能的高阶驾驶策略。
    """
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
        
        # 刚性物理裁判：精准捕捉越界与非弹性碰撞行为 (包括追尾与侧碰)
        crashed = getattr(ego, "crashed", False)
        is_out_of_road = info.get("is_out_of_road", False)
        is_not_on_road = not getattr(ego, "on_road", True)
        is_reverse = ego_speed_vx < -1.0

        # 1. 终止状态判定 (Terminal State Identification)
        if crashed or is_out_of_road or is_not_on_road or is_reverse:
            terminated = True
            if not self.is_eval:
                reward -= 50.0 # 施加赛道环境的极大值灾难惩罚
            info["crashed"] = True 
        else:
            if not self.is_eval:
                # 2. 动态生存期的平滑度与性能正则化 (Kinematics & Performance Regularization)
                steering_jerk = abs(action[1] - self.last_action[1])
                steering_mag = abs(action[1])
                reward -= (self.jerk_weight * (steering_jerk ** 2) + self.steering_weight * (steering_mag ** 2))
                
                # [动能激励启发式启发 (Kinetic Incentive Heuristic)]
                # 提升低速域惩罚下限至 15.0m/s，倒逼模型在狭窄弯道中也必须维持较高的时间效率，
                # 坚决抑制模型为逃避“越界死刑”而演化出的“龟速爬行”消极策略。
                if ego_speed_vx < 15.0:
                    reward -= (15.0 - ego_speed_vx) * 0.5 

                # [二次方局部寻迹惩罚 (Quadratic Local Tracking Penalty)]
                # 通过动态挂载最近车道参考线，施加基于 L2 范数的几何偏差惩罚。
                # 物理机制：允许模型在车道内执行微小幅度的最优切线过弯 (惩罚极低)，
                # 但强力遏制大幅度的“画龙 (Weaving)”与危险的跨线压界行为。
                try:
                    road = self.env.unwrapped.road
                    lane_index = road.network.get_closest_lane_index(ego.position, ego.heading)
                    lane = road.network.get_lane(lane_index)
                    lon, lat = lane.local_coordinates(ego.position)
                    target_heading = lane.heading_at(lon)
                    heading_diff = (ego.heading - target_heading + np.pi) % (2 * np.pi) - np.pi
                    
                    lat_penalty = (lat ** 2) * 0.05 
                    heading_penalty = abs(heading_diff) * 0.2 # 降低航向约束系数以换取大曲率过弯机动性
                    reward -= (lat_penalty + heading_penalty)
                except Exception:
                    pass

        self.last_action = action.copy()
        info["ego_speed_vx"] = ego_speed_vx
        return next_obs, reward, terminated, truncated, info

# ----------------------------------------------------
# 🎯 Diffusion 专属宽容生成包装器 (Tolerant Constraint Wrapper for Diffusion)
# ----------------------------------------------------
class DiffRacetrackAVControlWrapper(gym.Wrapper):
    """
    专为生成式扩散模型定制的价值防爆套件。
    物理意义：完全剥离了繁杂的寻迹与抖动惩罚，通过稠密的正向速度激励引导 Diffusion 
    执行去噪探索，从根源上消除多维复合惩罚所引发的 Q 价值崩溃 (Q-Value Collapse) 问题。
    """
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

        # 保留不可逾越的物理隔离墙 (Absolute Physical Isolation Boundaries)
        if crashed or is_out_of_road or is_not_on_road or is_reverse:
            terminated = True
            if not self.is_eval:
                reward = -10.0
        else:
            if not self.is_eval:
                base_reward = 1.0
                # 赛道圈速驱动激励 (Lap Time Driven Incentive)
                speed_reward = min((ego_speed_vx - 10.0) / 10.0, 1.0) if ego_speed_vx >= 10.0 else -0.1
                reward = max(min(base_reward + speed_reward, 4.0), -10.0)

        info["ego_speed_vx"] = ego_speed_vx
        return next_obs, reward, terminated, truncated, info

# ----------------------------------------------------
# 🎯 程序化域随机化包装器 (Procedural Domain Randomization Wrapper)
# ----------------------------------------------------
class RacetrackRandomSpawnWrapper(gym.Wrapper):
    """
    赛道时空拓扑的随机重构器 (Spatiotemporal Topology Random Reconstructor)。
    接管仿真底层的初始化管线，在复杂的赛道图结构中执行程序化的状态锚点生成。
    核心学术贡献：彻底摧毁因赛道起点单一导致的神经网络确定性过拟合机制，通过多模态的
    随机遭遇战，促使智能体学会在连续流行空间内的普适化规控映射。
    """
    def __init__(self, env, is_eval=False):
        super().__init__(env)
        self.is_eval = is_eval
        self.reset_count = 0
        if not is_eval:
            print("🎲 [Wrapper] 程序化域随机化引擎 (Domain Randomization) 已激活 - 覆盖全赛道拓扑")

    def _get_ahead_position(self, road, start_lane_index, start_s, ahead_distance):
        """
        路网拓扑前向追踪引擎 (Forward Topology Ray-casting Engine)。
        基于有向图结构沿车道向前方进行跨路段递归推演，确保在曲率多变的连续拼接赛道中，
        计算出的 NPC 生成锚点严格隶属于合法物理行驶空间。
        """
        s = start_s + ahead_distance
        lane_index = start_lane_index
        lane = road.network.get_lane(lane_index)
        
        # 执行有限深度的图节点遍历 (Graph Traversal)，消耗预期相对距离
        for _ in range(10): 
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
                s = lane.length - 0.1 # 拓扑边界阻断：回归当前段物理末端
                break
                
        return lane_index, lane, s

    def reset(self, **kwargs):
        # 1. 触发底层引擎的基础内存初始化
        obs, info = self.env.reset(**kwargs)
        
        if self.is_eval:
            return obs, info

        unwrapped = self.env.unwrapped
        road = unwrapped.road
        vehicle = unwrapped.vehicle
        
        # 2. 执行安全域内的随机锚点采样 (Safe Zone Random Anchor Sampling)
        lanes = road.network.lanes_list()
        valid_lanes = [l for l in lanes if l.length > 5.0]
        if not valid_lanes: valid_lanes = lanes
        
        random_lane = unwrapped.np_random.choice(valid_lanes)
        # 约束生成范围在 [0.1, 0.9] 闭区间内，规避跨片段拼接区产生的非线性奇点异常
        random_s = unwrapped.np_random.uniform(0.1 * random_lane.length, 0.9 * random_lane.length)
        
        # 3. 强行覆写主车初始物理场 (Ego State Forcible Override)
        new_pos = random_lane.position(random_s, 0)
        new_heading = random_lane.heading_at(random_s)
        
        vehicle.position = np.array(new_pos, dtype=float)
        vehicle.heading = float(new_heading)
        # 注入动能初值扰动：降低静止启动的无效计算，并提供动态博弈势能
        vehicle.speed = float(unwrapped.np_random.uniform(12.0, 18.0))
        vehicle.target_speed = vehicle.speed
        
        # 强制底层引擎执行图节点位置再同步，刷新局部感受野缓存
        vehicle.lane_index = road.network.get_closest_lane_index(vehicle.position, vehicle.heading)
        vehicle.lane = road.network.get_lane(vehicle.lane_index)
        
        # =========================================================
        # 🚗 [多模态随机遭遇机制 (Multi-modal Random Encounter Engine)]
        # 机制分解：以 50/50 离散概率动态构建具有截然不同挑战目标的强化学习课程。
        # =========================================================
        other_vehicles = [v for v in road.vehicles if v is not vehicle]
        if other_vehicles:
            dummy = other_vehicles[0] # 获取游离靶标节点
            is_encounter = unwrapped.np_random.uniform(0, 1) < 0.5
            
            if is_encounter:
                # 模式 A [近场紧急避让 (Close-Quarter Evasion)]
                # 时空分布：前方 30~50m 内 100% 同构侵入当前车道。
                # 学术目标：倒逼模型打破稳态巡航预期，习得在连续大曲率中紧急重构转向指令与强刹车技能。
                ahead_dist = unwrapped.np_random.uniform(30.0, 50.0)
                start_lane_idx = vehicle.lane_index
                dummy_lane_idx, dummy_lane, dummy_s = self._get_ahead_position(road, start_lane_idx, random_s, ahead_dist)
            else:
                # 模式 B [远距高速超车 (High-Speed Chase & Overtake)]
                # 时空分布：前方 100~150m 外随机生成，并以一定概率 (30%) 偏离本车道。
                # 学术目标：诱导主车率先累积巨大纵向动能，在真实的高速极限状态下评估其变道决策的安全性与稳定性。
                ahead_dist = unwrapped.np_random.uniform(100.0, 150.0)
                from_node, to_node, lane_id = vehicle.lane_index
                
                # 引入侧向欺骗噪声，打破模型建立“视线内有车必盲目变道”的错误因果绑定
                if unwrapped.np_random.uniform(0, 1) < 0.7:
                    start_lane_idx = vehicle.lane_index
                else:
                    lanes_list = road.network.graph.get(from_node, {}).get(to_node, [])
                    lanes_count = len(lanes_list) if len(lanes_list) > 0 else 1
                    other_lane_id = 1 - lane_id if lanes_count > 1 else lane_id
                    start_lane_idx = (from_node, to_node, other_lane_id)
                    
                dummy_lane_idx, dummy_lane, dummy_s = self._get_ahead_position(road, start_lane_idx, random_s, ahead_dist)
                
            # 施加低速扰动约束，构成不可忽视的实体路障效应
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
            
            # 彻底清洗 IDM 周边车辆内部由原生环境分配的寻路导航缓存 (Routing Cache)
            # 否则将导致逻辑冲突：物理场上被移走，大脑却仍然指挥其实施危险的大倾角切角变道返回初始原点
            dummy.lane_index = road.network.get_closest_lane_index(dummy.position, dummy.heading)
            dummy.lane = road.network.get_lane(dummy.lane_index)
            
            if hasattr(dummy, "target_lane_index"):
                dummy.target_lane_index = dummy.lane_index
            if hasattr(dummy, "route"):
                dummy.route = [dummy.lane_index]

        # 4. 执行状态快照：强制雷达系统对重置后的全局物理引擎执行重新扫描
        obs = unwrapped.observation_type.observe()
        
        self.reset_count += 1
        if self.reset_count % 100 == 0:
            print(f"📍 [Debug] Episode {self.reset_count} 随机生成位置: {vehicle.position.round(1)}, 航向: {round(vehicle.heading, 2)}")
             
        return obs, info

# ----------------------------------------------------
# 🎯 核心环境构造工厂 (Environment Construction Factory)
# ----------------------------------------------------
def create_racetrack_env(env_name="racetrack-v0", render_mode="rgb_array", is_eval=False, algo="sac", wrapper_config=None, env_config=None):
    """
    赛道竞速场景的标准化构建流水线。
    组合了底层物理动力学配置、多阶段随机化扩展、以及针对于异构架构动态路由的不同防御型封装模块。
    """
    env = gym.make(env_name, render_mode=render_mode)
    unwrapped_env = env.unwrapped

    # 🐛 [底层缺陷修补] 修复原生框架解析 ContinuousAction 机制时的奖励结算遗漏
    original_rewards_fn = unwrapped_env._rewards
    def patched_rewards(action):
        if isinstance(action, np.ndarray): return original_rewards_fn(1)
        return original_rewards_fn(action)
    unwrapped_env._rewards = patched_rewards

    # 🔧 赛道专属基础物理规范与 MDP 设计约束
    base_config = {
        # 强制暴露车身朝向解耦投影特征 (cos_h, sin_h)，建立物理空间绝对姿态认知
        "observation": {"type": "Kinematics", "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"], "absolute": False, "normalize": True},
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "controlled_vehicles": 1,
        # 显式截断其他无关游离交通流，构建纯净的基于程序化生成的 1v1 极限博弈域
        "vehicles_count": 1,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "duration": 50,
        "offroad_terminal": True,
        "collision_reward": -10.0 if not is_eval else -1.0,
        "high_speed_reward": 5.0 if not is_eval else 1.0,
        "reward_speed_range": [15, 30],
        "show_trajectories": True
    }

    if env_config: base_config.update(env_config)
    env.unwrapped.configure(base_config)
    env.reset() 
    
    # --- 组合套件级联注入 (Cascaded Injection Process) ---
    # 1. 首层嵌入拓扑随机化引擎 (必须保留多维结构表征以实施图遍历)
    env = RacetrackRandomSpawnWrapper(env, is_eval=is_eval)
    
    # 2. 嵌入张量降维算子
    env = RacetrackFlattenWrapper(env)

    # 3. 末端路由架构专属约束模块
    if algo == "diff":
        env = DiffRacetrackAVControlWrapper(env, is_eval=is_eval)
    else:
        jerk = wrapper_config.get("jerk_weight", 1.0) if wrapper_config else 1.0
        steering = wrapper_config.get("steering_weight", 0.1) if wrapper_config else 0.1
        env = RacetrackAVControlWrapper(env, jerk_weight=jerk if not is_eval else 0.0, steering_weight=steering if not is_eval else 0.0, is_eval=is_eval)

    return env