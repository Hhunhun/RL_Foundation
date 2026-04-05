"""
阶段一：专家轨迹数据采集流水线 (Expert Data Collection)

此模块负责为后续的 Diffusion 预训练提供高质量的“教材”。
它会加载一个已经训练好的、具备高超驾驶技术的基准 SAC 模型（专家），让它在环境中自动驾驶。
核心特性：
1. 概率性行为抖动：在专家的完美动作中偶尔加入微小偏差，促使专家做出“纠偏动作”，从而让生成的数据集更具鲁棒性。
2. 绝对纯净清洗：严格过滤数据，一旦发生撞车等致命错误（terminated），整局数据全部丢弃，确保喂给 Diffusion 模型的都是 100% 安全的示范。
"""

import os
import time
import numpy as np
from datetime import datetime
import warnings

# 忽略 Pygame 和某些底层库产生的烦人警告，保持控制台输出整洁
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from algorithms.sac.sac_agent import SACAgent
from envs.highway_wrapper import create_highway_env


def collect_expert_data(model_path, env_name="highway-v0", target_transitions=50000):
    print("=" * 60)
    print("🚀 [阶段一] 开始专家轨迹数据采集 (Expert Data Collection)")
    print(f"📦 目标采集量: {target_transitions} 步")
    print("=" * 60)

    # 创建评估模式的环境（is_eval=True 代表关闭训练期那些严苛的惩罚，仅测试纯粹的物理驾驶表现）
    env = create_highway_env(env_name, is_eval=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 实例化 SAC 专家，并挂载预训练好的模型权重
    agent = SACAgent(state_dim, action_dim, action_scale=max_action)
    agent.load_model(model_path)

    # 初始化用于存储离线数据集的字典
    dataset = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'terminals': []}
    collected_steps, successful_episodes, discarded_episodes = 0, 0, 0
    start_time = time.time()

    # 开始持续跑环境，直到收集够目标步数
    while collected_steps < target_transitions:
        state, _ = env.reset()
        # 创建临时列表，用于缓存当前这一局的数据
        ep_obs, ep_acts, ep_rews, ep_next_obs, ep_terms = [], [], [], [], []

        while True:
            # 1. 获取专家基准动作 (evaluate=True 表示关闭 SAC 的探索噪声，输出确定性的绝对最优解)
            action = agent.select_action(state, evaluate=True)

            # 2. 🚨 概率性注入行为抖动 (Probabilistic Behavioral Jitter) 🚨
            # 学术级数据增强操作：50% 概率保持绝对纯净，50% 概率加入微小的正态分布噪声。
            # 这能迫使专家展示“稍微偏离车道后如何救车”的动作，防止后续的克隆网络对完美状态过度拟合。
            if np.random.rand() < 0.5:
                noise = np.random.normal(0, 0.05, size=action_dim)
                action = np.clip(action + noise, -1.0, 1.0)

            # 将动作输入物理环境，获取下一步的反馈
            next_state, reward, terminated, truncated, _ = env.step(action)

            # 将这一步的数据暂存进当前局的缓存列表中
            ep_obs.append(state)
            ep_acts.append(action)
            ep_rews.append(reward)
            ep_next_obs.append(next_state)
            ep_terms.append(terminated)

            state = next_state
            # 如果撞车/出轨(terminated) 或达到最大步数(truncated)，当前局结束
            if terminated or truncated:
                break

        # 🚨 数据清洗与过滤逻辑 🚨
        if terminated:
            # 如果这局是以 terminated 结束的（代表专家翻车了），我们宁缺毋滥，直接丢弃这整局的数据！
            discarded_episodes += 1
            print(f"\r⚠️ 发现瑕疵轨迹，整局丢弃... (已丢弃 {discarded_episodes} 局)", end="")
        else:
            # 如果是平稳跑完达到 truncated 限制，说明这是一局完美的驾驶，将其正式并入大数据库
            successful_episodes += 1
            collected_steps += len(ep_obs)
            dataset['observations'].extend(ep_obs)
            dataset['actions'].extend(ep_acts)
            dataset['rewards'].extend(ep_rews)
            dataset['next_observations'].extend(ep_next_obs)
            dataset['terminals'].extend(ep_terms)

            # 实时打印进度条
            progress = min(100.0, (collected_steps / target_transitions) * 100)
            print(
                f"\r✅ 进度: [{progress:5.1f}%] | 已收集: {collected_steps}/{target_transitions} 步 | 有效局数: {successful_episodes}",
                end="")

    env.close()

    # 将数据截断到精确的 target_transitions 数量，并转化为 numpy 数组
    for key in dataset.keys():
        dataset[key] = np.array(dataset[key][:target_transitions], dtype=np.float32)

    # 根据当前时间生成唯一的存档目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("data", "expert_data", f"dataset_v5_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(save_dir, "expert_transitions.npz")

    # 采用高压缩比格式保存 .npz 文件，大幅节省硬盘空间
    np.savez_compressed(data_path, **dataset)

    print(f"\n💾 数据集已保存至: {data_path}")

    # 🚨 返回数据路径，供主控流水线 (02_train_pipeline) 直接读取传递给下一阶段
    return data_path