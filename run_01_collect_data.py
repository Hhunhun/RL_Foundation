"""
阶段一：专家轨迹数据采集流水线 (Expert Data Collection)

此模块负责为后续的 Diffusion 预训练提供高质量的“教材”。
它会加载一个已经训练好的、具备高超驾驶技术的基准 SAC 模型（专家），让它在环境中自动驾驶。

目前支持两种采集模式：
[Mode 1] 保守底座采集：包含概率性行为抖动，过滤所有碰撞局，用于构建安全先验。
[Mode 2] 极速神仙局采集：关闭抖动，过滤碰撞且强制要求回合均速 > 22.0 m/s，用于拓展流形上限。
"""

import os
import time
import numpy as np
from datetime import datetime
import warnings

# 锁定项目根目录以确保数据保存路径绝对安全
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 忽略 Pygame 和某些底层库产生的烦人警告，保持控制台输出整洁
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from algorithms.sac.sac_agent import SACAgent
from envs.highway_wrapper import create_highway_env


def collect_expert_data(model_path, env_name="highway-v0", target_transitions=50000, mode=1):
    print("\n" + "=" * 60)
    print("🚀 [阶段一] 开始专家轨迹数据采集 (Expert Data Collection)")
    print(f"📦 目标采集量: {target_transitions} 步 | 当前模式: Mode {mode}")
    print(f"🧠 加载权重: {model_path}")
    print("=" * 60)

    # 创建评估模式的环境（is_eval=True 代表关闭训练期那些严苛的惩罚，仅测试纯粹的物理驾驶表现）
    env = create_highway_env(env_name, is_eval=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 实例化 SAC 专家，并挂载预训练好的模型权重
    agent = SACAgent(state_dim, action_dim, action_scale=max_action)
    try:
        agent.load_model(model_path)
    except Exception as e:
        print(f"❌ 模型权重加载失败，请检查路径: {model_path}\n错误信息: {e}")
        return None

    # 初始化用于存储离线数据集的字典
    dataset = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'terminals': []}
    collected_steps, successful_episodes, discarded_episodes = 0, 0, 0
    start_time = time.time()

    # 开始持续跑环境，直到收集够目标步数
    while collected_steps < target_transitions:
        state, _ = env.reset()
        # 创建临时列表，用于缓存当前这一局的数据
        ep_obs, ep_acts, ep_rews, ep_next_obs, ep_terms = [], [], [], [], []
        ep_speeds = [] # 用于 Mode 2 的速度统计

        crashed = False

        while True:
            # 1. 获取专家基准动作 (evaluate=True 表示关闭 SAC 的探索噪声，输出确定性的绝对最优解)
            action = agent.select_action(state, evaluate=True)

            # 2. 🚨 概率性注入行为抖动 (仅 Mode 1 启用) 🚨
            # 学术级数据增强操作：50% 概率保持绝对纯净，50% 概率加入微小的正态分布噪声。
            # Mode 2 需要纯粹的极速发挥，因此关闭抖动。
            if mode == 1 and np.random.rand() < 0.5:
                noise = np.random.normal(0, 0.05, size=action_dim)
                action = np.clip(action + noise, -1.0, 1.0)

            # 将动作输入物理环境，获取下一步的反馈
            next_state, reward, terminated, truncated, info = env.step(action)

            # 将这一步的数据暂存进当前局的缓存列表中
            ep_obs.append(state)
            ep_acts.append(action)
            ep_rews.append(reward)
            ep_next_obs.append(next_state)
            ep_terms.append(terminated)
            ep_speeds.append(info.get("ego_speed_vx", 0.0))

            state = next_state

            # 如果撞车/出轨(terminated) 或达到最大步数(truncated)，当前局结束
            if terminated or truncated:
                if terminated:
                    crashed = True
                break

        # 🚨 回合级淘汰机制 (Episode-level Filtering) 🚨
        mean_speed = np.mean(ep_speeds)
        accept_episode = False

        if mode == 1:
            # Mode 1 逻辑：只要没撞车就收下
            if not crashed:
                accept_episode = True
            else:
                reason = "发生碰撞/出界"
        elif mode == 2:
            # Mode 2 逻辑：神仙局必须同时满足【不撞车】且【均速 > 22.0】
            if not crashed and mean_speed > 22.0:
                accept_episode = True
            else:
                reason = f"撞车:{crashed}, 均速:{mean_speed:.2f}m/s 未达标"

        # 执行数据并入或丢弃
        if accept_episode:
            successful_episodes += 1
            collected_steps += len(ep_obs)
            dataset['observations'].extend(ep_obs)
            dataset['actions'].extend(ep_acts)
            dataset['rewards'].extend(ep_rews)
            dataset['next_observations'].extend(ep_next_obs)
            dataset['terminals'].extend(ep_terms)

            # 实时打印进度条
            progress = min(100.0, (collected_steps / target_transitions) * 100)
            print(f"\r✅ 进度: [{progress:5.1f}%] | 均速: {mean_speed:.2f} m/s | 已收: {collected_steps}/{target_transitions} 步 | 有效局: {successful_episodes}   ", end="")
        else:
            discarded_episodes += 1
            print(f"\r⚠️ 劣质对局被过滤 ({reason})... (已丢弃 {discarded_episodes} 局)               ", end="")

    env.close()

    # 将数据截断到精确的 target_transitions 数量，并转化为 numpy 数组
    for key in dataset.keys():
        dataset[key] = np.array(dataset[key][:target_transitions], dtype=np.float32)

    # 根据采集模式和当前时间生成专属的存档目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_prefix = "dataset_v5_base_" if mode == 1 else "dataset_v6_pro_"
    save_dir = os.path.join(PROJECT_ROOT, "data", "expert_data", f"{dataset_prefix}{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(save_dir, "expert_transitions.npz")

    # 采用高压缩比格式保存 .npz 文件，大幅节省硬盘空间
    np.savez_compressed(data_path, **dataset)

    print(f"\n\n💾 数据集已完美保存至: {data_path}")
    print(f"⏱️ 耗时: {(time.time() - start_time) / 60:.2f} 分钟")

    # 🚨 返回数据路径，供主控流水线 (02_train_pipeline) 直接读取传递给下一阶段
    return data_path


if __name__ == "__main__":
    # ==========================================
    # 终端交互控制台
    # ==========================================
    print("🤖 欢迎使用专家数据采集终端")
    print("==========================================")
    print("[1] 基础底座模式 (Mode 1): 使用 v5.0 模型，含行为抖动，仅过滤碰撞局。(适用构建安全保底)")
    print("[2] 极速神仙局模式 (Mode 2): 使用 v6.0 模型，关闭抖动，严格过滤碰撞且强制要求均速 > 22.0 m/s。(适用破局上限)")
    print("==========================================")

    choice = input("👉 请输入采集模式 (1 或 2，默认 1): ").strip()

    # 你刚才提供的 v5.0 和 v6.0 模型相对路径
    V5_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "highway-v0_SAC_20260330_135449", "sac_highway_final.pth")
    V6_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "highway-v0_SAC_20260330_213300", "sac_highway_final.pth")

    if choice == '2':
        selected_mode = 2
        selected_model = V6_MODEL_PATH
    else:
        selected_mode = 1
        selected_model = V5_MODEL_PATH

    # 通宵挂机推荐目标量：50000步
    TARGET_STEPS = 50000

    if os.path.exists(selected_model):
        collect_expert_data(model_path=selected_model, target_transitions=TARGET_STEPS, mode=selected_mode)
    else:
        print(f"\n❌ 找不到指定的权重文件！请确保路径正确: {selected_model}")