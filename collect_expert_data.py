import os
import time
import numpy as np
from datetime import datetime
import warnings

# 屏蔽底层环境的烦人警告
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from algorithms.sac.sac_agent import SACAgent
from envs.highway_wrapper import create_highway_env


def collect_expert_data(model_path, env_name="highway-v0", target_transitions=50000):
    """
    在线静默运行 Baseline 模型，采集并清洗高质量专家轨迹数据。

    :param model_path: 专家模型的权重路径 (.pth)
    :param env_name: 环境名称
    :param target_transitions: 目标收集的转移步数 (推荐 5万 - 10万步)
    """
    print("=" * 60)
    print("🚀 [阶段一] 开始专家轨迹数据采集 (Expert Data Collection)")
    print(f"📦 目标采集量: {target_transitions} 步 (Transitions)")
    print(f"🧠 母体模型: {model_path}")
    print("=" * 60)

    # 1. 强制使用纯净客观的评估环境 (剥离人工训练惩罚)
    env = create_highway_env(env_name, is_eval=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 2. 实例化并加载 v5.0 专家大脑
    agent = SACAgent(state_dim, action_dim, action_scale=max_action)
    try:
        agent.load_model(model_path)
        print("✅ 专家模型权重加载成功！")
    except Exception as e:
        print(f"❌ 加载模型失败，请检查路径: {e}")
        env.close()
        return

    # 3. 全局数据池 (Global Buffer)
    dataset = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': [],
        'terminals': []  # 记录环境是否因为物理限制(碰撞/越野)结束
    }

    collected_steps = 0
    successful_episodes = 0
    discarded_episodes = 0

    start_time = time.time()

    # 4. 开始采数大循环
    while collected_steps < target_transitions:
        state, _ = env.reset()

        # 临时回合池 (Episode Buffer)：用于实现“撞车整局作废”的过滤逻辑
        ep_obs, ep_acts, ep_rews, ep_next_obs, ep_terms = [], [], [], [], []
        ep_reward_sum = 0

        while True:
            # evaluate=True: 强制 SAC 输出高斯分布的均值(最确定的专家动作)，去除探索噪声
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            ep_obs.append(state)
            ep_acts.append(action)
            ep_rews.append(reward)
            ep_next_obs.append(next_state)
            ep_terms.append(terminated)

            ep_reward_sum += reward
            state = next_state

            if terminated or truncated:
                break

        # 5. 数据清洗过滤 (Data Filtering)
        # 如果 terminated 为 True，说明发生了碰撞或驶出边界
        # 专家数据集绝不能包含“教你怎么撞车”的数据！
        if terminated:
            discarded_episodes += 1
            print(f"\r⚠️ 发现瑕疵轨迹 (碰撞/出界)，整局丢弃... (已丢弃 {discarded_episodes} 局)", end="")
        else:
            # 只有完美跑完的局 (truncated=True 且 terminated=False) 才会被合入主数据池
            successful_episodes += 1
            collected_steps += len(ep_obs)

            dataset['observations'].extend(ep_obs)
            dataset['actions'].extend(ep_acts)
            dataset['rewards'].extend(ep_rews)
            dataset['next_observations'].extend(ep_next_obs)
            dataset['terminals'].extend(ep_terms)

            # 打印进度条
            progress = min(100.0, (collected_steps / target_transitions) * 100)
            elapsed = time.time() - start_time
            print(
                f"\r✅ 进度: [{progress:5.1f}%] | 已收集: {collected_steps}/{target_transitions} 步 | 有效局数: {successful_episodes} | 耗时: {elapsed:.1f}s",
                end="")

    env.close()
    print("\n" + "=" * 60)
    print("🎉 数据采集与清洗完毕！")

    # 6. 数据压缩与规整
    # 将列表转换为 NumPy 矩阵，这是离线 RL 训练 (PyTorch DataLoader) 的标准格式
    for key in dataset.keys():
        dataset[key] = np.array(dataset[key][:target_transitions], dtype=np.float32)

    # 7. 结构化存档建档
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建具有独立时间戳的子文件夹，避免堆积
    save_dir = os.path.join("data", "expert_data", f"dataset_v5_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # 将海量矩阵压缩为一个 .npz 文件
    data_path = os.path.join(save_dir, "expert_transitions.npz")
    np.savez_compressed(data_path, **dataset)

    # 生成 Metadata 记录卡片，方便未来查阅
    meta_path = os.path.join(save_dir, "metadata.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("=== Expert Dataset Metadata ===\n")
        f.write(f"Collection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source Expert Model: {model_path}\n")
        f.write(f"Total Transitions: {target_transitions}\n")
        f.write(f"Successful Episodes Included: {successful_episodes}\n")
        f.write(f"Discarded Episodes (Crashes): {discarded_episodes}\n")
        f.write(f"State Dimension: {state_dim}\n")
        f.write(f"Action Dimension: {action_dim}\n")

    print(f"💾 数据集已高度压缩并保存至: {data_path}")
    print(f"📄 数据集元信息卡片已保存至: {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    # 配置你的 v5.0 模型路径
    V5_MODEL_PATH = "outputs/models/highway-v0_SAC_20260330_135449/sac_highway_final.pth"

    # 目标收集 50,000 步高质量数据。
    # 根据你的电脑性能，这通常需要运行 2 到 5 分钟。
    collect_expert_data(
        model_path=V5_MODEL_PATH,
        env_name="highway-v0",
        target_transitions=50000
    )