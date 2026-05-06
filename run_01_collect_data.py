"""
阶段一：专家轨迹数据采集流水线 (Expert Data Collection)

此模块负责为后续的 Diffusion 预训练提供高质量的“教材”。
它会加载一个已经训练好的、具备高超驾驶技术的基准 SAC 模型（专家），让它在环境中自动驾驶。

目前支持两种采集模式：
[Mode 1] 保守底座采集：包含概率性行为抖动，过滤所有碰撞局，用于构建安全先验。
[Mode 2] 极速局采集：关闭抖动，过滤碰撞且强制要求回合均速 > 22.0 m/s，用于拓展流形上限。
"""

import os
import time
import numpy as np
import re
from datetime import datetime
from gymnasium.wrappers import RecordVideo
import warnings

# 锁定项目根目录以确保数据保存路径绝对安全
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 忽略 Pygame 和某些底层库产生的烦人警告，保持控制台输出整洁
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from algorithms.sac.sac_agent import SACAgent
from envs import create_environment


def collect_expert_data(model_path, env_name="highway-v0", target_transitions=50000, mode=1, test_mode=False, max_steps_per_episode=1000, env_config=None):
    print("\n" + "=" * 60)
    print("🚀 [阶段一] 开始专家轨迹数据采集 (Expert Data Collection)")
    print(f"📦 目标采集量: {target_transitions} 步 | 环境: {env_name} | 当前模式: Mode {mode} {'(测试模式，不过滤碰撞)' if test_mode else ''}")
    print(f"🧠 加载权重: {model_path}")
    if env_config: print(f"⚙️ 环境自定义配置: {env_config}")
    print("=" * 60)

    env = create_environment(env_name, is_eval=True, algo="sac", env_config=env_config)
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
        ep_steps = 0

        while True:
            # 1. 获取专家基准动作 (evaluate=True 表示关闭 SAC 的探索噪声，输出确定性的绝对最优解)
            action = agent.select_action(state, evaluate=True)

            # 2. 🚨 概率性注入行为抖动 (仅 Mode 1 启用) 🚨
            # 学术级数据增强操作：50% 概率保持绝对纯净，50% 概率加入微小的正态分布噪声。
            # Mode 2 需要纯粹的极速发挥，因此关闭抖动。
            if mode == 1 and np.random.rand() < 0.5:
                if env_name == "merge-v0":
                    # 🐛 TTC 极度敏感修复: Merge 环境完全关闭步进式抖动！
                    # 哪怕是微小的纵向干扰，累计后也会摧毁 M8 精准的避让微操导致 100% 追尾。
                    # 数据多样性将由环境中随机初始化的背景车辆来自然保证。
                    pass
                else:
                    noise = np.random.normal(0, 0.05, size=action_dim)
                    action = np.clip(action + noise, -1.0, 1.0)

            # 将动作输入物理环境，获取下一步的反馈
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_steps += 1

            # 🚨 [核心修复] 强制截断：避免车辆一直开到地图尽头掉进虚空，被误判为“出界/撞车”
            if env_name == "merge-v0" and ep_steps >= 100:
                truncated = True
            elif env_name == "racetrack-v0" and ep_steps >= 500:
                truncated = True

            # 将这一步的数据暂存进当前局的缓存列表中
            ep_obs.append(state)
            ep_acts.append(action)
            ep_rews.append(reward)
            ep_next_obs.append(next_state)
            ep_terms.append(terminated)
            ep_speeds.append(info.get("ego_speed_vx", 0.0))

            state = next_state

            # 如果撞车/出轨(terminated) 或达到最大步数(truncated)，当前局结束
            if terminated or truncated or ep_steps >= max_steps_per_episode:
                if terminated:
                    # 🚨 [核心修复] 区分真假车祸，保护“极速超车”的神仙局
                    # Merge 道路极短，车辆如果加速超车，会在 100 步内冲出地图纵向尽头(out_of_road)，
                    # 此时底层会报 terminated=True。但这绝不是车祸，而是最高效的完赛！
                    if env_name == "merge-v0":
                        try:
                            actual_crash = getattr(env.unwrapped.vehicle, "crashed", False)
                            is_sideways = abs(env.unwrapped.vehicle.heading) > 0.4
                            is_not_on_road = not getattr(env.unwrapped.vehicle, "on_road", True)
                            crashed = actual_crash or is_sideways or is_not_on_road
                        except Exception:
                            crashed = True
                    elif env_name == "racetrack-v0":
                        try:
                            actual_crash = getattr(env.unwrapped.vehicle, "crashed", False)
                            is_not_on_road = not getattr(env.unwrapped.vehicle, "on_road", True)
                            crashed = actual_crash or is_not_on_road or info.get("crashed", False)
                        except Exception:
                            crashed = True
                    else:
                        try:
                            crashed = getattr(env.unwrapped.vehicle, "crashed", False)
                        except Exception:
                            crashed = True # 兜底逻辑
                break

        # 🚨 回合级淘汰机制 (Episode-level Filtering) 🚨
        mean_speed = np.mean(ep_speeds)
        accept_episode = False

        if test_mode:
            # 测试模式下，不进行任何过滤，直接接受所有数据
            accept_episode = True
            reason = "测试模式"
        else:
            if mode == 1:
                # Mode 1 逻辑：只要没撞车就收下
                if not crashed:
                    accept_episode = True
                    reason = "安全完赛"
                else:
                    reason = "发生碰撞/出界"
            elif mode == 2:
                # Mode 2 逻辑：神仙局必须同时满足【不撞车】且【均速 > 22.0】
                if env_name == "merge-v0":
                    if not crashed and mean_speed > 19.5: # 🚨 拔高门槛：专门榨取 M3 (激进专家) 的极限微操破局数据
                        accept_episode = True
                        reason = f"激进破局(均速:{mean_speed:.1f})"
                    else:
                        reason = f"撞车:{crashed}, 均速:{mean_speed:.2f}m/s 未达标(需>19.5)"
                elif env_name == "racetrack-v0":
                    if not crashed and mean_speed > 20.0:
                        accept_episode = True
                        reason = f"极限切弯(均速:{mean_speed:.1f})"
                    else:
                        reason = f"撞车:{crashed}, 均速:{mean_speed:.2f}m/s 未达标(需>20.0)"
                else:
                    if not crashed and mean_speed > 22.0:
                        accept_episode = True
                        reason = f"神仙局(均速:{mean_speed:.1f})"
                    else:
                        reason = f"撞车:{crashed}, 均速:{mean_speed:.2f}m/s 未达标(需>22.0)"

        # 执行数据并入或丢弃
        if accept_episode:
            successful_episodes += 1
            collected_steps += len(ep_obs)
            dataset['observations'].extend(ep_obs)
            dataset['actions'].extend(ep_acts)
            dataset['rewards'].extend(ep_rews)
            dataset['next_observations'].extend(ep_next_obs)
            dataset['terminals'].extend(ep_terms)
            status_msg = f"✅ 收录! ({reason})"
        else:
            discarded_episodes += 1
            status_msg = f"⚠️ 丢弃 ({reason})"

        # 📊 优化终端输出：实时更新全局看板，将成功和失败的信息汇总到一行
        progress = min(100.0, (collected_steps / target_transitions) * 100)
        print(f"\r📊 进度: {progress:5.1f}% ({collected_steps}/{target_transitions}步) | 收: {successful_episodes} 局 | 弃: {discarded_episodes} 局 | 最新: {status_msg}" + " " * 10, end="")

    env.close()

    # 将数据截断到精确的 target_transitions 数量，并转化为 numpy 数组
    for key in dataset.keys():
        dataset[key] = np.array(dataset[key][:target_transitions], dtype=np.float32)

    # 根据采集模式和当前时间生成专属的存档目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 自动解析专家模型的短代号 (例如从 SAC_M4_Safety_First 提取出 M4)
    model_dir_name = os.path.basename(os.path.dirname(model_path))
    # 匹配 SAC_ 后面紧跟的字母+数字组合
    short_match = re.search(r'SAC_([a-zA-Z0-9]+)_', model_dir_name)
    if short_match:
        source_name = short_match.group(1)
    else:
        source_name = "Expert"
        
    dataset_prefix = f"dataset_{source_name}_mode{mode}_"
    save_dir = os.path.join(PROJECT_ROOT, "data", "expert_data", env_name, f"{dataset_prefix}{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(save_dir, "expert_transitions.npz")

    # 采用高压缩比格式保存 .npz 文件，大幅节省硬盘空间
    np.savez_compressed(data_path, **dataset)

    print(f"\n\n💾 数据集已完美保存至: {data_path}")

    # 🎬 新增：在数据集文件夹内同时渲染并保存几局专家视频，便于人类直接观察
    if env_name in ["merge-v0", "racetrack-v0"]:
        print(f"🎬 正在为您录制 3 局专家实况录像，以供直观检查数据质量...")
        video_dir = os.path.join(save_dir, "videos")
        rec_env = create_environment(env_name, is_eval=True, algo="sac", env_config=env_config)
        rec_env = RecordVideo(rec_env, video_folder=video_dir, name_prefix=f"expert_demo_mode{mode}")
        
        for _ in range(3):
            obs, _ = rec_env.reset()
            rec_steps = 0
            while True:
                act = agent.select_action(obs, evaluate=True)
                obs, _, term, trunc, _ = rec_env.step(act)
                rec_steps += 1
                if env_name == "merge-v0" and rec_steps >= 100:
                    trunc = True
                elif rec_steps >= 500 and env_name == "racetrack-v0":
                    trunc = True
                if term or trunc:
                    break
        rec_env.close()
        print(f"✅ 录像已保存至子目录: {video_dir}")

    print(f"⏱️ 耗时: {(time.time() - start_time) / 60:.2f} 分钟")

    # 🚨 返回数据路径，供主控流水线 (02_train_pipeline) 直接读取传递给下一阶段
    return data_path


def extract_model_name(path):
    """从单模型数据集的路径中自动提取模型代号，例如 R05"""
    dirname = os.path.basename(os.path.dirname(path))
    match = re.search(r'dataset_([a-zA-Z0-9]+)_', dirname)
    return match.group(1) if match else "Expert"


def mix_datasets(path_safe, path_aggressive, save_dir, target_total_steps=50000, safe_ratio=0.8):
    """标准混合：按比例直接拼接并打乱两个数据集"""
    print("\n" + "=" * 60)
    print(f"🧬 [阶段一] 开始标准数据集融合 (保守:{safe_ratio * 100}% | 激进:{(1 - safe_ratio) * 100}%)")
    print("=" * 60)

    data_safe = np.load(path_safe)
    data_aggressive = np.load(path_aggressive)
    keys = data_safe.files

    safe_steps = int(target_total_steps * safe_ratio)
    aggressive_steps = target_total_steps - safe_steps

    actual_safe_steps = min(safe_steps, len(data_safe['observations']))
    actual_aggressive_steps = min(aggressive_steps, len(data_aggressive['observations']))
    print(f"🔪 切片方案: 抽取保守数据 {actual_safe_steps} 步，抽取激进数据 {actual_aggressive_steps} 步")

    mixed_data = {}
    for key in keys:
        safe_slice = data_safe[key][:actual_safe_steps]
        aggressive_slice = data_aggressive[key][:actual_aggressive_steps]
        mixed_data[key] = np.concatenate([safe_slice, aggressive_slice], axis=0)

    total_steps = len(mixed_data['observations'])
    print("🔀 正在随机打乱混合数据分布...")
    indices = np.random.permutation(total_steps)
    for key in keys:
        mixed_data[key] = mixed_data[key][indices]

    os.makedirs(save_dir, exist_ok=True)
    safe_name = extract_model_name(path_safe)
    agg_name = extract_model_name(path_aggressive)
    mix_label = f"{safe_ratio:.1f}{safe_name}_{1.0 - safe_ratio:.1f}{agg_name}"
    save_path = os.path.join(save_dir, f"expert_transitions_mixed_{mix_label}.npz")
    np.savez_compressed(save_path, **mixed_data)
    print(f"✅ 标准混合完成！已保存至: {save_path}")
    return save_path


def smart_mix_datasets(path_safe, path_aggressive, save_dir, env_name="highway-v0", target_total_steps=50000, safe_ratio=0.8):
    """智能混合：对激进数据进行条件过滤（保留直道极速），再与保守数据拼接打乱"""
    print("\n" + "=" * 60)
    print(f"🧬 [阶段一] 开始智能过滤式数据融合 (保守:{safe_ratio * 100}% | 激进(过滤后):{(1 - safe_ratio) * 100}%)")
    print("=" * 60)

    data_safe = np.load(path_safe)
    data_aggressive = np.load(path_aggressive)
    keys = data_safe.files

    safe_steps = int(target_total_steps * safe_ratio)
    agg_target_steps = target_total_steps - safe_steps

    agg_actions = data_aggressive['actions']
    
    # 🚨 核心路由：根据不同的物理环境，采用截然不同的流形过滤（Manifold Filtering）规则
    if env_name == "highway-v0":
        # 规则：油门踩下且方向盘几乎不动 (去除变道，提取纯净的直道冲刺)
        safe_fast_mask = (agg_actions[:, 0] > 0.2) & (np.abs(agg_actions[:, 1]) <= 0.05)
        desc = "直道冲刺"
    elif env_name == "merge-v0":
        # 规则：保持给油，且转向幅度在合理区间 (提取果断的加速汇入与并道动作)
        safe_fast_mask = (agg_actions[:, 0] > 0.0) & (np.abs(agg_actions[:, 1]) <= 0.3)
        desc = "加速汇入"
    elif env_name == "racetrack-v0":
        # 规则：绝不重踩刹车，完全释放方向盘限制 (赛道极速过弯必须允许满打方向盘)
        safe_fast_mask = (agg_actions[:, 0] >= -0.2)
        desc = "极限切弯"
        
    valid_agg_indices = np.where(safe_fast_mask)[0]
    print(f"🔬 从激进数据的 {len(agg_actions)} 步中，成功蒸馏出 {len(valid_agg_indices)} 步纯净的'{desc}'数据。")

    actual_agg_take = min(agg_target_steps, len(valid_agg_indices))
    selected_agg_indices = np.random.choice(valid_agg_indices, size=actual_agg_take, replace=False)
    actual_safe_take = min(safe_steps, len(data_safe['observations']))
    print(f"🔪 最终配方: 保守底盘 {actual_safe_take} 步，极速纯净针剂 {actual_agg_take} 步。")

    mixed_data = {}
    for key in keys:
        mixed_data[key] = np.concatenate([data_safe[key][:actual_safe_take], data_aggressive[key][selected_agg_indices]], axis=0)

    total_steps = len(mixed_data['observations'])
    print("🔀 正在随机打乱混合数据分布...")
    indices = np.random.permutation(total_steps)
    for key in keys:
        mixed_data[key] = mixed_data[key][indices]

    os.makedirs(save_dir, exist_ok=True)
    safe_name = extract_model_name(path_safe)
    agg_name = extract_model_name(path_aggressive)
    mix_label = f"smart_{safe_ratio:.1f}{safe_name}_{1.0 - safe_ratio:.1f}{agg_name}"
    save_path = os.path.join(save_dir, f"expert_transitions_mixed_{mix_label}.npz")
    np.savez_compressed(save_path, **mixed_data)
    print(f"✅ 智能混合完成！已保存至: {save_path}")
    return save_path


if __name__ == "__main__":
    # ==========================================
    # 终端交互控制台
    # ==========================================
    print("🤖 数据管线控制台 (Data Pipeline)")
    print("==========================================")
    print("--- 仿真采集 (Live Collection) ---")
    print("[1] 基础模式 (Mode 1): 使用保守安全模型，含行为抖动，仅过滤碰撞局。(适用构建安全保底)")
    print("[2] 极速模式 (Mode 2): 使用激进破局模型，关闭抖动，严格过滤速度阈值。(适用探寻上限)")
    print("--- 离线混合 (Offline Mixing) ---")
    print("[3] 标准混合 (Standard Mix): 将已有稳健和激进数据集按 80:20 直接拼接打乱。")
    print("[4] 智能混合 (Smart Mix): 从激进数据中严格过滤出直道加速流形，再进行混合。")
    print("==========================================")

    print("[H] Highway 环境 (highway-v0)")
    print("[M] Merge 环境 (merge-v0)")
    print("[R] Racetrack 环境 (racetrack-v0)")
    env_choice = input("👉 请选择采集环境 (H, M 或 R，默认 H): ").strip().upper()
    if env_choice == 'M':
        target_env = "merge-v0"
    elif env_choice == 'R':
        target_env = "racetrack-v0"
    else:
        target_env = "highway-v0"
    print("==========================================")

    choice = input("👉 请选择执行操作 (1/2/3/4，默认 1): ").strip()
    if choice not in ['1', '2', '3', '4']: choice = '1'

    # 🚨 环境资源配置清单 🚨
    if target_env == "merge-v0":
        SAFE_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "merge-v0", "models", "SAC_M4_Safety_First_20260420_170911", "sac_merge_final.pth")
        SAFE_ENV_CONFIG = {"reward_speed_range": [15, 25]}
        AGGRESSIVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "merge-v0", "models", "SAC_M3_Aggressive_Gap_Finding_20260420_162217", "sac_merge_final.pth")
        AGGRESSIVE_ENV_CONFIG = {"reward_speed_range": [20, 30]}
        # 离线混合依赖的已有数据集路径
        SAFE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "merge-v0", "YOUR_SAFE_DATASET_HERE", "expert_transitions.npz")
        AGGRESSIVE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "merge-v0", "YOUR_AGG_DATASET_HERE", "expert_transitions.npz")

    elif target_env == "racetrack-v0":
        SAFE_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "racetrack-v0", "models", "SAC_R05_SAC_Smooth_Racing_20260505_131614", "sac_racetrack_final.pth")
        SAFE_ENV_CONFIG = {"reward_speed_range": [15, 25]}
        AGGRESSIVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "racetrack-v0", "models", "SAC_R01_SAC_Baseline_20260505_033212", "sac_racetrack_final.pth")
        AGGRESSIVE_ENV_CONFIG = {"reward_speed_range": [15, 30]} 
        # 离线混合依赖的已有数据集路径 (需替换为您真实采出的文件夹名)
        SAFE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "racetrack-v0", "dataset_R05_mode1_20260506_011817", "expert_transitions.npz")
        AGGRESSIVE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "racetrack-v0", "dataset_R01_mode2_20260506_141859", "expert_transitions.npz")

    else:
        SAFE_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "highway-v0", "models", "SAC_20260330_135449", "sac_highway_final.pth")
        SAFE_ENV_CONFIG = None
        AGGRESSIVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "highway-v0", "models", "SAC_20260330_213300", "sac_highway_final.pth")
        AGGRESSIVE_ENV_CONFIG = None
        SAFE_DATA_PATH = ""
        AGGRESSIVE_DATA_PATH = ""

    timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    TARGET_STEPS = 20000 if target_env == "merge-v0" else 50000

    # --- 路由执行逻辑 ---
    if choice in ['1', '2']:
        selected_mode = int(choice)
        selected_model = AGGRESSIVE_MODEL_PATH if choice == '2' else SAFE_MODEL_PATH
        selected_env_config = AGGRESSIVE_ENV_CONFIG if choice == '2' else SAFE_ENV_CONFIG

        if os.path.exists(selected_model):
            collect_expert_data(model_path=selected_model, env_name=target_env, target_transitions=TARGET_STEPS, mode=selected_mode, env_config=selected_env_config)
        else:
            print(f"\n❌ 找不到指定的权重文件！请确保路径正确: {selected_model}")
            
    elif choice in ['3', '4']:
        if not os.path.exists(SAFE_DATA_PATH) or not os.path.exists(AGGRESSIVE_DATA_PATH):
            print("\n❌ 找不到源数据集！请打开脚本末尾，将 SAFE_DATA_PATH 或 AGGRESSIVE_DATA_PATH 替换为您实际生成的数据文件夹。")
        else:
            safe_ratio = 0.8
            safe_name = extract_model_name(SAFE_DATA_PATH)
            agg_name = extract_model_name(AGGRESSIVE_DATA_PATH)
            
            if choice == '3':
                mix_label = f"{safe_ratio:.1f}{safe_name}_{1.0 - safe_ratio:.1f}{agg_name}"
            else:
                mix_label = f"smart_{safe_ratio:.1f}{safe_name}_{1.0 - safe_ratio:.1f}{agg_name}"
                
            save_dir = os.path.join(PROJECT_ROOT, "data", "expert_data", target_env, f"dataset_mixed_{mix_label}_{timestamp_now}")
            
            if choice == '3':
                mix_datasets(SAFE_DATA_PATH, AGGRESSIVE_DATA_PATH, save_dir, target_total_steps=TARGET_STEPS, safe_ratio=safe_ratio)
            else:
                smart_mix_datasets(SAFE_DATA_PATH, AGGRESSIVE_DATA_PATH, save_dir, env_name=target_env, target_total_steps=TARGET_STEPS, safe_ratio=safe_ratio)