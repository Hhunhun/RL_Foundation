import numpy as np
import os
from datetime import datetime

# 锁定项目根目录 (向上两级，因为此脚本被移动到 old_scripts 子文件夹)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def smart_mix_datasets(path_v5, path_v6, save_dir, target_total_steps=50000, v5_ratio=0.9):
    print("=" * 60)
    print(f"🧬 开始执行外科手术式数据蒸馏 (目标比例 v5:{v5_ratio * 100}% | 纯净 v6:{(1 - v5_ratio) * 100}%)")
    print("=" * 60)

    # 1. 加载两个数据集
    print(f"📥 加载 v5.0 保守底盘: {path_v5}")
    data_v5 = np.load(path_v5)
    print(f"📥 加载 v6.0 狂野源数据: {path_v6}")
    data_v6 = np.load(path_v6)
    keys = data_v5.files

    # 2. 计算目标步数
    v5_steps = int(target_total_steps * v5_ratio)  # 90% -> 45000 步
    v6_target_steps = target_total_steps - v5_steps  # 10% -> 5000 步

    # 3. 🛡️ 核心手术：提取 v6.0 的“安全极速流形”
    v6_actions = data_v6['actions']

    # 过滤条件：油门踩下 (a[0] > 0.2) 且 几乎没有打方向盘 (abs(a[1]) < 0.05)
    # 这确保了提取出来的动作全部是“在安全直道上猛冲”的记忆，彻底剔除危险变道
    safe_fast_mask = (v6_actions[:, 0] > 0.2) & (np.abs(v6_actions[:, 1]) <= 0.05)
    valid_v6_indices = np.where(safe_fast_mask)[0]

    print(f"🔬 从 v6.0 的 {len(v6_actions)} 步中，成功蒸馏出 {len(valid_v6_indices)} 步极其纯净的'直道冲刺'数据。")

    # 随机抽取所需的 v6 步数 (如果不够就全拿)
    actual_v6_take = min(v6_target_steps, len(valid_v6_indices))
    selected_v6_indices = np.random.choice(valid_v6_indices, size=actual_v6_take, replace=False)

    actual_v5_take = min(v5_steps, len(data_v5['observations']))
    print(f"🔪 最终配方: v5 底盘抽取 {actual_v5_take} 步，v6 纯净提速针剂抽取 {actual_v6_take} 步。")

    # 4. 执行拼接
    mixed_data = {}
    for key in keys:
        v5_slice = data_v5[key][:actual_v5_take]
        # 使用高级索引，精准提取 v6 的过滤数据
        v6_filtered_slice = data_v6[key][selected_v6_indices]
        mixed_data[key] = np.concatenate([v5_slice, v6_filtered_slice], axis=0)

    total_steps = len(mixed_data['observations'])
    print(f"✅ 融合完成！混合数据集总步数: {total_steps}")

    # 5. 彻底打乱数据 (Shuffle)
    print("🔀 正在随机打乱混合数据分布...")
    indices = np.random.permutation(total_steps)
    for key in keys:
        mixed_data[key] = mixed_data[key][indices]

    # 6. 存档
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "expert_transitions_smart_90_10.npz")
    np.savez_compressed(save_path, **mixed_data)

    print(f"💾 智能混合数据集已保存至: {save_path}")
    return save_path


if __name__ == "__main__":
    # 使用安全的绝对路径拼接
    V5_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "dataset_v5_20260404_035105", "expert_transitions.npz")
    V6_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "dataset_v6_pro_20260408_042434",
                           "expert_transitions.npz")

    # 动态生成带有智能标识的文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = os.path.join(PROJECT_ROOT, "data", "expert_data", f"dataset_smart_mixed_90_10_{timestamp}")

    if not os.path.exists(V5_PATH) or not os.path.exists(V6_PATH):
        print("❌ 错误：找不到源数据集，请检查路径是否正确！")
    else:
        smart_mix_datasets(V5_PATH, V6_PATH, SAVE_DIR, v5_ratio=0.90)