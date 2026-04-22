import numpy as np
import os
from datetime import datetime

# 套两层 dirname，代表获取“当前脚本所在文件夹的上一级文件夹”
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mix_datasets(path_safe, path_aggressive, save_dir, safe_ratio=0.8):
    print("=" * 60)
    print(f"🧬 开始进行数据集黄金融合 (目标比例 保守:{safe_ratio * 100}% | 激进:{(1 - safe_ratio) * 100}%)")
    print("=" * 60)

    # 1. 加载两个数据集
    print(f"📥 加载保守安全底座: {path_safe}")
    data_safe = np.load(path_safe)
    print(f"📥 加载激进破局数据: {path_aggressive}")
    data_aggressive = np.load(path_aggressive)

    # 获取键名
    keys = data_safe.files

    # 2. 精确计算切片步数 (总目标保持 50000 步，确保和之前的实验规模一致)
    total_target_steps = 50000
    safe_steps = int(total_target_steps * safe_ratio)  # 80% -> 40000 步
    aggressive_steps = total_target_steps - safe_steps  # 20% -> 10000 步

    # 容错保护：确保原数据集足够长
    actual_safe_steps = min(safe_steps, len(data_safe['observations']))
    actual_aggressive_steps = min(aggressive_steps, len(data_aggressive['observations']))

    print(f"🔪 切片方案: 抽取保守数据 {actual_safe_steps} 步，抽取激进数据 {actual_aggressive_steps} 步")

    # 3. 执行切片与拼接
    mixed_data = {}
    for key in keys:
        # 🚨 核心修改点：分别截取指定长度后再拼接
        safe_slice = data_safe[key][:actual_safe_steps]
        aggressive_slice = data_aggressive[key][:actual_aggressive_steps]
        mixed_data[key] = np.concatenate([safe_slice, aggressive_slice], axis=0)

    total_steps = len(mixed_data['observations'])
    print(f"✅ 融合完成！混合数据集总步数: {total_steps}")

    # 4. 打乱数据 (Shuffle) - 极其重要，防止网络在 epoch 前期只学保守，后期只学激进
    print("🔀 正在随机打乱混合数据分布...")
    indices = np.random.permutation(total_steps)
    for key in keys:
        mixed_data[key] = mixed_data[key][indices]

    # 5. 保存混合后的数据集
    os.makedirs(save_dir, exist_ok=True)
    # 给文件名显式加上 80_20 标识
    save_path = os.path.join(save_dir, "expert_transitions_mixed_80_20.npz")
    np.savez_compressed(save_path, **mixed_data)

    print(f"💾 混合数据集已保存至: {save_path}")
    return save_path


if __name__ == "__main__":
    # 使用安全的绝对路径拼接
    SAFE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "highway-v0", "dataset_v5_20260404_035105", "expert_transitions.npz")
    AGGRESSIVE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "highway-v0", "dataset_v6_pro_20260408_042434", "expert_transitions.npz")

    # 🚨 动态生成带有当前时间戳的文件夹，防止覆盖之前 50:50 的数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = os.path.join(PROJECT_ROOT, "data", "expert_data", "highway-v0", f"dataset_mixed_80_20_{timestamp}")

    if not os.path.exists(SAFE_DATA_PATH) or not os.path.exists(AGGRESSIVE_DATA_PATH):
        print("❌ 错误：找不到源数据集，请检查路径是否正确！")
    else:
        mix_datasets(SAFE_DATA_PATH, AGGRESSIVE_DATA_PATH, SAVE_DIR, safe_ratio=0.8)