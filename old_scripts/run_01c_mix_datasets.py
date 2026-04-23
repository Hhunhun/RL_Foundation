import numpy as np
import os
import re
from datetime import datetime

# 套两层 dirname，代表获取“当前脚本所在文件夹的上一级文件夹”
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mix_datasets(path_safe, path_aggressive, save_dir, safe_ratio=0.8, mix_label="80_20"):
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
    # 给文件名显式加上混合比例标识 (如 0.8M4_0.2M3)
    save_path = os.path.join(save_dir, f"expert_transitions_mixed_{mix_label}.npz")
    np.savez_compressed(save_path, **mixed_data)

    print(f"💾 混合数据集已保存至: {save_path}")
    return save_path

def extract_model_name(path):
    """从单模型数据集的路径中自动提取模型代号，例如 M4"""
    dirname = os.path.basename(os.path.dirname(path))
    match = re.search(r'dataset_([a-zA-Z0-9]+)_', dirname)
    return match.group(1) if match else "Expert"

if __name__ == "__main__":
    # 🚨 准备工作：请先运行 run_01_collect_data.py 分别采出 M4(Mode1) 和 M3(Mode2) 的数据
    # 然后将生成的文件夹名替换到下方的路径中！
    SAFE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "merge-v0", "YOUR_M4_MODE1_DATASET_DIR", "expert_transitions.npz")
    AGGRESSIVE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "merge-v0", "YOUR_M3_MODE2_DATASET_DIR", "expert_transitions.npz")

    if not os.path.exists(SAFE_DATA_PATH) or not os.path.exists(AGGRESSIVE_DATA_PATH):
        print("❌ 错误：找不到源数据集，请确保您已经在上方填入了真实的数据集文件夹名！")
    else:
        # 自动生成类似 0.8M4_0.2M3 的极致清晰标识
        safe_ratio = 0.8
        safe_name = extract_model_name(SAFE_DATA_PATH)
        agg_name = extract_model_name(AGGRESSIVE_DATA_PATH)
        mix_label = f"{safe_ratio:.1f}{safe_name}_{1.0 - safe_ratio:.1f}{agg_name}"
        
        env_name = "merge-v0" if "merge-v0" in SAFE_DATA_PATH else "highway-v0"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        SAVE_DIR = os.path.join(PROJECT_ROOT, "data", "expert_data", env_name, f"dataset_mixed_{mix_label}_{timestamp}")

        mix_datasets(SAFE_DATA_PATH, AGGRESSIVE_DATA_PATH, SAVE_DIR, safe_ratio=safe_ratio, mix_label=mix_label)