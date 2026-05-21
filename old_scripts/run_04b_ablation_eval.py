"""
自动化消融实验评估脚本 (Diffusion Steps Ablation Evaluation)
用途：加载真实的 Diff-SAC 模型，在不同采样步数下运行环境，
      精确测量真实的平均奖励与单步网络推理耗时，为消融图表提供真实物理数据。
"""

import os
import sys
import time
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 动态追加根目录环境变量
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs import create_environment
from algorithms.diffusion_sac.diff_sac_agent import DiffSACAgent
from core.offline_buffer import MixedReplayBuffer

def run_ablation_eval():
    # ==========================================
    # 1. 实验配置区 (批量自动化测试矩阵)
    # ==========================================
    STEPS_TO_TEST = [1, 3, 5, 8, 10, 15, 20]
    EPISODES_PER_STEP = 100  # 🌟 通宵挂机配置：100局足以提供极高置信度的统计分布
    
    # 自动加载三大环境中表现最佳的混合专家 Diff-SAC (DM06 / DR06 / DH06)
    EXPERIMENTS = {
        "Merge (匝道汇入)": {
            "env_name": "merge-v0",
            "model_path": os.path.join(PROJECT_ROOT, "outputs", "merge-v0", "models", "DiffSAC_DM06_Mixed_Micro_Q_20260513_175844", "online_finetune", "diff_sac_final.pth"),
            "expert_data_path": os.path.join(PROJECT_ROOT, "data", "expert_data", "merge-v0", "dataset_mixed_0.8M04_0.2M03_20260513_161828", "expert_transitions_mixed_0.8M04_0.2M03.npz")
        },
        "Racetrack (极限赛道)": {
            "env_name": "racetrack-v0",
            "model_path": os.path.join(PROJECT_ROOT, "outputs", "racetrack-v0", "models", "DiffSAC_DR06_Mixed_Micro_Q_20260510_191112", "online_finetune", "diff_sac_final.pth"),
            "expert_data_path": os.path.join(PROJECT_ROOT, "data", "expert_data", "racetrack-v0", "dataset_mixed_0.8R05_0.2R01_20260506_142446", "expert_transitions_mixed_0.8R05_0.2R01.npz")
        },
        "Highway (高速巡航)": {
            "env_name": "highway-v0",
            "model_path": os.path.join(PROJECT_ROOT, "outputs", "highway-v0", "models", "DiffSAC_DH06_Mixed_Micro_Q_20260515_100237", "online_finetune", "diff_sac_final.pth"),
            "expert_data_path": os.path.join(PROJECT_ROOT, "data", "expert_data", "highway-v0", "dataset_mixed_0.8H02_0.2H01_20260513_204225", "expert_transitions_mixed_0.8H02_0.2H01.npz")
        }
    }

    print("=" * 60)
    print("🚀 开始运行扩散步数消融 [批量] 自动化测试")
    print(f"待测实验数: {len(EXPERIMENTS)} 个环境 | 每步测试局数: {EPISODES_PER_STEP} 局")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 当前推理设备: {device.upper()}")

    all_results = {}

    for exp_name, cfg in EXPERIMENTS.items():
        TARGET_ENV = cfg["env_name"]
        MODEL_PATH = cfg["model_path"]
        EXPERT_DATA_PATH = cfg["expert_data_path"]

        print(f"\n{'='*50}")
        print(f"🧪 正在测试实验队列: {exp_name}")
        print(f"{'='*50}")

        if not os.path.exists(MODEL_PATH) or not os.path.exists(EXPERT_DATA_PATH):
            print(f"⚠️ 找不到 {TARGET_ENV} 的模型权重或数据，自动跳过本环境！")
            continue

        # ------------------------------------------
        # 2. 环境与模型初始化
        # ------------------------------------------
        env = create_environment(TARGET_ENV, is_eval=True, algo="diff")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        print("🧠 正在挂载模型与数据归一化器...")
        buffer = MixedReplayBuffer(expert_data_path=EXPERT_DATA_PATH, max_online_size=10, device=device)
        agent = DiffSACAgent(state_dim, action_dim, device=device)
        
        try:
            agent.load_pretrained_actor(MODEL_PATH)
            agent.ema_actor.model.load_state_dict(agent.actor.state_dict())
            print("✅ 模型加载并修复同步完成！\n")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            env.close()
            continue

        # ------------------------------------------
        # 3. 开始执行该环境下的消融循环
        # ------------------------------------------
        final_rewards = []
        final_times = []

        for step_val in STEPS_TO_TEST:
            print(f"▶ 正在测试 Diffusion Steps = {step_val} ...")
            step_rewards = []
            step_inference_times = []

            for ep in range(EPISODES_PER_STEP):
                state, _ = env.reset(seed=42 + ep)  # 固定种子序列保证不同 Step 面临的考题严格一致
                ep_reward = 0
                ep_steps = 0

                while True:
                    norm_state = buffer.state_normalizer.normalize(state)
                    
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    
                    norm_action = agent.select_action(norm_state, sample_steps=step_val, explore=False)
                    
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    
                    step_inference_times.append((t1 - t0) * 1000) # 转换为 ms

                    action = buffer.action_normalizer.unnormalize(norm_action)
                    action = np.clip(action, -1.0, 1.0)
                    
                    next_state, reward, terminated, truncated, info = env.step(action)
                    ep_reward += reward
                    ep_steps += 1
                    
                    if TARGET_ENV == "merge-v0" and ep_steps >= 100:
                        truncated = True
                    elif TARGET_ENV == "racetrack-v0" and ep_steps >= 500:
                        truncated = True

                    state = next_state
                    if terminated or truncated:
                        break
                
                step_rewards.append(ep_reward)

            avg_reward = np.mean(step_rewards)
            avg_time = np.mean(step_inference_times[10:] if len(step_inference_times) > 20 else step_inference_times)
            
            # 使用 float() 去除 np.float64 类型外壳，输出更干净的数组
            final_rewards.append(round(float(avg_reward), 2))
            final_times.append(round(float(avg_time), 2))
            
            print(f"  └─ 🎯 总结: 平均奖励 {avg_reward:.2f} | 纯推理耗时 {avg_time:.2f} ms\n")

        env.close()
        all_results[exp_name] = {"rewards": final_rewards, "times": final_times}

    # ==========================================
    # 4. 打印全部汇总结果（可直接复制）
    # ==========================================
    print("=" * 60)
    print("🎉 全部矩阵测试完成！请直接复制以下纯净数组：")
    print("=" * 60)
    print(f"steps = {STEPS_TO_TEST}\n")
    for exp_name, res in all_results.items():
        print(f"### {exp_name} ###")
        print(f"rewards = {res['rewards']}")
        print(f"times = {res['times']}\n")
    print("=" * 60)

if __name__ == "__main__":
    run_ablation_eval()