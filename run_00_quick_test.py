import os
import sys
import traceback
import torch

# 导入核心流水线模块
from envs import create_environment
from algorithms.sac.sac_agent import SACAgent
from run_01_collect_data import collect_expert_data
from runners.train_offline_bc import train_diffusion_bc
from runners.train_online_diff import train_online_diffusion
from run_03_evaluate import evaluate_single_model

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def test_env_pipeline(env_name):
    print(f"\n{'='*80}")
    print(f"🧪 开始全面综合测试: [{env_name}] 环境")
    print(f"{'='*80}")

    try:
        # ==========================================
        # 1. 测试 SAC 初始化与保存 (Dummy Model)
        # ==========================================
        print("\n>>> [1/6] 测试 SAC 模型初始化与权重保存...")
        env = create_environment(env_name, is_eval=False, algo="sac")
        agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], action_scale=float(env.action_space.high[0]))
        
        save_dir = os.path.join(PROJECT_ROOT, "outputs", env_name, "models", "dummy_test")
        os.makedirs(save_dir, exist_ok=True)
        sac_model_path = os.path.join(save_dir, "sac_dummy.pth")
        agent.save_model(sac_model_path)
        print("✅ SAC 初始化与权重保存成功。")

        # ==========================================
        # 2. 测试 SAC 评估模块
        # ==========================================
        print("\n>>> [2/6] 测试统一评估模块 (SAC 模式)...")
        eval_dir = os.path.join(PROJECT_ROOT, "outputs", env_name, "eval_results", "dummy_test")
        evaluate_single_model(
            model_id=f"SAC_{env_name}_Test",
            model_path=sac_model_path,
            display_label="SAC 冒烟测试",
            env_name=env_name,
            eval_run_dir=eval_dir,
            num_episodes=1,     # 极速测1局
            record_video=False,  # 不测视频 IO 以节约时间
            max_steps_per_episode=10 # 增加超时保险
        )
        print("✅ SAC 评估模块运行成功。")

        # ==========================================
        # 3. 测试专家数据采集模块
        # ==========================================
        print("\n>>> [3/6] 测试专家数据采集模块...")
        data_path = collect_expert_data(
            model_path=sac_model_path,
            env_name=env_name,
            target_transitions=5,  # 极速测试，仅采集5步
            mode=1,
            test_mode=True, # 开启测试模式，不过滤碰撞，确保快速完成
            max_steps_per_episode=10 # 增加超时保险
        )
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError("数据采集失败，未生成 npz 数据文件。")
        print("✅ 数据采集模块运行成功。")

        # ==========================================
        # 4. 测试 Diff-SAC 离线 BC 预训练
        # ==========================================
        print("\n>>> [4/6] 测试 Diff-SAC 离线 BC 预训练...")
        bc_model_path = train_diffusion_bc(
            data_path=data_path,
            env_name=env_name,
            num_epochs=1,    # 仅测1轮
            batch_size=1,   # 极小 batch，确保即使只有少量数据也能形成批次
            learning_rate=3e-4
        )
        print("✅ Diff-SAC 离线 BC 运行成功。")

        # ==========================================
        # 5. 测试 Diff-SAC 在线微调
        # ==========================================
        print("\n>>> [5/6] 测试 Diff-SAC 在线 RL 微调...")
        train_online_diffusion(
            pretrained_actor_path=bc_model_path,
            expert_data_path=data_path,
            env_name=env_name,
            max_episodes=1,  # 仅测1局
            batch_size=1,
            q_weight=0.01,
            lr=3e-4,
            max_steps_per_episode=10 # 增加超时保险，防止随机模型卡住
        )
        print("✅ Diff-SAC 在线微调运行成功。")

        # ==========================================
        # 6. 测试 Diff-SAC 评估模块 (含归一化器)
        # ==========================================
        print("\n>>> [6/6] 测试统一评估模块 (Diff-SAC 模式)...")
        evaluate_single_model(
            model_id=f"Diff_{env_name}_Test",
            model_path=bc_model_path,
            display_label="Diff-SAC 冒烟测试",
            env_name=env_name,
            eval_run_dir=eval_dir,
            num_episodes=1,
            record_video=False,
            expert_data_path=data_path,  # 传入数据建立归一化器基准
            max_steps_per_episode=10 # 增加超时保险
        )
        print("✅ Diff-SAC 评估模块运行成功。")

        return True, None

    except Exception as e:
        error_msg = traceback.format_exc()
        return False, error_msg


if __name__ == "__main__":
    envs_to_test = ["highway-v0", "merge-v0"]
    results = {}

    print("🚀 启动自动化全链路测试系统...")
    for env_name in envs_to_test:
        success, err = test_env_pipeline(env_name)
        results[env_name] = (success, err)

    print(f"\n{'='*80}")
    print("🏁 全链路快速冒烟测试总结")
    print(f"{'='*80}")
    
    all_passed = True
    for env_name, (success, err) in results.items():
        if success:
            print(f"🟢 {env_name.ljust(12)} : ALL PASSED")
        else:
            print(f"🔴 {env_name.ljust(12)} : FAILED")
            print(f"\n[错误追踪 - {env_name}]\n{err}")
            all_passed = False

    if all_passed:
        print("\n🎉 恭喜！Highway 与 Merge 双环境下的 SAC 与 Diff-SAC 训练及评估流水线均完美畅通！架构重构非常成功。")
    else:
        print("\n⚠️ 发现错误！请查阅上方错误追踪进行修复。")