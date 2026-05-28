"""
Module: SAC Baseline Training Pipeline for Merge (匝道汇入基线训练管线)
Description:
    本模块是针对 merge-v0 高动态非合作博弈场景的 SAC 算法基线 (Baseline) 训练中枢。
    相比于高速巡航，匝道汇入极易陷入“原地等待”的局部最优解。本管线深度集成了基于惩罚系数动态放缩的
    课程学习 (Curriculum Learning) 引擎，旨在为后续生成式模型提供兼顾突破性与安全性的先验策略底座。

Key Features:
    - Wrapper Penetration Scheduling: 穿透多层 Gym Wrapper 架构，动态实施惩罚系数的线性退火，实现从“莽撞探索”到“平滑规控”的柔性过渡。
    - Policy Annealing: 训练中后期的学习率平滑衰减，防止 SAC 算法在收敛后期因过频探索引发的策略崩塌 (Policy Collapse)。
    - Heuristic VRAM Garbage Collection: 基于连续生存期探测的主动显存碎片回收机制，
      阻断高频撞车引发的计算图极速堆积与 OOM (Out of Memory) 灾难。
"""

import os
import sys
import gc
import numpy as np
import torch
from datetime import datetime

# 动态将项目根目录添加到包搜索路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from core.replay_buffer import ReplayBuffer
from algorithms.sac.sac_agent import SACAgent
from utils.logger import Logger
from gymnasium.wrappers import RecordVideo
from envs.merge_wrapper import create_merge_env

# =====================================================================
# 🛠️ 核心辅助函数：穿透 Wrapper 层级动态修改环境惩罚权重 (课程学习用)
# =====================================================================
def update_env_penalties(env, jerk, steering):
    """
    环境约束穿透调度算子 (Environment Constraint Penetration Scheduler)。
    利用 Python 的动态属性反射机制，逐层剥离 (Peeling) 标准化 Gym 环境嵌套，
    精准锚定底层控制封装器 (MergeAVControlWrapper)，实时注入随训练阶段演进的动力学惩罚系数。
    """
    curr_env = env
    # 逐层剥开嵌套的 gym.Wrapper，直到锚定目标控制包装器
    while hasattr(curr_env, 'env'):
        if hasattr(curr_env, 'jerk_weight'):
            curr_env.jerk_weight = jerk
            curr_env.steering_weight = steering
            return
        curr_env = curr_env.env

# =====================================================================
# 🛠️ 核心辅助函数：动态修改 Agent 学习率 (策略固化用)
# =====================================================================
def set_agent_lr(agent, lr):
    """
    学习率动态退火算子 (Learning Rate Annealing Operator)。
    在强化学习后期强行收缩参数更新步长，促使策略向局部最优解深层收敛，
    阻断因过量的信息熵注入而导致的无意义震荡与分数倒挂。
    """
    if hasattr(agent, 'set_lr'):
        agent.set_lr(lr)
    else:
        # 如果 SACAgent 内部没有写 set_lr 方法，强行修改优化器参数兜底
        if hasattr(agent, 'actor_optimizer'):
            for p in agent.actor_optimizer.param_groups: p['lr'] = lr
            for p in agent.critic_optimizer.param_groups: p['lr'] = lr
            for p in agent.alpha_optimizer.param_groups: p['lr'] = lr


def run_single_experiment(config):
    """
    单体实验流水线引擎。
    涵盖环境构建、课程学习调度、稳定收敛及评估录制的完整马尔可夫决策生命周期。
    """
    env_name = 'merge-v0'
    
    print("\n" + "="*60)
    print(f"🚀 Starting Merge Experiment: {config['name']}")
    print("="*60)

    # 1. 结构化创建环境
    env = create_merge_env(
        env_name=env_name, 
        is_eval=False, 
        algo="sac", 
        wrapper_config=config["wrapper_config"],
        env_config=config["env_config"]
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(2e5))
    initial_lr = 3e-4
    agent = SACAgent(state_dim, action_dim, action_scale=max_action, lr=initial_lr)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"SAC_{config['name']}_{timestamp}"

    # 路径解析：使用绝对路径定锚，彻底杜绝子进程调用时的相对路径漂移问题
    base_output_dir = os.path.join(PROJECT_ROOT, "outputs")
    
    # 给 Logger 传入明确的算法前缀，使其生成的文件夹格式带有 SAC_ 前缀
    logger = Logger(log_dir=os.path.join(base_output_dir, env_name, "logs"), env_name=f"SAC_{config['name']}")

    model_save_dir = os.path.join(base_output_dir, env_name, "models", run_id)
    os.makedirs(model_save_dir, exist_ok=True)
    video_save_dir = os.path.join(base_output_dir, env_name, "videos", run_id) 
    os.makedirs(video_save_dir, exist_ok=True)
    print(f"📁 本次运行的模型权重将保存在: {model_save_dir}")

    max_steps = config["max_steps"]
    start_steps = 1000
    batch_size = 256
    total_steps = 0
    episode = 0
    reward_scale = 1.0
    consecutive_quick_deaths = 0  # 新增：连续暴毙计数器

    # 获取目标参数
    target_jerk = config["wrapper_config"].get("jerk_weight", 0.0)
    target_steering = config["wrapper_config"].get("steering_weight", 0.0)

    while total_steps < max_steps:
        # ==========================================
        # 📈 阶段一：课程学习介入 (Curriculum Learning Phase)
        # 物理意义：在强制 TTC 时空微扰的博弈初期，模型极易被严苛的动作抖动惩罚吓退而选择消极等待。
        # 机制：在预热期内 (Warmup Episodes) 对横向控制约束执行线性松绑，赋予智能体激进变道的试错自由度。
        # ==========================================
        warmup_episodes = 500
        if episode < warmup_episodes:
            cur_jerk = target_jerk * (episode / warmup_episodes)
            cur_steering = target_steering * (episode / warmup_episodes)
        else:
            cur_jerk = target_jerk
            cur_steering = target_steering
        update_env_penalties(env, cur_jerk, cur_steering)

        # ==========================================
        # 📉 阶段二：学习率衰减与策略固化 (Policy Solidification)
        # 机制：在确立了高势能的博弈路径后，大幅收敛优化器的探索步长，锁定全局次优/最优解流域。
        # ==========================================
        decay_start_ep = 800
        decay_duration = 500 
        min_lr = 1e-5
        cur_lr = initial_lr
        
        if episode >= decay_start_ep:
            decay_ratio = min(1.0, (episode - decay_start_ep) / decay_duration)
            cur_lr = initial_lr - decay_ratio * (initial_lr - min_lr)
            set_agent_lr(agent, cur_lr)

        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        c_loss_list, a_loss_list = [], []

        while True:
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # [强制时序截断] 阻断因极度消极博弈导致的无限时序推演陷阱
            if episode_steps >= 100:
                truncated = True
                
            done_bool = float(terminated)
            replay_buffer.add(state, action, reward * reward_scale, next_state, done_bool)

            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1

            if replay_buffer.size > batch_size:
                loss_dict = agent.update(replay_buffer, batch_size)
                c_loss_list.append(loss_dict["critic_loss"])
                a_loss_list.append(loss_dict["actor_loss"])
                
                logger.log_scalar("Loss/Critic", loss_dict["critic_loss"], total_steps)
                logger.log_scalar("Loss/Actor", loss_dict["actor_loss"], total_steps)
                logger.log_scalar("Loss/Alpha", loss_dict.get("alpha_loss", 0), total_steps)
                logger.log_scalar("Metrics/Alpha_Value", loss_dict.get("alpha", 0), total_steps)

            print(f"\r⏳ 引擎运转中... | 全局进度: {total_steps}/{max_steps} 步 | 当前局存活: {episode_steps} 步", end="")

            if terminated or truncated:
                break

        episode += 1
        logger.log_scalar("Reward/Episode_Reward", episode_reward, episode)
        logger.log_scalar("Metrics/Episode_Steps", episode_steps, episode)
        # 记录调度曲线，便于在 TensorBoard 验证
        logger.log_scalar("Schedules/Jerk_Weight", cur_jerk, episode)
        logger.log_scalar("Schedules/Learning_Rate", cur_lr, episode)

        avg_c_loss = np.mean(c_loss_list) if c_loss_list else 0.0
        avg_a_loss = np.mean(a_loss_list) if a_loss_list else 0.0

        print(f"\r🏁 Episode {episode:03d} | Reward: {episode_reward:5.1f} | Steps: {episode_steps:3d} | LR: {cur_lr:.1e} | C_Loss: {avg_c_loss:.3f}")

        # ==========================================
        # 🛡️ 硬件安全护盾：防“一瞬崩塌”显存溢出 (Heuristic VRAM Protection)
        # 故障背景：在连续高速试错时，极短生存期 (<=3步) 的密集回合会导致张量计算图在 GPU 内极速堆积。
        # 解决逻辑：基于生存期阈值进行异常状态探测，触发系统级线程阻塞与深度内存回收。
        # ==========================================
        if episode_steps <= 3:
            consecutive_quick_deaths += 1
        else:
            consecutive_quick_deaths = 0
            
        if consecutive_quick_deaths >= 5:
            import time
            time.sleep(0.5)  # 暂停 0.5 秒，让系统回收资源
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if episode % 200 == 0:
            checkpoint_path = os.path.join(model_save_dir, f"sac_merge_ep{episode}.pth")
            agent.save_model(checkpoint_path)

            # ==========================================
            # 周期性物理探针评估与可视量化录制
            # ==========================================
            print(f"\n🎬 [评估与录制] Episode {episode}，开始录制当前策略表现...")
            eval_env = create_merge_env(env_name=env_name, is_eval=True, algo="sac")
            eval_env = RecordVideo(eval_env, video_folder=video_save_dir, name_prefix=f"ep{episode}")
            
            for i in range(1):
                eval_state, _ = eval_env.reset()
                eval_steps = 0
                while True:
                    action = agent.select_action(eval_state, evaluate=True)
                    eval_state, _, eval_terminated, eval_truncated, _ = eval_env.step(action)
                    eval_steps += 1
                    if eval_steps >= 100:
                        eval_truncated = True
                    if eval_terminated or eval_truncated:
                        break
            eval_env.close()
            print(f"✅ 录制完成，视频已保存至: {video_save_dir}\n")

        # ==========================================
        # 🧹 周期性深度碎片整理 (Periodic Defragmentation)
        # ==========================================
        if episode % 500 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_path = os.path.join(model_save_dir, "sac_merge_final.pth")
    agent.save_model(final_path)

    env.close()
    logger.close()
    print(f"\n🏁 Experiment {config['name']} finished! Models saved to: {model_save_dir}")


if __name__ == "__main__":
    # =====================================================================
    # 🧪 SAC 基础演化消融矩阵 (Ablation Matrix)
    # =====================================================================
    experiment_configs = [
        {
            "name": "M01_Base_Merge",
            "max_steps": 100000,
            "env_config": {
                "collision_reward": -5.0, 
                "high_speed_reward": 1.0, 
                "reward_speed_range": [15, 25],
            },
            "wrapper_config": {"jerk_weight": 0.5, "steering_weight": 0.2}
        },
        # {
        #     "name": "M2_Efficient_Smooth",
        #     "max_steps": 100000, 
        #     "env_config": {
        #         "collision_reward": -8.0, 
        #         "high_speed_reward": 1.5, 
        #         "reward_speed_range": [18, 28], 
        #     },
        #     "wrapper_config": {
        #         "jerk_weight": 0.3, 
        #         "steering_weight": 0.15 
        #     }
        # },
        # {
        #     "name": "M3_Aggressive_Gap_Finding",
        #     "max_steps": 120000, 
        #     "env_config": {
        #         "collision_reward": -10.0, 
        #         "high_speed_reward": 2.0, 
        #         "reward_speed_range": [20, 30], 
        #     },
        #     "wrapper_config": {
        #         "jerk_weight": 0.1, 
        #         "steering_weight": 0.05 
        #     }
        # },
        # {
        #     "name": "M4_Safety_First",
        #     "max_steps": 200000, 
        #     "env_config": {
        #         "collision_reward": -20.0, 
        #         "high_speed_reward": 0.5,  
        #         "reward_speed_range": [15, 25],
        #     },
        #     "wrapper_config": {"jerk_weight": 0.5, "steering_weight": 0.2}
        # },
        #{
        #    "name": "M5_Patient_Merger",
        #    "max_steps": 200000,
        #    "env_config": {
        #        "collision_reward": -15.0,
        #        "high_speed_reward": 1.0,
        #       "reward_speed_range": [15, 20], # 配合 15m/s 的苟活底线
        #    },
        #    "wrapper_config": {"jerk_weight": 0.3, "steering_weight": 0.15} 
        #},
        #{
        #    "name": "M6_Extreme_Penalty",
        #    "max_steps": 200000,
        #    "env_config": {
        #        "collision_reward": -50.0, 
        #        "high_speed_reward": 1.0,
        #        "reward_speed_range": [15, 25],
        #    },
        #    "wrapper_config": {"jerk_weight": 0.1, "steering_weight": 0.05} 
        #},
        #{
        #    "name": "M7_Smooth_Marathon",
        #    "max_steps": 300000, 
        #    "env_config": {
        #        "collision_reward": -15.0,
        #        "high_speed_reward": 1.5,
        #        "reward_speed_range": [15, 25],
        #    },
        #    "wrapper_config": {"jerk_weight": 0.5, "steering_weight": 0.2} 
        #},
        #{
        #    "name": "M8_Ultimate_Merge",
        #    "max_steps": 300000, 
        #    "env_config": {
        #        "collision_reward": -30.0, 
        #        "high_speed_reward": 1.0,  
        #        "reward_speed_range": [15, 25], # 底线保持在 15m/s
        #    },
        #    "wrapper_config": {"jerk_weight": 0.3, "steering_weight": 0.1}
        #},
    ]

    for config in experiment_configs:
        run_single_experiment(config)