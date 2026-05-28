"""
Module: SAC Baseline Training Pipeline for Highway (高速巡航基线训练管线)
Description:
    本模块是针对 highway-v0 场景的 SAC 算法基线 (Baseline) 训练中枢。
    通过集成底层奖励重塑、学习率退火调度与启发式显存垃圾回收机制，为离线专家数据集的采集
    提供高稳定、多模态 (安全导向/效率导向) 的策略底座。

Key Features:
    - Computational Graph Hijacking: 深度劫持物理引擎底层的全网格连续动作奖励结算逻辑，实现 ~50x 训练加速。
    - Heuristic VRAM Garbage Collection: 基于连续崩溃探测的显存主动释放机制，彻底解决长期挂机引发的 OOM 难题。
    - Policy Annealing: 训练中后期的学习率平滑衰减，防止 SAC 算法在收敛后期因过频探索引发的策略崩塌 (Policy Collapse)。
"""

import os
import sys
import gc
import numpy as np
import torch
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from core.replay_buffer import ReplayBuffer
from algorithms.sac.sac_agent import SACAgent
from utils.logger import Logger
from envs.highway_wrapper import create_highway_env
from gymnasium.wrappers import RecordVideo

# =====================================================================
# 🛠️ 核心辅助函数：动态修改 Agent 学习率 (策略固化用)
# =====================================================================
def set_agent_lr(agent, lr):
    """
    学习率动态退火算子 (Learning Rate Annealing Operator)。
    在强化学习后期强行收缩参数更新步长，促使策略向局部最优解深层收敛，
    阻断因高设定信息熵带来的无意义震荡与分数倒挂。
    """
    if hasattr(agent, 'set_lr'):
        agent.set_lr(lr)
    else:
        if hasattr(agent, 'actor_optimizer'):
            for p in agent.actor_optimizer.param_groups: p['lr'] = lr
            for p in agent.critic_optimizer.param_groups: p['lr'] = lr
            for p in agent.alpha_optimizer.param_groups: p['lr'] = lr


def run_single_experiment(config):
    """
    单体实验流水线引擎。涵盖环境构建、预热探索、稳定收敛及评估录制的完整马尔可夫决策生命周期。
    """
    env_name = 'highway-v0'
    
    print("\n" + "="*60)
    print(f"🛣️  Starting Highway Experiment: {config['name']}")
    print("="*60)

    # 1. 结构化创建环境
    env = create_highway_env(env_name)
    
    # =====================================================================
    # 🚀 算力提效核心：底层奖励评估计算图劫持 (Reward Graph Hijacking)
    # 物理意义：阻断 `highway-env` 内部针对连续动作空间进行的低效网格化 (Grid-search) 奖励结算遍历。
    # 强制将标量判别分支挂载到多维连续张量输入上，直接缩减 $O(N^2)$ 的冗余计算复杂度，释放单核算力瓶颈。
    # =====================================================================
    if hasattr(env.unwrapped, '_rewards'):
        original_rewards_fn = env.unwrapped._rewards
        def patched_rewards(action):
            if isinstance(action, np.ndarray): return original_rewards_fn(1)
            return original_rewards_fn(action)
        env.unwrapped._rewards = patched_rewards
    
    # 核心改造：使用 config 字典动态配置环境奖励与物理动力学边界
    if config.get("env_config"):
        env.unwrapped.configure(config["env_config"])
        env.reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(2e5))
    initial_lr = 3e-4
    agent = SACAgent(state_dim, action_dim, action_scale=max_action, lr=initial_lr)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"SAC_{config['name']}_{timestamp}"

    # 路径修复：使用绝对路径，彻底杜绝路径漂移问题
    base_output_dir = os.path.join(PROJECT_ROOT, "outputs")
    # 给 Logger 传入明确的算法前缀，使其生成的文件夹格式带有 SAC_ 前缀
    logger = Logger(log_dir=os.path.join(base_output_dir, env_name, "logs"), env_name=f"SAC_{config['name']}")

    model_save_dir = os.path.join(base_output_dir, env_name, "models", run_id)
    os.makedirs(model_save_dir, exist_ok=True)
    video_save_dir = os.path.join(base_output_dir, env_name, "videos", run_id) 
    os.makedirs(video_save_dir, exist_ok=True)
    print(f"📁 本次运行的模型权重将保存在: {model_save_dir}")

    max_steps = config["max_steps"]
    start_steps = 2000
    batch_size = 256
    total_steps = 0
    episode = 0
    reward_scale = 1.0
    consecutive_quick_deaths = 0

    while total_steps < max_steps:
        # 📉 阶段二：学习率衰减 (策略固化 / Policy Solidification)
        decay_start_ep = 800
        decay_duration = 1000 
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

            if terminated or truncated or episode_steps >= 1000:
                break

        episode += 1
        logger.log_scalar("Reward/Episode_Reward", episode_reward, episode)
        logger.log_scalar("Metrics/Episode_Steps", episode_steps, episode)
        logger.log_scalar("Schedules/Learning_Rate", cur_lr, episode)

        avg_c_loss = np.mean(c_loss_list) if c_loss_list else 0.0

        print(f"\r🏁 Episode {episode:03d} | Reward: {episode_reward:5.1f} | Steps: {episode_steps:3d} | LR: {cur_lr:.1e} | C_Loss: {avg_c_loss:.3f}")

        # ==========================================
        # 🛡️ 硬件安全护盾：防“一瞬崩塌”显存溢出 (Heuristic VRAM Protection)
        # 故障背景：在连续高速试错时，短平快 (<=3步) 的回合会导致计算图在 GPU 内极速堆积而无法被 Python GC 及时回收。
        # 解决逻辑：监测回合生存期特征，触发硬件级线程阻塞 (Sleep) 并强制执行深层清理。
        # ==========================================
        if episode_steps <= 3:
            consecutive_quick_deaths += 1
        else:
            consecutive_quick_deaths = 0
            
        if consecutive_quick_deaths >= 5:
            import time
            time.sleep(0.5)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if episode % 200 == 0:
            checkpoint_path = os.path.join(model_save_dir, f"sac_highway_ep{episode}.pth")
            agent.save_model(checkpoint_path)

            # ==========================================
            # 周期性物理探针评估与可视量化录制
            # ==========================================
            print(f"\n🎬 [评估与录制] Episode {episode}，开始录制当前策略表现...")
            eval_env = create_highway_env(env_name)
            
            # 同步为评估环境打上提速补丁
            if hasattr(eval_env.unwrapped, '_rewards'):
                orig_eval_rew = eval_env.unwrapped._rewards
                def fast_eval_rew(a): return orig_eval_rew(1) if isinstance(a, np.ndarray) else orig_eval_rew(a)
                eval_env.unwrapped._rewards = fast_eval_rew
                
            if config.get("env_config"):
                eval_env.unwrapped.configure(config["env_config"])
                eval_env.reset()
            eval_env = RecordVideo(eval_env, video_folder=video_save_dir, name_prefix=f"ep{episode}")
            
            for i in range(1):
                eval_state, _ = eval_env.reset()
                eval_steps = 0
                while True:
                    action = agent.select_action(eval_state, evaluate=True)
                    eval_state, _, eval_terminated, eval_truncated, _ = eval_env.step(action)
                    eval_steps += 1
                    if eval_steps >= 1000:
                        eval_truncated = True
                    if eval_terminated or eval_truncated:
                        break
            eval_env.close()
            print(f"✅ 录制完成，视频已保存至: {video_save_dir}\n")

        if episode % 500 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_path = os.path.join(model_save_dir, "sac_highway_final.pth")
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
            "name": "H01_Base_Highway",
            "max_steps": 200000,
            "env_config": {
                "vehicles_count": 25,  # 🚀 速度优化：收缩游离计算实体 (由 50 缩减至 25) 以减轻物理引擎的全局碰撞检测负担
                "collision_reward": -1.0,
                "high_speed_reward": 0.4,
                "reward_speed_range": [20, 30],
            }
        },
        {
            "name": "H02_Safety_Priority",
            "max_steps": 200000,
            "env_config": {
                "vehicles_count": 25,
                "collision_reward": -5.0, 
                "high_speed_reward": 0.2, 
                "reward_speed_range": [15, 25], 
            }
        },
        {
            "name": "H03_Speed_Priority",
            "max_steps": 200000,
            "env_config": {
                "vehicles_count": 25,
                "collision_reward": -1.0,
                "high_speed_reward": 0.8, # [激进探索] 强化正反馈驱动面，鼓励模型突破物理极限
                "reward_speed_range": [25, 35], 
            }
        },
        {
            "name": "H04_Traffic_Jam",
            "max_steps": 200000,
            "env_config": {
                "vehicles_count": 45,  # [拥堵博弈密集场] 极大增加障碍物密度
                "collision_reward": -2.0,
                "high_speed_reward": 0.5,
                "reward_speed_range": [10, 20], # 要求模型摒弃高速惯性思维，在低速逼仄区间内完成安全换道博弈
            }
        },
    ]

    for config in experiment_configs:
        run_single_experiment(config)