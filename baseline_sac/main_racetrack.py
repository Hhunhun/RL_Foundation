"""
Module: SAC Baseline Training Pipeline for Racetrack (赛道竞速基线训练管线)
Description:
    本模块是针对 racetrack-v0 高动态连续曲率场景的 SAC 算法基线 (Baseline) 训练中枢。
    赛道场景要求智能体在极限边缘平衡横向机动性与纵向动能。本管线集成了课程学习调度
    与严格的马尔可夫吸收态修正机制，旨在为后续生成式模型提供兼顾“极限寻迹”与“高速避障”的优质先验底座。

Key Features:
    - Absorbing State Correction: 严格解耦物理终结 (Terminated) 与时序截断 (Truncated)，修复价值网络在长周期任务中的截断截断过估计/低估陷阱。
    - Wrapper Penetration Scheduling: 穿透底层 Gym Wrapper 架构，动态实施平滑惩罚系数的线性退火，实现从“宽容探索”到“精准寻迹”的柔性过渡。
    - Multi-modal Ablation Matrix: 构建了从“保守寻迹”到“极限漂移”的多模态奖励拓扑矩阵，
      以穷举方式生成覆盖不同分布流形 (Data Manifold) 的异构专家数据集。
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
from envs.racetrack_wrapper import create_racetrack_env

# =====================================================================
# 🛠️ 核心辅助函数：穿透 Wrapper 层级动态修改环境惩罚权重 (课程学习用)
# =====================================================================
def update_env_penalties(env, jerk, steering):
    """
    环境约束穿透调度算子 (Environment Constraint Penetration Scheduler)。
    利用 Python 的动态属性反射机制，逐层剥离 (Peeling) 标准化 Gym 环境嵌套，
    精准锚定底层寻迹控制封装器 (RacetrackAVControlWrapper)，实时注入随训练阶段演进的动力学惩罚系数。
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
    阻断因高设定信息熵带来的无意义震荡与分数倒挂。
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
    涵盖环境构建、课程学习调度、价值防爆评估及评估录制的完整马尔可夫决策生命周期。
    """
    env_name = 'racetrack-v0'
    
    print("\n" + "="*60)
    print(f"🏎️ Starting Racetrack Experiment: {config['name']}")
    print("="*60)

    # 1. 结构化创建环境
    env = create_racetrack_env(
        env_name=env_name, 
        is_eval=False, 
        algo="sac", 
        wrapper_config=config["wrapper_config"],
        env_config=config["env_config"]
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
    initial_lr = 3e-4
    agent = SACAgent(state_dim, action_dim, action_scale=max_action, lr=initial_lr)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"SAC_{config['name']}_{timestamp}"

    # 路径解析：使用绝对路径定锚，彻底杜绝子进程调用时的相对路径漂移问题
    base_output_dir = os.path.join(PROJECT_ROOT, "outputs")
    # 给 Logger 传入明确的算法前缀，使其生成的文件夹格式类似于 SAC_R1_Base_时间戳
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
    consecutive_quick_deaths = 0  # 连续崩溃计数器

    # 获取目标参数
    target_jerk = config["wrapper_config"].get("jerk_weight", 0.0)
    target_steering = config["wrapper_config"].get("steering_weight", 0.0)

    while total_steps < max_steps:
        # ==========================================
        # 📈 阶段一：课程学习介入 (Curriculum Learning Phase)
        # 机制：在长视界赛道预热期内 (Warmup Episodes) 对一阶抖动与横向控制约束执行线性松绑，
        # 允许智能体在早期的参数空间探索中采取激进打盘动作以快速适应极端的几何曲率。
        # ==========================================
        warmup_episodes = 800
        if episode < warmup_episodes:
            cur_jerk = target_jerk * (episode / warmup_episodes)
            cur_steering = target_steering * (episode / warmup_episodes)
        else:
            cur_jerk = target_jerk
            cur_steering = target_steering
        update_env_penalties(env, cur_jerk, cur_steering)

        # ==========================================
        # 📉 阶段二：学习率衰减与策略固化 (Policy Solidification)
        # ==========================================
        decay_start_ep = 1500
        decay_duration = 1500 
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
            
            # ==========================================
            # [时序截断与吸收态修正 (Time-Limit Truncation & Absorbing State Correction)]
            # 物理意义：赛道场景时空跨度极大。若将超时截断误判为物理碰撞，会导致 Q 价值网络在此处产生严重的陡降断层。
            # 算法机制：严格剥离 Terminated 与 Truncated。超时 (Truncated) 时将 done_bool 置 0，
            # 保证贝尔曼方程中 \gamma V(s_{t+1}) 的连续性计算，从而输出无偏差的时间差分误差 (TD-Error)。
            # ==========================================
            if episode_steps >= 500: 
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
        logger.log_scalar("Schedules/Jerk_Weight", cur_jerk, episode)
        logger.log_scalar("Schedules/Learning_Rate", cur_lr, episode)

        avg_c_loss = np.mean(c_loss_list) if c_loss_list else 0.0

        print(f"\r🏁 Episode {episode:03d} | Reward: {episode_reward:5.1f} | Steps: {episode_steps:3d} | LR: {cur_lr:.1e} | C_Loss: {avg_c_loss:.3f}")

        # ==========================================
        # 🛡️ 硬件安全护盾：防“一瞬崩塌”显存溢出 (Heuristic VRAM Protection)
        # 故障背景：长赛道极其容易产生出门即撞墙 (<=3步) 的病态探索循环，
        # 密集的重置会导致张量图在 GPU 内极速堆积而无法被 Python GC 及时回收。
        # 解决逻辑：监测回合生存期特征，触发硬件级线程阻塞 (Sleep) 并强制执行深层垃圾清理。
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

        if episode % 250 == 0:
            checkpoint_path = os.path.join(model_save_dir, f"sac_racetrack_ep{episode}.pth")
            agent.save_model(checkpoint_path)

            # ==========================================
            # 周期性物理探针评估与可视量化录制
            # ==========================================
            print(f"\n🎬 [评估与录制] Episode {episode}，开始录制当前策略表现...")
            eval_env = create_racetrack_env(env_name=env_name, is_eval=False, algo="sac")
            eval_env = RecordVideo(eval_env, video_folder=video_save_dir, name_prefix=f"ep{episode}")
            
            for i in range(1):
                eval_state, _ = eval_env.reset()
                eval_steps = 0
                while True:
                    action = agent.select_action(eval_state, evaluate=True)
                    eval_state, _, eval_terminated, eval_truncated, _ = eval_env.step(action)
                    eval_steps += 1
                    if eval_steps >= 500:
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

    final_path = os.path.join(model_save_dir, "sac_racetrack_final.pth")
    agent.save_model(final_path)

    env.close()
    logger.close()
    print(f"\n🏁 Experiment {config['name']} finished! Models saved to: {model_save_dir}")


if __name__ == "__main__":
    # =====================================================================
    # 🧪 SAC 多模态演化消融矩阵 (Multi-modal Ablation Matrix)
    # =====================================================================
    experiment_configs = [
        {
            "name": "R01_SAC_Baseline",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -10.0,
                "high_speed_reward": 2.0,
                "reward_speed_range": [15, 30],
            },
            "wrapper_config": {"jerk_weight": 0.5, "steering_weight": 0.2} # [基准配置] 标准 L2 动力学惩罚
        },
        {
            "name": "R02_SAC_Speed_Priority",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -15.0, 
                "high_speed_reward": 3.0, 
                "reward_speed_range": [20, 35], # [激进探索] 强化正反馈驱动面，强迫冲击更高速度阈值
            },
            "wrapper_config": {"jerk_weight": 0.05, "steering_weight": 0.1} # 解除横向平滑约束，激发高速极限漂移过弯潜力
        },
        {
            "name": "R03_SAC_Safety_Priority",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -30.0, 
                "high_speed_reward": 1.5, 
                "reward_speed_range": [10, 20], # [稳健控制] 压低激励域，确立安全为主的控制协议
            },
            "wrapper_config": {"jerk_weight": 1.0, "steering_weight": 0.5} # 重度惩罚游离变轨，逼迫系统沿最优曲率切线稳健行驶
        },
        {
            "name": "R04_SAC_Extreme_Drift",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -5.0, # 降低安全边界权重，极大鼓励动力学试错
                "high_speed_reward": 3.5,
                "reward_speed_range": [25, 40], 
            },
            "wrapper_config": {"jerk_weight": 0.0, "steering_weight": 0.0} # 彻底关闭防画龙机制，构成纯自由度流形
        },
        {
            "name": "R05_SAC_Smooth_Racing",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -20.0,
                "high_speed_reward": 2.0,
                "reward_speed_range": [15, 25],
            },
            "wrapper_config": {"jerk_weight": 0.8, "steering_weight": 0.3} # [专家寻迹] 高平滑正则化，逼迫模型沿 "外-内-外" 的赛车几何最优线行驶
        },
        {
            "name": "R06_SAC_Wide_Dynamic",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -25.0,
                "high_speed_reward": 2.5,
                "reward_speed_range": [10, 35], # [宽泛感知域] 极宽的速度奖励区间，评估连续刹车与极限加速的动态切换能力
            },
            "wrapper_config": {"jerk_weight": 0.2, "steering_weight": 0.2}
        },
        {
            "name": "R07_SAC_Zero_Tolerance",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -50.0, # 施加最高级别的死亡惩罚阻断网格
                "high_speed_reward": 1.5,
                "reward_speed_range": [15, 25],
            },
            "wrapper_config": {"jerk_weight": 0.5, "steering_weight": 0.2}
        },
        {
            "name": "R08_SAC_Expert_Pro",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -40.0,
                "high_speed_reward": 2.5,
                "reward_speed_range": [15, 30],
            },
            "wrapper_config": {"jerk_weight": 0.1, "steering_weight": 0.1} # [多源融合] 适中的平滑容忍度，提纯专用的高维六边形专家底层
        }
    ]

    for config in experiment_configs:
        run_single_experiment(config)