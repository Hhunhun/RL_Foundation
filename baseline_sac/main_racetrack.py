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
    curr_env = env
    # 逐层剥开 gym.Wrapper，直到找到我们的 RacetrackAVControlWrapper
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
    if hasattr(agent, 'set_lr'):
        agent.set_lr(lr)
    else:
        # 如果 SACAgent 内部没有写 set_lr 方法，强行修改优化器参数兜底
        if hasattr(agent, 'actor_optimizer'):
            for p in agent.actor_optimizer.param_groups: p['lr'] = lr
            for p in agent.critic_optimizer.param_groups: p['lr'] = lr
            for p in agent.alpha_optimizer.param_groups: p['lr'] = lr

def run_single_experiment(config):
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

    # 路径修复：使用绝对路径，彻底杜绝路径漂移问题
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

    # 获取目标参数
    target_jerk = config["wrapper_config"].get("jerk_weight", 0.0)
    target_steering = config["wrapper_config"].get("steering_weight", 0.0)

    while total_steps < max_steps:
        # 📈 阶段一：课程学习 (前期放开方向盘惩罚，让模型敢于探索弯道)
        warmup_episodes = 800
        if episode < warmup_episodes:
            cur_jerk = target_jerk * (episode / warmup_episodes)
            cur_steering = target_steering * (episode / warmup_episodes)
        else:
            cur_jerk = target_jerk
            cur_steering = target_steering
        update_env_penalties(env, cur_jerk, cur_steering)

        # 📉 阶段二：学习率衰减 (固化策略，防止因 SAC 探索导致突然崩盘)
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
            
            # [关键修改] Racetrack 赛道很长，不要轻易 truncated
            # 只有当环境真正由于碰撞或出界返回 terminated 时，才标记为死亡
            # 步数限制导致的截断不应计入 Q 值的死亡衰减
            if episode_steps >= 500: # 随机起点可能需要更多步数来跑
                truncated = True
                
            # SAC 核心修正：
            # 如果是碰撞(terminated)，done_bool=1
            # 如果是时间到了(truncated)，done_bool=0
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

        if episode % 250 == 0:
            checkpoint_path = os.path.join(model_save_dir, f"sac_racetrack_ep{episode}.pth")
            agent.save_model(checkpoint_path)

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

        if episode % 500 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_path = os.path.join(model_save_dir, "sac_racetrack_final.pth")
    agent.save_model(final_path)

    env.close()
    logger.close()
    print(f"\n🏁 Experiment {config['name']} finished! Models saved to: {model_save_dir}")


# ... existing code ...
if __name__ == "__main__":
    experiment_configs = [
        {
            "name": "R01_SAC_Baseline",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -10.0,
                "high_speed_reward": 2.0,
                "reward_speed_range": [15, 30],
            },
            "wrapper_config": {"jerk_weight": 0.5, "steering_weight": 0.2} # 标准平滑惩罚
        },
        {
            "name": "R02_SAC_Speed_Priority",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -15.0, 
                "high_speed_reward": 3.0, # 极速诱惑
                "reward_speed_range": [20, 35], # 强迫模型冲击更高速度
            },
            "wrapper_config": {"jerk_weight": 0.05, "steering_weight": 0.1} # 解除平滑紧箍咒，允许漂移过弯
        },
        {
            "name": "R03_SAC_Safety_Priority",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -30.0, 
                "high_speed_reward": 1.5, 
                "reward_speed_range": [10, 20], # 收敛车速，安全第一
            },
            "wrapper_config": {"jerk_weight": 1.0, "steering_weight": 0.5} # 重罚方向盘画龙，逼迫老奶奶开法
        },
        {
            "name": "R04_SAC_Extreme_Drift",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -5.0, # 极低碰撞惩罚，鼓励试错
                "high_speed_reward": 3.5,
                "reward_speed_range": [25, 40], # 狂野极速
            },
            "wrapper_config": {"jerk_weight": 0.0, "steering_weight": 0.0} # 完全不管方向盘
        },
        {
            "name": "R05_SAC_Smooth_Racing",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -20.0,
                "high_speed_reward": 2.0,
                "reward_speed_range": [15, 25],
            },
            "wrapper_config": {"jerk_weight": 0.8, "steering_weight": 0.3} # 逼迫模型走外内外完美赛车线
        },
        {
            "name": "R06_SAC_Wide_Dynamic",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -25.0,
                "high_speed_reward": 2.5,
                "reward_speed_range": [10, 35], # 极宽的动态域，测试刹车和加速的适应性
            },
            "wrapper_config": {"jerk_weight": 0.2, "steering_weight": 0.2}
        },
        {
            "name": "R07_SAC_Zero_Tolerance",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -50.0, # 死亡阴影
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
            "wrapper_config": {"jerk_weight": 0.1, "steering_weight": 0.1} # 适中平滑，提纯专用的终极六边形
        }
    ]

    for config in experiment_configs:
        run_single_experiment(config)