import os
import sys
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
# 专门引入新的 Merge 工厂函数
from envs.merge_wrapper import create_merge_env


def run_single_experiment(config):
    env_name = 'merge-v0'
    
    print("\n" + "="*60)
    print(f"🚀 Starting Merge Experiment: {config['name']}")
    print("="*60)

    # 1. 结构化创建环境（配置直接在内部合并生效）
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
    agent = SACAgent(state_dim, action_dim, action_scale=max_action, lr=3e-4)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"SAC_{config['name']}_{timestamp}"

    # 路径修复：使用绝对路径，彻底杜绝路径漂移问题
    base_output_dir = os.path.join(PROJECT_ROOT, "outputs")
    logger = Logger(log_dir=os.path.join(base_output_dir, env_name, "logs"), env_name=run_id)

    model_save_dir = os.path.join(base_output_dir, env_name, "models", run_id)
    os.makedirs(model_save_dir, exist_ok=True)
    video_save_dir = os.path.join(base_output_dir, env_name, "videos", run_id) # 新增视频保存目录
    os.makedirs(video_save_dir, exist_ok=True)
    print(f"📁 本次运行的模型权重将保存在: {model_save_dir}")

    max_steps = config["max_steps"]
    start_steps = 2000
    batch_size = 256
    total_steps = 0
    episode = 0
    reward_scale = 1.0

    while total_steps < max_steps:
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
            
            # [强制物理截断] 修复 1000 步陷阱！匝道博弈不可能超过 20 秒(100步)
            # 如果环境底层 duration 机制失效，这里做最后一道防线
            if episode_steps >= 100:
                truncated = True
                
            # SAC 关键细节：对于截断(超时)，并不认为是真正的死亡(terminated)
            # done_bool 只能由 terminated 触发，这样 Q 值才能正确引导
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
                logger.log_scalar("Loss/Alpha", loss_dict["alpha_loss"], total_steps)
                logger.log_scalar("Metrics/Alpha_Value", loss_dict["alpha"], total_steps)

            print(f"\r⏳ 引擎运转中... | 全局进度: {total_steps}/{max_steps} 步 | 当前局存活: {episode_steps} 步", end="")

            # 只要触发了死亡(terminated)或超时截断(truncated)，就结束本局
            if terminated or truncated:
                break

        episode += 1
        logger.log_scalar("Reward/Episode_Reward", episode_reward, episode)
        logger.log_scalar("Metrics/Episode_Steps", episode_steps, episode)

        avg_c_loss = np.mean(c_loss_list) if c_loss_list else 0.0
        avg_a_loss = np.mean(a_loss_list) if a_loss_list else 0.0

        print(f"\r🏁 Episode {episode:03d} | Reward: {episode_reward:5.1f} | Steps: {episode_steps:3d} | Total: {total_steps}/{max_steps} | C_Loss: {avg_c_loss:.3f} | A_Loss: {avg_a_loss:.3f}")

        if episode % 200 == 0:
            checkpoint_path = os.path.join(model_save_dir, f"sac_merge_ep{episode}.pth")
            agent.save_model(checkpoint_path)

            # ==========================================
            # 定期评估与视频录制
            # ==========================================
            print(f"\n🎬 [评估与录制] Episode {episode}，开始录制当前策略表现...")
            eval_env = create_merge_env(env_name=env_name, is_eval=True, algo="sac")
            # 开启自动录屏，配置好触发条件
            eval_env = RecordVideo(eval_env, video_folder=video_save_dir, name_prefix=f"ep{episode}")
            
            for i in range(1): # 只录制一局作为快照
                eval_state, _ = eval_env.reset()
                eval_steps = 0
                while True:
                    action = agent.select_action(eval_state, evaluate=True)
                    eval_state, _, eval_terminated, eval_truncated, _ = eval_env.step(action)
                    eval_steps += 1
                    # 评估时也必须加入强制防卡死截断
                    if eval_steps >= 100:
                        eval_truncated = True
                    if eval_terminated or eval_truncated:
                        break
            eval_env.close()
            print(f"✅ 录制完成，视频已保存至: {video_save_dir}\n")
            # ==========================================

    final_path = os.path.join(model_save_dir, "sac_merge_final.pth")
    agent.save_model(final_path)

    env.close()
    logger.close()
    print(f"\n🏁 Experiment {config['name']} finished! Models saved to: {model_save_dir}")


if __name__ == "__main__":
    experiment_configs = [
        # {
        #     "name": "M1_Base_Merge",
        #     "max_steps": 80000,
        #     "env_config": {
        #         "collision_reward": -5.0, # 初期给一点容错率，跑通之后可加强到 -10.0
        #         "high_speed_reward": 1.0, 
        #         "reward_speed_range": [15, 25],
        #     },
        #     "wrapper_config": {"jerk_weight": 0.5, "steering_weight": 0.2}
        # },
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
        {
            "name": "M4_Safety_First",
            "max_steps": 200000, # 假设是欠拟合，将步数翻倍以充分探索博弈
            "env_config": {
                "collision_reward": -20.0, # 大幅加大碰撞惩罚
                "high_speed_reward": 0.5,  # 削减速度诱惑，鼓励"宁愿慢点也要活下来"
                "reward_speed_range": [15, 25],
            },
            "wrapper_config": {"jerk_weight": 0.5, "steering_weight": 0.2}
        },
        {
            "name": "M5_Patient_Merger",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -15.0,
                "high_speed_reward": 1.0,
                "reward_speed_range": [10, 20], # 核心改变：大幅下调目标速度下限，允许智能体耐心减速甚至停车等待间隙
            },
            "wrapper_config": {"jerk_weight": 0.3, "steering_weight": 0.15} # 延续 M2 的优秀平滑参数
        },
        {
            "name": "M6_Extreme_Penalty",
            "max_steps": 200000,
            "env_config": {
                "collision_reward": -50.0, # 增大惩罚，测试高分但坠毁的模型是否仅仅是因为代价不够痛
                "high_speed_reward": 1.0,
                "reward_speed_range": [15, 25],
            },
            "wrapper_config": {"jerk_weight": 0.1, "steering_weight": 0.05} # 给动作更多物理自由度去规避致命碰撞
        },
        {
            "name": "M7_Smooth_Marathon",
            "max_steps": 300000, # 超长马拉松！10万步可能根本无法穷尽主路车辆的随机组合
            "env_config": {
                "collision_reward": -15.0,
                "high_speed_reward": 1.5,
                "reward_speed_range": [15, 25],
            },
            "wrapper_config": {"jerk_weight": 0.5, "steering_weight": 0.2} # 强平滑约束，保证 30 万步依然保持低策略方差
        },
        {
            "name": "M8_Ultimate_Merge",
            "max_steps": 300000, 
            "env_config": {
                "collision_reward": -30.0, 
                "high_speed_reward": 1.0,  
                "reward_speed_range": [12, 22], # 最符合真实汇入场景的黄金物理区间
            },
            "wrapper_config": {"jerk_weight": 0.3, "steering_weight": 0.1}
        },
    ]

    for config in experiment_configs:
        run_single_experiment(config)