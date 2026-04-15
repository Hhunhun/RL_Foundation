import os
import numpy as np
import torch
from core.replay_buffer import ReplayBuffer
from algorithms.sac.sac_agent import SACAgent
from utils.logger import Logger
from envs import create_environment
from datetime import datetime # Added import


def run_single_experiment(config):
    env_name = 'merge-v0'
    
    print("\n" + "="*60)
    print(f"🚀 Starting Experiment: {config['name']}")
    print("="*60)

    # Create environment with specific config
    env = create_environment(env_name, is_eval=False, algo="sac", wrapper_config=config["wrapper_config"])
    env.unwrapped.configure(config["env_config"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(2e5))
    agent = SACAgent(state_dim, action_dim, action_scale=max_action, lr=3e-4)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"SAC_{config['name']}_{timestamp}"

    logger = Logger(log_dir=os.path.join("../outputs", env_name, "logs"), env_name=run_id)

    model_save_dir = os.path.join("../outputs", env_name, "models", run_id)
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"📁 本次运行的模型权重将独立保存在: {model_save_dir}")

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
                logger.log_scalar("Loss/Critic", loss_dict["critic_loss"], total_steps)
                logger.log_scalar("Loss/Actor", loss_dict["actor_loss"], total_steps)
                logger.log_scalar("Loss/Alpha", loss_dict["alpha_loss"], total_steps)
                logger.log_scalar("Metrics/Alpha_Value", loss_dict["alpha"], total_steps)

            if terminated or truncated:
                break

        episode += 1
        logger.log_scalar("Reward/Episode_Reward", episode_reward, episode)
        logger.log_scalar("Metrics/Episode_Steps", episode_steps, episode)

        print(
            f"Episode: {episode} | 存活步数: {episode_steps} | 总步数: {total_steps}/{max_steps} | 回报: {episode_reward:.2f}")

        if episode % 100 == 0:
            checkpoint_path = os.path.join(model_save_dir, f"sac_merge_ep{episode}.pth")
            agent.save_model(checkpoint_path)

    final_path = os.path.join(model_save_dir, "sac_merge_final.pth")
    agent.save_model(final_path)

    env.close()
    logger.close()
    print(f"🏁 Experiment {config['name']} finished! Models saved to: {model_save_dir}")


if __name__ == "__main__":
    experiment_configs = [
        {
            "name": "M1_Base_Merge",
            "max_steps": 80000,
            "env_config": {
                "collision_reward": -5.0, "right_lane_reward": 0.2,
                "high_speed_reward": 1.0, "reward_speed_range": [15, 25],
            },
            "wrapper_config": {"jerk_weight": 0.5, "steering_weight": 0.2}
        },
        {
            "name": "M2_Efficient_Merge",
            "max_steps": 100000,
            "env_config": {
                "collision_reward": -10.0, "right_lane_reward": 0.1,
                "high_speed_reward": 3.0, "reward_speed_range": [20, 30],
            },
            "wrapper_config": {"jerk_weight": 1.0, "steering_weight": 0.5}
        },
        {
            "name": "M3_Aggressive_Merge",
            "max_steps": 120000,
            "env_config": {
                "collision_reward": -10.0, "right_lane_reward": 0.1,
                "high_speed_reward": 5.0, "reward_speed_range": [25, 35],
            },
            "wrapper_config": {"jerk_weight": 0.8, "steering_weight": 0.4}
        },
    ]

    for config in experiment_configs:
        run_single_experiment(config)