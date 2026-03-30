import os
import numpy as np
import torch
from core.replay_buffer import ReplayBuffer
from algorithms.sac.sac_agent import SACAgent
from utils.logger import Logger
from envs.highway_wrapper import create_highway_env


def main():
    env_name = 'highway-v0'
    print("正在初始化 HighwayEnv v4.0 (LQR护甲 + 步数驱动版)...")
    env = create_highway_env(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"[{env_name}] State dim: {state_dim} | Action dim: {action_dim} | Max action: {max_action}")

    # 实例化核心组件
    # 扩容经验池：复杂环境需要更多的历史数据来稳定 Critic 的评估
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(2e5))
    agent = SACAgent(state_dim, action_dim, action_scale=max_action, lr=3e-4)
    logger = Logger(log_dir="outputs/logs", env_name=env_name)

    # 动态模型存储路径对齐 (解决模型坟场问题)
    run_name = os.path.basename(os.path.normpath(logger.run_dir))
    model_save_dir = os.path.join("outputs", "models", run_name)
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"📁 本次运行的模型权重将独立保存在: {model_save_dir}")

    # ---------------------------------------------------------
    # 🎯 [v4.0 大修] 训练逻辑从“按局数”升级为“按总步数”
    # 彻底解决早期暴毙(活不过5步)导致的 Actor 训练不充分问题
    # ---------------------------------------------------------
    max_steps = 120000  # 核心KPI：强制小车在环境里实打实地活够 4 万步
    start_steps = 2000  # 缩短随机探索期(前2000步瞎打方向盘积累负面教训)，早点让大脑接管
    batch_size = 256

    total_steps = 0  # 全局步数计数器
    episode = 0  # 局数计数器 (仅用于日志记录和保存检查点)
    reward_scale = 1.0  # 奖励缩放因子

    print(f"\n🚀 引擎点火，目标 {max_steps} 步，开始自动驾驶魔鬼训练...")

    # 核心循环：以总步数为绝对衡量标准
    while total_steps < max_steps:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0

        while True:
            # 动作选择 (带有预热探索机制)
            if total_steps < start_steps:
                action = env.action_space.sample()  # 纯随机：必然导致早期疯狂冲出草地暴毙
            else:
                action = agent.select_action(state, evaluate=False)  # 大脑接管：开始求生

            # 环境交互
            next_state, reward, terminated, truncated, _ = env.step(action)

            # 致命 Bug 防御：只将真实的物理死亡(terminated)视为 done
            done_bool = float(terminated)

            # 存入经验池 (应用 reward_scale)
            replay_buffer.add(state, action, reward * reward_scale, next_state, done_bool)

            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1

            # 网络更新 (当经验池攒够一个 Batch 时才开始反向传播)
            if replay_buffer.size > batch_size:
                loss_dict = agent.update(replay_buffer, batch_size)

                # 将高频数据推送到 TensorBoard (每步记录，利用 flush 实时落盘)
                logger.log_scalar("Loss/Critic", loss_dict["critic_loss"], total_steps)
                logger.log_scalar("Loss/Actor", loss_dict["actor_loss"], total_steps)
                logger.log_scalar("Loss/Alpha", loss_dict["alpha_loss"], total_steps)
                logger.log_scalar("Metrics/Alpha_Value", loss_dict["alpha"], total_steps)

            if terminated or truncated:
                break

        episode += 1

        # 记录宏观情景指标 (每局记录)
        logger.log_scalar("Reward/Episode_Reward", episode_reward, episode)
        logger.log_scalar("Metrics/Episode_Steps", episode_steps, episode)

        # 动态日志输出：此时“总步数进度”比“当前局数”重要得多
        print(
            f"Episode: {episode} | 存活步数: {episode_steps} | 总步数: {total_steps}/{max_steps} | 回报: {episode_reward:.2f}")

        # 定期保存检查点 (每跑完 100 局存一次档)
        if episode % 100 == 0:
            checkpoint_path = os.path.join(model_save_dir, f"sac_ep{episode}.pth")
            agent.save_model(checkpoint_path)

    # 最终完整保存 (基于总步数达成)
    final_path = os.path.join(model_save_dir, "sac_highway_final.pth")
    agent.save_model(final_path)

    env.close()
    logger.close()
    print(f"🏁 {max_steps} 步魔鬼训练彻底结束！所有权重已安全归档至: {model_save_dir}")


if __name__ == "__main__":
    main()