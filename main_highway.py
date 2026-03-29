import os
import numpy as np
import torch
from core.replay_buffer import ReplayBuffer
from algorithms.sac.sac_agent import SACAgent
from utils.logger import Logger
from envs.highway_wrapper import create_highway_env


def main():
    env_name = 'highway-v0'

    # 1. 使用我们定制的“环境工厂”实例化 HighwayEnv
    print("正在初始化 HighwayEnv 并应用降维装甲...")
    env = create_highway_env(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"[{env_name}] State dim: {state_dim} | Action dim: {action_dim} | Max action: {max_action}")

    # 2. 实例化核心组件
    # 扩容经验池：复杂环境需要更多的历史数据来稳定 Critic 的评估
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(2e5))
    agent = SACAgent(state_dim, action_dim, action_scale=max_action, lr=3e-4)

    # 实例化日志记录器
    logger = Logger(log_dir="outputs/logs", env_name=env_name)

    # ---------------------------------------------------------
    # [架构优化] 动态模型存储路径对齐
    # 提取 logger 自动生成的时间戳文件夹名，在 models 下建立同名镜像目录
    # 解决模型权重混淆的问题
    # ---------------------------------------------------------
    run_name = os.path.basename(os.path.normpath(logger.run_dir))
    model_save_dir = os.path.join("outputs", "models", run_name)
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"📁 本次运行的模型权重将独立保存在: {model_save_dir}")

    # 3. 设定实战级超参数
    max_episodes = 500  # HighwayEnv 难度较大，需要更多轮次
    batch_size = 256
    start_steps = 3000  # 延长纯随机探索期，让车辆在初期多撞几次，积累“血的教训”(负面奖励)
    total_steps = 0
    reward_scale = 1.0  # 奖励缩放因子 (保留此接口，若后续发现 Q 值太小，可调至 5.0)

    print("\n开始训练...")

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0

        while True:
            # 动作选择 (带有预热探索机制)
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)

            # 环境交互
            next_state, reward, terminated, truncated, _ = env.step(action)

            # 致命 Bug 防御：只将真实的物理死亡(terminated)视为 done
            # 如果是因为达到了最大步数而超时(truncated)，未来状态仍然有价值，绝不能截断 Bellman 方程！
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

                # 将高频数据推送到 TensorBoard (每步记录)
                logger.log_scalar("Loss/Critic", loss_dict["critic_loss"], total_steps)
                logger.log_scalar("Loss/Actor", loss_dict["actor_loss"], total_steps)
                logger.log_scalar("Loss/Alpha", loss_dict["alpha_loss"], total_steps)
                logger.log_scalar("Metrics/Alpha_Value", loss_dict["alpha"], total_steps)

            if terminated or truncated:
                break

        # 记录宏观情景指标 (每局记录)
        logger.log_scalar("Reward/Episode_Reward", episode_reward, episode)
        logger.log_scalar("Metrics/Episode_Steps", episode_steps, episode)  # 存活时间是极佳的性能指标

        print(
            f"Episode: {episode + 1}/{max_episodes} | 存活步数: {episode_steps} | 总步数: {total_steps} | 回报: {episode_reward:.2f}")

        # 定期保存检查点 (防止中途断电或崩溃)
        if (episode + 1) % 100 == 0:
            checkpoint_path = os.path.join(model_save_dir, f"sac_ep{episode+1}.pth")
            agent.save_model(checkpoint_path)

    # 最终完整保存
    final_path = os.path.join(model_save_dir, "sac_highway_final.pth")
    agent.save_model(final_path)

    env.close()
    logger.close()
    print("训练主循环结束！")


if __name__ == "__main__":
    main()