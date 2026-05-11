import os
import sys
import numpy as np
import torch

# 动态将项目根目录添加到包搜索路径中，防止 ModuleNotFoundError
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from core.replay_buffer import ReplayBuffer
from algorithms.sac.sac_agent import SACAgent
from utils.logger import Logger
from envs.highway_wrapper import create_highway_env
from datetime import datetime # Added import


def main():
    env_name = 'highway-v0'
    print("正在初始化 HighwayEnv v4.0 (LQR护甲 + 步数驱动版)...")
    env = create_highway_env(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"[{env_name}] State dim: {state_dim} | Action dim: {action_dim} | Max action: {max_action}")

    # 实例化核心组件
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(2e5))
    agent = SACAgent(state_dim, action_dim, action_scale=max_action, lr=3e-4)
    
    config_name = "H1_Base_Highway"
    # 动态生成本次运行的唯一标识符
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"SAC_{config_name}_{timestamp}"

    # 使用绝对路径
    base_output_dir = os.path.join(PROJECT_ROOT, "outputs")
    logger = Logger(log_dir=os.path.join(base_output_dir, env_name, "logs"), env_name=config_name)

    model_save_dir = os.path.join(base_output_dir, env_name, "models", run_id)
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
    consecutive_quick_deaths = 0  # 新增：连续暴毙计数器

    print(f"\n🚀 引擎点火，目标 {max_steps} 步，开始自动驾驶魔鬼训练...")

    # 核心循环：以总步数为绝对衡量标准
    while total_steps < max_steps:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        c_loss_list, a_loss_list = [], []

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
                c_loss_list.append(loss_dict["critic_loss"])
                a_loss_list.append(loss_dict["actor_loss"])

                # 将高频数据推送到 TensorBoard (每步记录，利用 flush 实时落盘)
                logger.log_scalar("Loss/Critic", loss_dict["critic_loss"], total_steps)
                logger.log_scalar("Loss/Actor", loss_dict["actor_loss"], total_steps)
                logger.log_scalar("Loss/Alpha", loss_dict["alpha_loss"], total_steps)
                logger.log_scalar("Metrics/Alpha_Value", loss_dict["alpha"], total_steps)

            # 实时打印单步进度（覆盖同行，不换行）
            print(f"\r⏳ 引擎运转中... | 全局进度: {total_steps}/{max_steps} 步 | 当前局存活: {episode_steps} 步", end="")

            if terminated or truncated or episode_steps >= 1000:
                break

        episode += 1

        # 记录宏观情景指标 (每局记录)
        logger.log_scalar("Reward/Episode_Reward", episode_reward, episode)
        logger.log_scalar("Metrics/Episode_Steps", episode_steps, episode)

        avg_c_loss = np.mean(c_loss_list) if c_loss_list else 0.0
        avg_a_loss = np.mean(a_loss_list) if a_loss_list else 0.0

        # 局末打印详尽的汇总指标
        print(f"\r🏁 Episode {episode:03d} | Reward: {episode_reward:5.1f} | Steps: {episode_steps:3d} | Total: {total_steps}/{max_steps} | C_Loss: {avg_c_loss:.3f} | A_Loss: {avg_a_loss:.3f}")

        # ==========================================
        # 🛡️ 硬件级保护机制：防“1步暴毙”内存溢出
        # ==========================================
        if episode_steps <= 3:
            consecutive_quick_deaths += 1
        else:
            consecutive_quick_deaths = 0
            
        if consecutive_quick_deaths >= 5:
            import time
            import gc
            time.sleep(0.5)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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