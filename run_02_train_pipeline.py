"""
阶段二与三：自动化训练中央调度流水线 (Automated Training Pipeline)

此模块是整个 RL Foundation 项目的最高层控制中枢。
它的核心作用是将“数据采集”、“离线预训练(BC)”和“在线微调(Online RL)”三大模块无缝拼装成一个自动化流水线。
为了满足高强度的科研实验需求，它内置了三种工作模式：
1. 冒烟测试 (SMOKE_TEST)：用于修改代码后极速验证流程是否跑通，防止漫长训练在一开始就报错。
2. 单次运行 (SINGLE)：用于跑定稿的最佳参数，专注出图。
3. 通宵挂机 (OVERNIGHT)：科研利器，睡前设定参数矩阵和明早的起床时间，显卡会自动跑完所有消融实验并按时关机收工。
"""

import os
import gc
import torch
import time
from datetime import datetime

# 导入流水线的各个子模块
from run_01_collect_data import collect_expert_data
from runners.train_offline_bc import train_diffusion_bc
from runners.train_online_diff import train_online_diffusion


def clear_gpu_memory():
    """
    强制清理 PyTorch 显存垃圾，防止 OOM (Out of Memory)。
    在执行多个连续的强化学习实验（尤其是非常吃显存的 Diffusion 模型）时，
    旧模型的计算图很容易残留在显存中，这个函数会在每次实验切换时给显卡做一次深度清理。
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_user_configuration():
    """
    终端交互逻辑：在脚本启动时，通过一问一答的方式获取用户的运行模式与预期截止时间。
    设计了完善的容错机制，防止用户输入非法时间格式导致程序崩溃。
    """
    print("=" * 60)
    print("🤖 欢迎使用 RL Foundation 自动化训练调度系统")
    print("=" * 60)
    print("请选择运行模式:")
    print("  [0] 快速冒烟测试 (SMOKE_TEST) - 极速跑几局，用于验证代码是否会崩溃。")
    print("  [1] 单次运行模式 (SINGLE) - 跑完预设的单版配置后即刻停止。")
    print("  [2] 通宵挂机模式 (OVERNIGHT) - 自动循环执行参数矩阵，到达设定时间后安全停止。")

    run_mode = None
    while True:
        choice = input("👉 请输入选择 (0, 1 或 2): ").strip()
        if choice == '0':
            run_mode = "SMOKE_TEST"
            break
        elif choice == '1':
            run_mode = "SINGLE"
            break
        elif choice == '2':
            run_mode = "OVERNIGHT"
            break
        else:
            print("⚠️ 输入无效，请按 0, 1 或 2 进行选择。")

    target_time = None
    if run_mode == "OVERNIGHT":
        print("\n" + "-" * 60)
        print("请设置通宵挂机的截止时间。")
        print("格式示例: 2026-04-05 08:00")
        while True:
            time_str = input("👉 请输入截止时间: ").strip()
            try:
                # 将用户输入的字符串解析为真正的 datetime 时间对象
                target_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                if target_time <= datetime.now():
                    print("⚠️ 截止时间必须晚于当前时间，请重新输入。")
                else:
                    print(f"✅ 截止时间已锁定为: {target_time.strftime('%Y-%m-%d %H:%M:00')}")
                    break
            except ValueError:
                print("⚠️ 时间格式解析失败！请严格按照 YYYY-MM-DD HH:MM 格式输入 (注意空格和横杠)。")

    return run_mode, target_time


if __name__ == "__main__":
    # ==========================================
    # ⚙️ 终端交互与基础全局配置
    # ==========================================
    RUN_MODE, TARGET_END_TIME = get_user_configuration()

    # 全局数据路径配置 (控制是否复用之前辛苦跑出来的专家数据)
    REUSE_DATA = True
    EXISTING_DATA_PATH = "data/expert_data/dataset_v5_20260404_035105/expert_transitions.npz"
    V5_MODEL_PATH = "outputs/models/highway-v0_SAC_20260330_135449/sac_highway_final.pth"

    # ==========================================
    # 1. 统一的数据流准备 (Data Preparation)
    # ==========================================
    # 在所有实验开始前，先确保我们有充足的专家数据
    print("\n" + "=" * 60)
    if REUSE_DATA and os.path.exists(EXISTING_DATA_PATH):
        print(f"📦 阶段一: 复用已有专家数据 -> {EXISTING_DATA_PATH}")
        data_path = EXISTING_DATA_PATH
    else:
        print("🚀 阶段一: 重新采集专家数据...")
        # 如果没有历史数据，就现场召唤 SAC 专家跑出 5 万步的数据集
        data_path = collect_expert_data(
            model_path=V5_MODEL_PATH,
            env_name="highway-v0",
            target_transitions=50000
        )
    clear_gpu_memory()

    # ==========================================
    # 2. 路由分发逻辑 (Routing)
    # 根据用户在终端的选择，执行不同的训练流水线
    # ==========================================

    if RUN_MODE == "SMOKE_TEST":
        # ---------------------------------------------------
        # 模式 0: 快速冒烟测试
        # ---------------------------------------------------
        print("\n" + "🛡️" * 30)
        print("启动 [快速冒烟测试] 模式 - 仅用于验证代码畅通")
        print("🛡️" * 30)

        # 极速参数：仅跑 2 个 Epoch 和 5 局游戏，通常两分钟内就能跑完
        smoke_config = {"name": "Exp_0_Smoke_Test", "bc_epochs": 2, "q_weight": 0.05, "lr": 3e-4, "episodes": 5}
        print(f"📊 运行参数: {smoke_config}")

        # 测试离线预训练管线
        pretrained_model_path = train_diffusion_bc(
            data_path=data_path,
            num_epochs=smoke_config["bc_epochs"],
            batch_size=256,
            learning_rate=smoke_config["lr"]
        )
        clear_gpu_memory()

        # 测试在线微调管线
        train_online_diffusion(
            pretrained_actor_path=pretrained_model_path,
            expert_data_path=data_path,
            max_episodes=smoke_config["episodes"],
            batch_size=256,
            q_weight=smoke_config["q_weight"],
            lr=smoke_config["lr"]
        )
        print("\n✅ 冒烟测试圆满结束！所有管线畅通无阻，您可以放心启动 OVERNIGHT 模式了。")

    elif RUN_MODE == "SINGLE":
        # ---------------------------------------------------
        # 模式 1: 单次精调运行
        # ---------------------------------------------------
        print("\n" + "⚡" * 30)
        print("启动 [单次运行] 模式")
        print("⚡" * 30)

        # 单次运行的特定参数 (当前锁定为实验四：长跑稳定验证)
        single_config = {
            "name": "Exp_4_Stable_Long",
            "bc_epochs": 80,  # 增加专家预训练轮次，打好基本功
            "q_weight": 0.05,  # 标准 Q 引导权重
            "lr": 1e-4,  # 🚨 降低学习率，追求更平稳的长期收敛
            "episodes": 500  # 延长在线训练局数
        }
        print(f"📊 运行参数: {single_config}")

        # 执行离线 BC
        pretrained_model_path = train_diffusion_bc(
            data_path=data_path,
            num_epochs=single_config["bc_epochs"],
            batch_size=256,
            learning_rate=single_config["lr"]
        )
        clear_gpu_memory()

        # 执行在线微调 (确保环境是平滑过渡的 algo="diff" 模式)
        train_online_diffusion(
            pretrained_actor_path=pretrained_model_path,
            expert_data_path=data_path,
            max_episodes=single_config["episodes"],
            batch_size=256,
            q_weight=single_config["q_weight"],
            lr=single_config["lr"]
        )
        print("\n✅ 单次运行任务圆满结束！")

    elif RUN_MODE == "OVERNIGHT":
        # ---------------------------------------------------
        # 模式 2: 通宵参数矩阵扫图
        # ---------------------------------------------------
        print("\n" + "🌟" * 30)
        print("启动 [通宵挂机] 模式 - 参数矩阵轮询")
        print("🌟" * 30)

        # 定义消融实验的参数矩阵 (Ablation Matrix)
        experiment_configs = [
            # 实验组 1: 微弱引导 (Baseline)
            # 偏保守，主要依靠 BC 模仿专家，Q 网络只提供极其微弱的避障建议
            {"name": "Exp_1_Gentle_Q", "bc_epochs": 50, "q_weight": 0.01, "lr": 3e-4, "episodes": 400},

            # 实验组 2: 标准引导 (推荐配置)
            # 模仿与自主学习的平衡点，最有可能跑出高均速的组合
            {"name": "Exp_2_Standard_Q", "bc_epochs": 50, "q_weight": 0.05, "lr": 3e-4, "episodes": 400},

            # 实验组 3: 强力引导 (压力测试)
            # 给 Critic 更大的话语权，探究平滑环境下的 Critic 是否会导致策略崩坏 (过估计陷阱)
            {"name": "Exp_3_Strong_Q", "bc_epochs": 50, "q_weight": 0.10, "lr": 3e-4, "episodes": 400},

            # 实验组 4: 降学习率长跑 (稳定测试)
            # 用更低的学习率和更长的时间，探究算法的理论上限
            {"name": "Exp_4_Stable_Long", "bc_epochs": 80, "q_weight": 0.05, "lr": 1e-4, "episodes": 500},
        ]

        exp_index = 0
        total_exps = len(experiment_configs)

        # 核心挂机循环：只要没到明早设定的时间，就一直干活
        while datetime.now() < TARGET_END_TIME:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n⏰ 当前时间: {current_time} | 目标结束时间: {TARGET_END_TIME.strftime('%Y-%m-%d %H:%M:%S')}")

            # 如果预设的矩阵跑完了但天还没亮，就复制最后一组参数继续跑（加长版），榨干显卡价值
            if exp_index < total_exps:
                config = experiment_configs[exp_index]
            else:
                config = experiment_configs[-1].copy()
                config["name"] = f"Exp_Extra_{exp_index}"
                config["episodes"] += 100

            print(f"\n🚀 开始执行实验组: {config['name']} | 参数: {config}")

            # 执行该组参数的离线 BC
            pretrained_model_path = train_diffusion_bc(
                data_path=data_path,
                num_epochs=config["bc_epochs"],
                batch_size=256,
                learning_rate=config["lr"]
            )
            clear_gpu_memory()

            # 执行该组参数的在线微调
            train_online_diffusion(
                pretrained_actor_path=pretrained_model_path,
                expert_data_path=data_path,
                max_episodes=config["episodes"],
                batch_size=256,
                q_weight=config["q_weight"],
                lr=config["lr"]
            )
            clear_gpu_memory()

            print(f"✅ 实验组 {config['name']} 彻底完成！")
            exp_index += 1
            time.sleep(10)  # 跑完一组休息 10 秒，让显存飞一会儿

        print("\n" + "=" * 60)
        print(f"⏰ 设定时间 ({TARGET_END_TIME.strftime('%H:%M')}) 已到达！通宵脚本顺利收工。")
        print("=" * 60)