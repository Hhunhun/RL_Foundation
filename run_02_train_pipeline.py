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

# 🚨 核心修复：锁定项目根目录绝对路径，防止子脚本寻找路径时发生漂移
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


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
    print("🤖 使用 RL Foundation 自动化训练调度系统")
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

    # 🚨 修复：使用 os.path.join 和 PROJECT_ROOT 构建绝对路径
    #EXPERT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "dataset_v5_20260404_035105", "expert_transitions.npz")
    V5_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "highway-v0_SAC_20260330_135449", "sac_highway_final.pth")

    # 🚨 进阶：将专家数据路径指向昨天跑出的“v5+v6 混合数据集”
    EXPERT_DATA_PATH = os.path.join(PROJECT_ROOT,"data","expert_data","dataset_mixed_80_20_20260410_031547", "expert_transitions_mixed_80_20.npz")

    # ==========================================
    # 1. 统一的数据流准备 (Data Preparation)
    # ==========================================
    # 在所有实验开始前，先确保我们有充足的专家数据
    print("\n" + "=" * 60)
    if REUSE_DATA and os.path.exists(EXPERT_DATA_PATH):
        print(f"📦 阶段一: 复用已有专家数据 -> {EXPERT_DATA_PATH}")
        data_path = EXPERT_DATA_PATH
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
        # ==========================================
        # 第一期消融实验矩阵
        # ==========================================
            # 实验组 1: 微弱引导 (Baseline)
            # 偏保守，主要依靠 BC 模仿专家，Q 网络只提供极其微弱的避障建议
            #{"name": "Exp_1_Gentle_Q", "bc_epochs": 50, "q_weight": 0.01, "lr": 3e-4, "episodes": 400},

            # 实验组 2: 标准引导 (推荐配置)
            # 模仿与自主学习的平衡点，最有可能跑出高均速的组合
            #{"name": "Exp_2_Standard_Q", "bc_epochs": 50, "q_weight": 0.05, "lr": 3e-4, "episodes": 400},

            # 实验组 3: 强力引导 (压力测试)
            # 给 Critic 更大的话语权，探究平滑环境下的 Critic 是否会导致策略崩坏 (过估计陷阱)
            #{"name": "Exp_3_Strong_Q", "bc_epochs": 50, "q_weight": 0.10, "lr": 3e-4, "episodes": 400},

            # 实验组 4: 降学习率长跑 (稳定测试)
            # 用更低的学习率和更长的时间，探究算法的理论上限
            #{"name": "Exp_4_Stable_Long", "bc_epochs": 80, "q_weight": 0.05, "lr": 1e-4, "episodes": 500},
        # ==========================================
        # 第二期消融实验矩阵 (探寻极简与极致稳健)
        # ==========================================
            # 实验组 5: 极微引导 (Micro-Q)
            # 既然 0.01 依然会引起震荡，我们直接将 Critic 的话语权再砍一半。
            # 探究多小的 Q 值能在不破坏专家安全底线的情况下，依然起到提速作用。
            #{"name": "Exp_5_Micro_Q", "bc_epochs": 50, "q_weight": 0.005, "lr": 3e-4, "episodes": 400},

            # 实验组 6: 铁壁底座 (Overfit Prior)
            # 疯狂增加离线预训练轮次（120 轮），让 Diffusion Actor 对专家动作产生“肌肉记忆”（过拟合）。
            # 看看极其坚固的先验底座，能否抵御住标准 Q 值 (0.05) 的冲击。
            #{"name": "Exp_6_Bulletproof_BC", "bc_epochs": 120, "q_weight": 0.05, "lr": 3e-4, "episodes": 400},

            # 实验组 7: 冰封微调 (Frozen Fine-tuning)
            # 结合最稳妥的参数：扎实的预训练 + 极小的 Q 引导 + 极低的学习率。
            # 就像用小刀雕刻冰雕，一点一点地逼近物理极限，这是最有可能诞生 SOTA 的神仙组合。
            #{"name": "Exp_7_Frozen_Finetune", "bc_epochs": 80, "q_weight": 0.005, "lr": 5e-5, "episodes": 500},

            # 实验组 8: 零引导对照组 (Zero-Q Control)
            # 极其重要的学术对照组！彻底关闭 Critic 的引导 (q=0.0)，在线阶段完全退化为基于混合经验池的自我模仿学习。
            # 用于在论文中证明：我们加入强化学习 (RL) 到底有没有用？是不是光靠单纯的 BC 就能达到这个分数？
            #{"name": "Exp_8_Zero_Q_Control", "bc_epochs": 50, "q_weight": 0.0, "lr": 3e-4, "episodes": 400},
        # ==========================================
        # 第三期消融实验矩阵 (探寻 SOTA 的绝对极限)
        # ==========================================
            # 实验组 9: 终极防御底座 (Deep BC + Frozen Finetune)
            # Exp_6 证明了即使 120 轮 BC 也挡不住 q=0.05 的破坏。
            # 那如果我们把最厚的装甲 (bc=120) 和最温柔的刀 (q=0.005, lr=5e-5) 结合呢？
            # 探究最牢固的先验底座是否能让微调过程的方差降到绝对的 0。
            #{"name": "Exp_9_Ultimate_Safe_SOTA", "bc_epochs": 120, "q_weight": 0.005, "lr": 5e-5, "episodes": 500},

            # 实验组 10: 加速冰封 (Moderate LR + Micro Q)
            # Exp_7 的 lr=5e-5 极其稳定，但可能收敛太慢。
            # 我们把学习率稍微提一点点到 1e-4（Exp_4 证明它在 q=0.05 时会崩，但在 q=0.005 下安全吗？）。
            # 测试在安全 Q 权重下，网络更新步长的安全上限。
            #{"name": "Exp_10_Accelerated_Finetune", "bc_epochs": 80, "q_weight": 0.005, "lr": 1e-4, "episodes": 500},

            # 实验组 11: 极限微丝引导 (Ultra-Micro Q)
            # 探索 Exp_5 (q=0.005) 和 Exp_8 (q=0.0 纯 BC) 之间的地带。
            # q=0.001 是一个极小的值，它到底是一缕能缓慢提速的清风，还是弱到跟完全关闭 (0.0) 没区别？
            #{"name": "Exp_11_Ultra_Micro_Q", "bc_epochs": 50, "q_weight": 0.001, "lr": 3e-4, "episodes": 400},

            # 实验组 12: 冰封马拉松 (The Marathon)
            # 既然 Exp_7 (q=0.005, lr=5e-5) 在 500 局结束时 Q 值还在稳步上升（没有平波），
            # 说明它还没碰到真正的天花板！我们直接给它 800 局的超长时间。
            # 探究：它是会最终收敛到一个超越所有人的史诗级高分，还是在长期积累后发生“延迟崩溃”？
            #{"name": "Exp_12_Frozen_Marathon", "bc_epochs": 80, "q_weight": 0.005, "lr": 5e-5, "episodes": 800},

        # ==========================================
        # 第四期消融实验矩阵 (黄金融合与终极天花板)
        # 核心策略：废弃极低学习率，融合最强先验 (BC=120) 与最优微导 (q=0.01~0.001)
        # ==========================================
            # 实验组 13: 终极无坚不摧 (Heavy BC + Ultra-Micro Q)
            # 结合 Exp_6 的“铁壁底座”与 Exp_11 的“最强微丝引导”。
            # 用 120 轮预训练筑起绝对安全的防线，然后用极其轻柔的 q=0.001 进行提速。
            # 这是理论上既能 100% 存活，又能打破均速上限的最优解。
            #{"name": "Exp_13_Unbreakable_SOTA", "bc_epochs": 120, "q_weight": 0.001, "lr": 3e-4, "episodes": 400},

            # 实验组 14: 厚甲利刃 (Heavy BC + Gentle Q)
            # 结合 Exp_6 的“铁壁底座”与 第一期冠军 Exp_1 的“微弱引导”。
            # q=0.01 的提速动力更足，我们看看 120 轮的厚重底座能否完美抗住这股更强的探索冲动。
            #{"name": "Exp_14_Thick_Shield_Gentle_Q", "bc_epochs": 120, "q_weight": 0.01, "lr": 3e-4, "episodes": 400},

            # 实验组 15: 纯粹克隆的物理极限 (Absolute BC Upper Bound)
            # Exp_8 (BC=50, q=0.0) 表现很好，那如果我们不加任何 RL，纯靠 120 轮死记硬背呢？
            # 这是一个极其关键的学术对照组，用于对比 Exp_13 和 14，证明在同等底座厚度下，RL 引导依然不可或缺。
            #{"name": "Exp_15_Deep_BC_Control", "bc_epochs": 120, "q_weight": 0.0, "lr": 3e-4, "episodes": 400},

            # 实验组 16: 微丝引导马拉松 (Ultra-Micro Q Marathon)
            # Exp_11 (q=0.001, BC=50) 是第三期的冠军。
            # 我们保持它的完美参数，但给它更长的在线交互时间（600局），探究极微弱引导在长期运行下会不会发生延迟崩溃，还是会爬上巅峰。
            #{"name": "Exp_16_Ultra_Micro_Marathon", "bc_epochs": 50, "q_weight": 0.001, "lr": 3e-4, "episodes": 600},

        # ==========================================
        # 第五期实验矩阵 (混合数据集突围测试)
        # 核心目的：验证混合流形能否在保持高存活率的同时，打破 22 m/s 均速天花板
        # ==========================================
            # 实验组 17: 纯混合克隆基准 (Mixed BC Control)
            # 对应之前的 Exp_8。完全关闭 Q 引导 (q=0.0)。
            # 这是极其关键的基准线！我们要看仅仅是“喂了更好的数据”，模型纯靠模仿，能否在速度上超越以前的 Exp_8。
            #{"name": "Exp_17_Mixed_BC_Control", "bc_epochs": 50, "q_weight": 0.0, "lr": 3e-4, "episodes": 400},

            # 实验组 18: 混合流形冠军 (Mixed Ultra-Micro Q)
            # 对应之前的全场最佳 Exp_11。
            # 这是我们冲击最终 SOTA 的主力军！看看在混合神仙数据的加持下，0.001 的微弱提速能否完美兑现。
            #{"name": "Exp_18_Mixed_Ultra_Micro", "bc_epochs": 50, "q_weight": 0.001, "lr": 3e-4, "episodes": 400},

            # 实验组 19: 数据容量扩充测试 (Mixed Thicker Base)
            # 这是一个新策略！因为混合数据集包含了“减速”和“极速”两种互相矛盾的动作，流形变复杂了。
            # 50 轮预训练可能背不过这么复杂的规律，所以我们把底座适度加厚到 80 轮（但避开 120 轮的死板陷阱）。
            #{"name": "Exp_19_Mixed_Thicker_Base", "bc_epochs": 80, "q_weight": 0.001, "lr": 3e-4, "episodes": 400},

            # 实验组 20: 混合马拉松 (Mixed Marathon)
            # 对应之前的 Exp_16。
            # 既然数据更丰富了，给它更长的在线交互时间（600局），看它能否彻底融会贯通，攀上均速的巅峰。
            {"name": "Exp_20_Mixed_Marathon", "bc_epochs": 50, "q_weight": 0.001, "lr": 3e-4, "episodes": 600},
        ]

        exp_index = 0
        total_exps = len(experiment_configs)

        # 🚨 核心修复：跑完即停，绝不恋战。删除了之前的扩展逻辑。
        while exp_index < total_exps:
            # 安全检查：时间到了立即安全退出
            if TARGET_END_TIME and datetime.now() >= TARGET_END_TIME:
                print(f"⏰ 到达设定的截止时间 {TARGET_END_TIME.strftime('%Y-%m-%d %H:%M:%S')}，正在安全终止后续实验...")
                break

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            config = experiment_configs[exp_index]

            print(f"\n==================================================")
            print(f"🚀 [进度 {exp_index + 1}/{total_exps}] 开始执行实验组: {config['name']}")
            print(f"⏰ 当前时间: {current_time}")
            print(f"📊 参数配置: {config}")
            print(f"==================================================")

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
        print("🎯 通宵跑参调度器任务结束。所有规划的实验已成功跑完或安全终止。")
        print("=" * 60)