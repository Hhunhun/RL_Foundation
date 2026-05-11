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
import inspect
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

    print("\n" + "-" * 60)
    print("请选择目标训练环境:")
    print("  [H] Highway 环境 (highway-v0)")
    print("  [M] Merge 环境 (merge-v0)")
    print("  [R] Racetrack 环境 (racetrack-v0)")
    
    target_env = None
    while True:
        env_choice = input("👉 请输入选择 (H, M 或 R，默认 H): ").strip().upper()
        if env_choice == 'M':
            target_env = "merge-v0"
            break
        elif env_choice == 'R':
            target_env = "racetrack-v0"
            break
        elif env_choice == 'H' or env_choice == '':
            target_env = "highway-v0"
            break
        else:
            print("⚠️ 输入无效，请按 H 或 M 进行选择。")

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

    return run_mode, target_time, target_env


if __name__ == "__main__":
    # ==========================================
    # ⚙️ 终端交互与基础全局配置
    # ==========================================
    RUN_MODE, TARGET_END_TIME, TARGET_ENV = get_user_configuration()

    # 全局数据路径配置 (控制是否复用之前辛苦跑出来的专家数据)
    REUSE_DATA = True

    # 🚨 动态配置：根据选择的环境切换专家模型和数据集路径
    # 🚨🚨🚨 重要：请在应用新的文件夹结构后，手动更新以下 SAFE_MODEL_PATH 和 EXPERT_DATA_PATH 的值！
    # 它们需要指向新结构下的正确路径，例如：
    # outputs/merge-v0/models/SAC_YYYYMMDD_HHMMSS/sac_merge_final.pth
    # data/expert_data/merge-v0/dataset_base_YYYYMMDD_HHMMSS/expert_transitions.npz
    if TARGET_ENV == "merge-v0":
        # ==========================================
        # 🚨 [Merge 配置区] 同时定义单专家和混合专家的数据集路径
        # ==========================================
        SINGLE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "merge-v0", "dataset_M4_mode1_20260423_154904", "expert_transitions.npz")
        # 🚨 请将 YOUR_MIXED_DATASET_HERE 替换为您实际生成的混合数据集的文件夹名
        MIXED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "merge-v0", "YOUR_MIXED_DATASET_HERE", "expert_transitions_mixed.npz")
        
        # 默认回退路径 (供 SMOKE_TEST 和 SINGLE 模式兜底使用)
        SAFE_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "merge-v0", "models", "SAC_M4_Safety_First_20260420_170911", "sac_merge_final.pth")
        SAFE_ENV_CONFIG = {"reward_speed_range": [15, 25]}
        EXPERT_DATA_PATH = SINGLE_DATA_PATH
        TARGET_DATA_STEPS = 20000 # 按照最新要求，采集 1-2 万步即可
    elif TARGET_ENV == "racetrack-v0":
        # ==========================================
        # 🚨 [Racetrack 配置区] 
        # 同时定义单专家和混合专家的数据集路径，方便通宵模式一口气跑完
        # ==========================================
        SINGLE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "racetrack-v0", "dataset_R05_mode1_20260506_011817", "expert_transitions.npz")
        MIXED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "racetrack-v0", "dataset_mixed_0.8R05_0.2R01_20260506_142446", "expert_transitions_mixed_0.8R05_0.2R01.npz")
        
        # 默认回退路径 (供 SMOKE_TEST 和 SINGLE 模式兜底使用)
        SAFE_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "racetrack-v0", "models", "SAC_R05_SAC_Smooth_Racing_20260505_131614", "sac_racetrack_final.pth")
        EXPERT_DATA_PATH = SINGLE_DATA_PATH
            
        SAFE_ENV_CONFIG = {"reward_speed_range": [15, 25]}
        
        TARGET_DATA_STEPS = 50000
    else:
        # Highway 环境默认路径
        SAFE_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "highway-v0", "models", "SAC_20260330_135449", "sac_highway_final.pth")
        SAFE_ENV_CONFIG = None
        EXPERT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "highway-v0", "dataset_smart_mixed_90_10_20260413_031136", "expert_transitions_smart_90_10.npz")
        TARGET_DATA_STEPS = 50000
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
        # 如果没有历史数据，就现场召唤 SAC 专家跑出数据集
        data_path = collect_expert_data(
            model_path=SAFE_MODEL_PATH,
            env_name=TARGET_ENV,
            target_transitions=TARGET_DATA_STEPS,
            env_config=SAFE_ENV_CONFIG
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
        test_name = "DM0_Smoke_Test" if TARGET_ENV == "merge-v0" else "DH0_Smoke_Test"
        smoke_config = {"name": test_name, "bc_epochs": 2, "q_weight": 0.05, "lr": 3e-4, "max_steps": 1000}
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"DiffSAC_{smoke_config['name']}_{current_time}"
        print(f"📊 运行参数: {smoke_config}")

        # 测试离线预训练管线
        bc_kwargs = {
            "data_path": data_path,
            "env_name": TARGET_ENV,
            "num_epochs": smoke_config["bc_epochs"],
            "batch_size": 256,
            "learning_rate": smoke_config["lr"]
        }
        if "run_name" in inspect.signature(train_diffusion_bc).parameters:
            bc_kwargs["run_name"] = run_name
        pretrained_model_path = train_diffusion_bc(**bc_kwargs)
        clear_gpu_memory()

        # 测试在线微调管线
        train_online_diffusion(
            pretrained_actor_path=pretrained_model_path,
            expert_data_path=data_path,
            env_name=TARGET_ENV,
            max_steps=smoke_config["max_steps"],
            batch_size=256,
            q_weight=smoke_config["q_weight"],
            lr=smoke_config["lr"],
            run_name=run_name
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
        single_name = "DM4_Stable_Long" if TARGET_ENV == "merge-v0" else "DH4_Stable_Long"
        single_config = {
            "name": single_name,
            "bc_epochs": 80,  # 增加专家预训练轮次，打好基本功
            "q_weight": 0.05,  # 标准 Q 引导权重
            "lr": 1e-4,  # 🚨 降低学习率，追求更平稳的长期收敛
            "max_steps": 100000  # 统一使用环境物理步数
        }
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"DiffSAC_{single_config['name']}_{current_time}"
        print(f"📊 运行参数: {single_config}")

        # 执行离线 BC
        bc_kwargs = {
            "data_path": data_path, "env_name": TARGET_ENV,
            "num_epochs": single_config["bc_epochs"], "batch_size": 256,
            "learning_rate": single_config["lr"]
        }
        if "run_name" in inspect.signature(train_diffusion_bc).parameters:
            bc_kwargs["run_name"] = run_name
        pretrained_model_path = train_diffusion_bc(**bc_kwargs)
        clear_gpu_memory()

        # 执行在线微调 (确保环境是平滑过渡的 algo="diff" 模式)
        train_online_diffusion(
            pretrained_actor_path=pretrained_model_path,
            expert_data_path=data_path,
            env_name=TARGET_ENV,
            max_steps=single_config["max_steps"],
            batch_size=256,
            q_weight=single_config["q_weight"],
            lr=single_config["lr"],
            run_name=run_name
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
            #{"name": "DH1_Gentle_Q", "bc_epochs": 50, "q_weight": 0.01, "lr": 3e-4, "max_steps": 100000},
            #{"name": "DH2_Standard_Q", "bc_epochs": 50, "q_weight": 0.05, "lr": 3e-4, "max_steps": 100000},
            #{"name": "DH3_Strong_Q", "bc_epochs": 50, "q_weight": 0.10, "lr": 3e-4, "max_steps": 100000},
            #{"name": "DH4_Stable_Long", "bc_epochs": 80, "q_weight": 0.05, "lr": 1e-4, "max_steps": 125000},
        # ==========================================
        # 第二期消融实验矩阵 (探寻极简与极致稳健)
        # ==========================================
            #{"name": "DH5_Micro_Q", "bc_epochs": 50, "q_weight": 0.005, "lr": 3e-4, "max_steps": 100000},
            #{"name": "DH6_Bulletproof_BC", "bc_epochs": 120, "q_weight": 0.05, "lr": 3e-4, "max_steps": 100000},
            #{"name": "DH7_Frozen_Finetune", "bc_epochs": 80, "q_weight": 0.005, "lr": 5e-5, "max_steps": 125000},
            #{"name": "DH8_Zero_Q_Control", "bc_epochs": 50, "q_weight": 0.0, "lr": 3e-4, "max_steps": 100000},
        # ==========================================
        # 第三期消融实验矩阵 (探寻 SOTA 的绝对极限)
        # ==========================================
            #{"name": "DH9_Ultimate_Safe_SOTA", "bc_epochs": 120, "q_weight": 0.005, "lr": 5e-5, "max_steps": 125000},
            #{"name": "DH10_Accelerated_Finetune", "bc_epochs": 80, "q_weight": 0.005, "lr": 1e-4, "max_steps": 125000},
            #{"name": "DH11_Ultra_Micro_Q", "bc_epochs": 50, "q_weight": 0.001, "lr": 3e-4, "max_steps": 100000},
            #{"name": "DH12_Frozen_Marathon", "bc_epochs": 80, "q_weight": 0.005, "lr": 5e-5, "max_steps": 200000},
        # ==========================================
        # 第四期消融实验矩阵 (黄金融合与终极天花板)
        # 核心策略：废弃极低学习率，融合最强先验 (BC=120) 与最优微导 (q=0.01~0.001)
        # ==========================================
            #{"name": "DH13_Unbreakable_SOTA", "bc_epochs": 120, "q_weight": 0.001, "lr": 3e-4, "max_steps": 100000},
            #{"name": "DH14_Thick_Shield_Gentle_Q", "bc_epochs": 120, "q_weight": 0.01, "lr": 3e-4, "max_steps": 100000},
            #{"name": "DH15_Deep_BC_Control", "bc_epochs": 120, "q_weight": 0.0, "lr": 3e-4, "max_steps": 100000},
            #{"name": "DH16_Ultra_Micro_Marathon", "bc_epochs": 50, "q_weight": 0.001, "lr": 3e-4, "max_steps": 150000},
        # ==========================================
        # 第五期实验矩阵 (混合数据集突围测试)
        # 核心目的：验证混合流形能否在保持高存活率的同时，打破 22 m/s 均速天花板
        # ==========================================
            # {"name": "DH17_Mixed_BC_Control", "bc_epochs": 50, "q_weight": 0.0, "lr": 3e-4, "max_steps": 100000},
            # {"name": "DH18_Mixed_Ultra_Micro", "bc_epochs": 50, "q_weight": 0.001, "lr": 3e-4, "max_steps": 100000},
            # {"name": "DH19_Mixed_Thicker_Base", "bc_epochs": 80, "q_weight": 0.001, "lr": 3e-4, "max_steps": 100000},
            # {"name": "DH20_Mixed_Marathon", "bc_epochs": 50, "q_weight": 0.001, "lr": 3e-4, "max_steps": 150000},
        ]

        # ==========================================
        # Diff-SAC 针对 Merge 环境的初始探索矩阵
        # ==========================================
        merge_experiment_configs = [
            # === 第一期：单专家消融 (Single Expert) ===
            {"name": "DM01_Pure_BC", "bc_epochs": 50, "q_weight": 0.0, "lr": 3e-4, "max_steps": 100000, "data_path": SINGLE_DATA_PATH},
            {"name": "DM02_Micro_Q", "bc_epochs": 50, "q_weight": 0.005, "lr": 3e-4, "max_steps": 100000, "data_path": SINGLE_DATA_PATH},
            {"name": "DM03_Standard_Q", "bc_epochs": 50, "q_weight": 0.01, "lr": 3e-4, "max_steps": 100000, "data_path": SINGLE_DATA_PATH},
            {"name": "DM04_Strong_Q", "bc_epochs": 50, "q_weight": 0.05, "lr": 3e-4, "max_steps": 100000, "data_path": SINGLE_DATA_PATH},

            # === 第二期：混合专家突围 (Mixed Experts) ===
            # {"name": "DM05_Mixed_BC", "bc_epochs": 50, "q_weight": 0.0, "lr": 3e-4, "max_steps": 100000, "data_path": MIXED_DATA_PATH},
            # {"name": "DM06_Mixed_Micro_Q", "bc_epochs": 50, "q_weight": 0.01, "lr": 3e-4, "max_steps": 100000, "data_path": MIXED_DATA_PATH},
            # {"name": "DM07_Mixed_Standard_Q", "bc_epochs": 50, "q_weight": 0.1, "lr": 3e-4, "max_steps": 100000, "data_path": MIXED_DATA_PATH},
            # {"name": "DM08_Mixed_Strong_Q", "bc_epochs": 50, "q_weight": 1.0, "lr": 3e-4, "max_steps": 100000, "data_path": MIXED_DATA_PATH},
        ]

        # ==========================================
        # Diff-SAC 针对 Racetrack 环境的初始探索矩阵
        # 核心目的：验证纯粹的 R055 稳健底座（15.6m/s），能否通过 Q 引导打破自身的物理天花板
        # ==========================================
        racetrack_experiment_configs = [
            # === 第一期：单专家消融 (已完成) ===
            #{"name": "DR01_Pure_BC", "bc_epochs": 50, "q_weight": 0.0, "lr": 3e-4, "max_steps": 200000, "data_path": SINGLE_DATA_PATH},
            #{"name": "DR02_Micro_Q", "bc_epochs": 50, "q_weight": 0.01, "lr": 3e-4, "max_steps": 200000, "data_path": SINGLE_DATA_PATH},
            #{"name": "DR03_Standard_Q", "bc_epochs": 50, "q_weight": 0.1, "lr": 3e-4, "max_steps": 200000, "data_path": SINGLE_DATA_PATH},
            #{"name": "DR04_Strong_Q", "bc_epochs": 50, "q_weight": 1.0, "lr": 3e-4, "max_steps": 200000, "data_path": SINGLE_DATA_PATH},

            # === 第二期：混合专家突围 (Mixed Experts) ===
            #{"name": "DR05_Mixed_BC", "bc_epochs": 50, "q_weight": 0.0, "lr": 3e-4, "max_steps": 200000, "data_path": MIXED_DATA_PATH},
            #{"name": "DR06_Mixed_Micro_Q", "bc_epochs": 50, "q_weight": 0.01, "lr": 3e-4, "max_steps": 200000, "data_path": MIXED_DATA_PATH},
            {"name": "DR07_Mixed_Standard_Q", "bc_epochs": 50, "q_weight": 0.1, "lr": 3e-4, "max_steps": 200000, "data_path": MIXED_DATA_PATH},
            {"name": "DR08_Mixed_Strong_Q", "bc_epochs": 50, "q_weight": 0.1, "lr": 3e-4, "max_steps": 200000, "data_path": MIXED_DATA_PATH},
        ]
        # 动态判定：根据终端输入，无缝切换任务队列
        if TARGET_ENV == "merge-v0": active_configs = merge_experiment_configs
        elif TARGET_ENV == "racetrack-v0": active_configs = racetrack_experiment_configs
        else: active_configs = experiment_configs

        exp_index = 0
        total_exps = len(active_configs)

        # 🚨 核心修复：跑完即停，绝不恋战。删除了之前的扩展逻辑。
        while exp_index < total_exps:
            # 安全检查：时间到了立即安全退出
            if TARGET_END_TIME and datetime.now() >= TARGET_END_TIME:
                print(f"⏰ 到达设定的截止时间 {TARGET_END_TIME.strftime('%Y-%m-%d %H:%M:%S')}，正在安全终止后续实验...")
                break

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            config = active_configs[exp_index]
            
            # 聚合实验代号生成
            run_name = f"DiffSAC_{config['name']}_{current_time.replace('-', '').replace(':', '').replace(' ', '_')}"

            print(f"\n==================================================")
            print(f"🚀 [进度 {exp_index + 1}/{total_exps}] 开始执行实验组: {config['name']}")
            print(f"📁 聚合存档代号: {run_name}")
            print(f"⏰ 当前时间: {current_time}")
            
            # 动态获取当前配置对应的数据集，如果没有指定则回退到全局默认 data_path
            current_data_path = config.get("data_path", data_path)
            print(f"📦 数据集路径: {current_data_path}")
            print(f" 参数配置: {config}")
            print(f"==================================================")
            
            if not os.path.exists(current_data_path):
                print(f"❌ 找不到对应的数据集: {current_data_path}，跳过此实验！")
                exp_index += 1
                continue

            bc_kwargs = {
                "data_path": current_data_path, "env_name": TARGET_ENV,
                "num_epochs": config["bc_epochs"], "batch_size": 256, "learning_rate": config["lr"]
            }
            if "run_name" in inspect.signature(train_diffusion_bc).parameters:
                bc_kwargs["run_name"] = run_name
            pretrained_model_path = train_diffusion_bc(**bc_kwargs)
            clear_gpu_memory()

            # 执行该组参数的在线微调
            train_online_diffusion(
                pretrained_actor_path=pretrained_model_path,
                expert_data_path=current_data_path,
                env_name=TARGET_ENV,
                max_steps=config["max_steps"],
                batch_size=256,
                q_weight=config["q_weight"],
                lr=config["lr"],
                run_name=run_name
            )
            clear_gpu_memory()

            print(f"✅ 实验组 {config['name']} 彻底完成！")
            exp_index += 1
            time.sleep(10)  # 跑完一组休息 10 秒，让显存飞一会儿

        print("\n" + "=" * 60)
        print("🎯 通宵跑参调度器任务结束。所有规划的实验已成功跑完或安全终止。")
        print("=" * 60)