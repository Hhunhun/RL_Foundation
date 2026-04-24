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
        # 🚨 [专家底座配置区] 支持动态适配不同的 SAC 模型
        # ==========================================
        EXPERT_CONFIGS = {
            # 之前的 M8 专家基座 (已作为次优保守策略弃用，留作历史对照)
            # "M8_Ultimate": {
            #     "model_path": os.path.join(PROJECT_ROOT, "outputs", "merge-v0", "models", "SAC_M8_Ultimate_Merge_20260421_023258", "sac_merge_final.pth"),
            #     "env_config": {"reward_speed_range": [15, 25]}
            # },
            "M4_Safety": {
                "model_path": os.path.join(PROJECT_ROOT, "outputs", "merge-v0", "models", "SAC_M4_Safety_First_20260420_170911", "sac_merge_final.pth"),
                "env_config": {"reward_speed_range": [15, 25]}
            }
            # 未来如果想用 M2，只需在这里添加:
            # "M2_Efficient": {"model_path": "...", "env_config": {"reward_speed_range": [18, 28]}}
        }
        
        ACTIVE_EXPERT = "M4_Safety" # 👉 更改此处名称，即可一键切换底层专家和对应的环境速度区间
        SAFE_MODEL_PATH = EXPERT_CONFIGS[ACTIVE_EXPERT]["model_path"]
        SAFE_ENV_CONFIG = EXPERT_CONFIGS[ACTIVE_EXPERT]["env_config"]
        
        # 👉 已为您自动填入刚刚采集完美的 2 万步数据集名称
        # EXPERT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "merge-v0", "dataset_base_20260422_014135", "expert_transitions.npz") # 之前的 M8 底座
        EXPERT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "merge-v0", "dataset_M4_mode1_20260423_154904", "expert_transitions.npz") # 最新的 M4 底座
        TARGET_DATA_STEPS = 20000 # 按照最新要求，采集 1-2 万步即可
    elif TARGET_ENV == "racetrack-v0":
        # ==========================================
        # 🚨 [Racetrack 配置区] 
        # 请在采集完 racetrack 数据后更新以下两个路径
        # ==========================================
        ACTIVE_EXPERT = "R1_Base"
        SAFE_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "racetrack-v0", "models", "SAC_R1_Base_XXXXXX", "sac_racetrack_final.pth")
        SAFE_ENV_CONFIG = {"reward_speed_range": [15, 30]}
        
        EXPERT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "expert_data", "racetrack-v0", "dataset_R1_mode1_XXXXXX", "expert_transitions.npz")
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
        smoke_config = {"name": test_name, "bc_epochs": 2, "q_weight": 0.05, "lr": 3e-4, "episodes": 5}
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
            max_episodes=smoke_config["episodes"],
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
            "episodes": 500  # 延长在线训练局数
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
            max_episodes=single_config["episodes"],
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
            #{"name": "DH1_Gentle_Q", "bc_epochs": 50, "q_weight": 0.01, "lr": 3e-4, "episodes": 400},
            #{"name": "DH2_Standard_Q", "bc_epochs": 50, "q_weight": 0.05, "lr": 3e-4, "episodes": 400},
            #{"name": "DH3_Strong_Q", "bc_epochs": 50, "q_weight": 0.10, "lr": 3e-4, "episodes": 400},
            #{"name": "DH4_Stable_Long", "bc_epochs": 80, "q_weight": 0.05, "lr": 1e-4, "episodes": 500},
        # ==========================================
        # 第二期消融实验矩阵 (探寻极简与极致稳健)
        # ==========================================
            #{"name": "DH5_Micro_Q", "bc_epochs": 50, "q_weight": 0.005, "lr": 3e-4, "episodes": 400},
            #{"name": "DH6_Bulletproof_BC", "bc_epochs": 120, "q_weight": 0.05, "lr": 3e-4, "episodes": 400},
            #{"name": "DH7_Frozen_Finetune", "bc_epochs": 80, "q_weight": 0.005, "lr": 5e-5, "episodes": 500},
            #{"name": "DH8_Zero_Q_Control", "bc_epochs": 50, "q_weight": 0.0, "lr": 3e-4, "episodes": 400},
        # ==========================================
        # 第三期消融实验矩阵 (探寻 SOTA 的绝对极限)
        # ==========================================
            #{"name": "DH9_Ultimate_Safe_SOTA", "bc_epochs": 120, "q_weight": 0.005, "lr": 5e-5, "episodes": 500},
            #{"name": "DH10_Accelerated_Finetune", "bc_epochs": 80, "q_weight": 0.005, "lr": 1e-4, "episodes": 500},
            #{"name": "DH11_Ultra_Micro_Q", "bc_epochs": 50, "q_weight": 0.001, "lr": 3e-4, "episodes": 400},
            #{"name": "DH12_Frozen_Marathon", "bc_epochs": 80, "q_weight": 0.005, "lr": 5e-5, "episodes": 800},
        # ==========================================
        # 第四期消融实验矩阵 (黄金融合与终极天花板)
        # 核心策略：废弃极低学习率，融合最强先验 (BC=120) 与最优微导 (q=0.01~0.001)
        # ==========================================
            #{"name": "DH13_Unbreakable_SOTA", "bc_epochs": 120, "q_weight": 0.001, "lr": 3e-4, "episodes": 400},
            #{"name": "DH14_Thick_Shield_Gentle_Q", "bc_epochs": 120, "q_weight": 0.01, "lr": 3e-4, "episodes": 400},
            #{"name": "DH15_Deep_BC_Control", "bc_epochs": 120, "q_weight": 0.0, "lr": 3e-4, "episodes": 400},
            #{"name": "DH16_Ultra_Micro_Marathon", "bc_epochs": 50, "q_weight": 0.001, "lr": 3e-4, "episodes": 600},
        # ==========================================
        # 第五期实验矩阵 (混合数据集突围测试)
        # 核心目的：验证混合流形能否在保持高存活率的同时，打破 22 m/s 均速天花板
        # ==========================================
            # {"name": "DH17_Mixed_BC_Control", "bc_epochs": 50, "q_weight": 0.0, "lr": 3e-4, "episodes": 400},
            # {"name": "DH18_Mixed_Ultra_Micro", "bc_epochs": 50, "q_weight": 0.001, "lr": 3e-4, "episodes": 400},
            # {"name": "DH19_Mixed_Thicker_Base", "bc_epochs": 80, "q_weight": 0.001, "lr": 3e-4, "episodes": 400},
            # {"name": "DH20_Mixed_Marathon", "bc_epochs": 50, "q_weight": 0.001, "lr": 3e-4, "episodes": 600},
        ]

        # ==========================================
        # Diff-SAC 针对 Merge 环境的初始探索矩阵
        # 根据 Highway 积累的经验，我们围绕最稳妥的参数区间制定了 Merge 第一期探雷策略：
        # ==========================================
        merge_experiment_configs = [
            # === 第一期：微弱 Q 引导探雷 (经评估发现 Q 权重被 BC 淹没，表现为纯模仿) ===
            # {"name": "DM1_Zero_Q", "bc_epochs": 50, "q_weight": 0.0, "lr": 3e-4, "episodes": 400},
            # {"name": "DM2_Micro_Q", "bc_epochs": 50, "q_weight": 0.005, "lr": 3e-4, "episodes": 400},
            # {"name": "DM3_Gentle_Q", "bc_epochs": 50, "q_weight": 0.01, "lr": 3e-4, "episodes": 400},
            # {"name": "DM4_Standard_Q", "bc_epochs": 50, "q_weight": 0.05, "lr": 3e-4, "episodes": 400},

            # === 第二期：寻找相变点 (Phase Transition) 跨量级 Q 引导突围 ===
            # 目的：强行放大 Q-weight，观察扩散网络何时撕裂 BC 安全护甲，从“保守避让”转变为“激进寻隙”。
            #{"name": "DM5_Mild_Transition", "bc_epochs": 50, "q_weight": 0.1, "lr": 3e-4, "episodes": 400},   # 破冰试探：0.1 梯度干预
            #{"name": "DM6_Moderate_Override", "bc_epochs": 50, "q_weight": 0.5, "lr": 3e-4, "episodes": 400}, # 中度干预：预期均速开始攀升
            #{"name": "DM7_Strong_Override", "bc_epochs": 50, "q_weight": 1.0, "lr": 3e-4, "episodes": 400},   # 强力干预：RL 与 BC 的正面对抗
            #{"name": "DM8_Extreme_Domination", "bc_epochs": 50, "q_weight": 2.0, "lr": 3e-4, "episodes": 400},# 极限干预：预期存活率断崖，寻找生存极限

            # === 第三期 diff-SAC 实验 (基于 100% M4 稳健专家底座) ===
            # 核心目的：探究更宽广的 M4 动作流形，能否承受住比 M8 更大的 Q 梯度冲击，推迟 OOD 崩溃点。
            {"name": "DM9_M4_Prior_Only", "bc_epochs": 50, "q_weight": 0.0, "lr": 3e-4, "episodes": 400},   # 确立新基准：纯 BC 拟合 M4 专家，验证能否完美复刻 100% 存活率和 18.6 m/s 均速
            {"name": "DM10_M4_Standard_Q", "bc_epochs": 50, "q_weight": 0.1, "lr": 3e-4, "episodes": 400},  # 低烈度探测：在 M4 的开阔流形下，测试扩散模型对常规弱 Q 信号的敏感度
            {"name": "DM11_M4_Strong_Q", "bc_epochs": 50, "q_weight": 1.0, "lr": 3e-4, "episodes": 400},    # 强力博弈：Q 梯度深度介入，期望在不折损存活率的前提下，均速历史性突破 19.0 m/s
            {"name": "DM12_M4_Extreme_Q", "bc_epochs": 50, "q_weight": 5.0, "lr": 3e-4, "episodes": 400},   # 极限突破：施加暴力大权重，探寻 M4 底座的“安全与效率”相变崩溃点
        ]

        # ==========================================
        # Diff-SAC 针对 Racetrack 环境的初始探索矩阵
        # ==========================================
        racetrack_experiment_configs = [
            {"name": "DR1_Zero_Q", "bc_epochs": 50, "q_weight": 0.0, "lr": 3e-4, "episodes": 400},
            {"name": "DR2_Micro_Q", "bc_epochs": 50, "q_weight": 0.005, "lr": 3e-4, "episodes": 400},
            {"name": "DR3_Standard_Q", "bc_epochs": 50, "q_weight": 0.05, "lr": 3e-4, "episodes": 400},
            {"name": "DR4_Strong_Q", "bc_epochs": 50, "q_weight": 0.5, "lr": 3e-4, "episodes": 400},
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
            print(f"📊 参数配置: {config}")
            print(f"==================================================")

            bc_kwargs = {
                "data_path": data_path, "env_name": TARGET_ENV,
                "num_epochs": config["bc_epochs"], "batch_size": 256, "learning_rate": config["lr"]
            }
            if "run_name" in inspect.signature(train_diffusion_bc).parameters:
                bc_kwargs["run_name"] = run_name
            pretrained_model_path = train_diffusion_bc(**bc_kwargs)
            clear_gpu_memory()

            # 执行该组参数的在线微调
            train_online_diffusion(
                pretrained_actor_path=pretrained_model_path,
                expert_data_path=data_path,
                env_name=TARGET_ENV,
                max_episodes=config["episodes"],
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