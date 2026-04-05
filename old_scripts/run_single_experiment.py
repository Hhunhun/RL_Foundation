import gc
import torch

# 导入三大模块
from runners.train_offline_bc import train_diffusion_bc
from runners.train_online_diff import train_online_diffusion


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 60)
    print("⚡ 快速单次验证脚本 (Single Run Pipeline)")
    print("=" * 60)

    # ==========================================
    # 1. 数据准备
    # ==========================================
    # 根据你昨晚的日志，数据已经成功采集过了！
    # 我们直接使用昨晚的数据路径，跳过 collect_expert_data，节约几分钟时间。
    # 如果你想重新采数，可以把这里的硬编码注释掉，解除下方 collect_expert_data 的注释。
    data_path = "../data/expert_data/dataset_v5_20260404_035105/expert_transitions.npz"
    print(f"📦 复用已有专家数据: {data_path}")

    # data_path = collect_expert_data(
    #     model_path="outputs/models/highway-v0_SAC_20260330_135449/sac_highway_final.pth",
    #     env_name="highway-v0",
    #     target_transitions=50000
    # )

    clear_gpu_memory()

    # ==========================================
    # 2. 阶段二：离线预训练 (Behavior Cloning)
    # ==========================================
    # 时间紧迫，将 Epoch 从 50 降到 30。对于 Diffusion 来说，30 个 Epoch 通常已经能看出收敛趋势。
    pretrained_model_path = train_diffusion_bc(
        data_path=data_path,
        num_epochs=30,
        batch_size=256,
        learning_rate=3e-4
    )

    clear_gpu_memory()

    # ==========================================
    # 3. 阶段三：在线微调 (Online RL)
    # ==========================================
    # 将 episodes 从 300 降至 100 局。用于快速验证代码是否畅通，且 Q 值引导是否生效。
    train_online_diffusion(
        pretrained_actor_path=pretrained_model_path,
        expert_data_path=data_path,
        max_episodes=100,
        batch_size=256,
        q_weight=0.05,
        lr=3e-4
    )

    clear_gpu_memory()

    print("=" * 60)
    print("🎉 单次完整训练流程结束！请前往 outputs/logs 查看 TensorBoard 曲线。")
    print("=" * 60)