# RL Foundation: 强化学习算法基建与实战

本项目旨在从零构建高标准、模块化的深度强化学习（DRL）算法底层框架。目前已实现核心的 Soft Actor-Critic (SAC) 算法，并成功在连续控制基准环境中通过收敛测试。未来将接入 HighwayEnv 等复杂自动驾驶仿真环境。

## 🌟 核心特性
* **模块化解耦**: 抽象出统一的 `BaseAgent` 接口，经验回放池与算法逻辑完全分离。
* **高性能数据管道**: 基于 NumPy 预分配内存的 `ReplayBuffer`，支持极速批量采样。
* **严谨的数学推导还原**:
    * Actor：高斯分布策略 + 重参数化技巧 + $\tanh$ 雅可比行列式修正。
    * Critic：双 Q 网络防高估 + 平滑软更新 (Soft Update) + 自动熵调整 (Alpha Tuning - 待接入)。
* **科学的可视化诊断**: 原生集成 TensorBoard，实时监控 Reward、Loss 及网络参数分布。

## 📂 项目架构
├── algorithms/     # 核心算法库 (目前包含 SAC)
├── core/           # 底层基建 (BaseAgent, ReplayBuffer)
├── outputs/        # 统一输出管理 (Logs, Models, Videos)
├── utils/          # 工具链 (Logger 等)
├── main_train.py   # 训练入口
└── test_render.py  # 可视化评估脚本

## 🚀 快速开始
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行基础收敛测试 (Pendulum-v1)
python main_train.py

# 3. 查看实时训练曲线
tensorboard --logdir=outputs/logs