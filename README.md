# RL Foundation - 自动驾驶强化学习流水线

本项目是一个高度模块化、自动化的强化学习实验框架，专注于解决自动驾驶场景下的规控问题。目前已完美支持 **Highway (高速巡航)** 与 **Merge (匝道汇入)** 双测试场景，并集成了经典的 **Soft Actor-Critic (SAC)** 算法以及前沿的 **Diffusion-RL (基于扩散模型的强化学习)** 架构。

## ✨ 核心特性 (Key Features)

- **双重复杂场景支持**: 动态适配 `highway-v0` 与 `merge-v0`，底层环境奖励与平滑约束(Jerk/Steering)高度可配置。
- **SOTA 算法基座**: 
  - **SAC (Soft Actor-Critic)**: 用于构建极其稳定的法规级专家底座。
  - **Diff-SAC (Diffusion-SAC)**: 结合行为克隆(BC)与强化学习微调，突破传统 SAC 的均速与存活率瓶颈。
- **外科手术式数据蒸馏**: 支持从多个 SAC 专家模型中提取特定流形的数据并进行智能混合，构建神仙级离线数据集。
- **全自动通宵挂机流水线**: 内置参数矩阵轮询系统，可一键挂机执行数十个消融实验，并在指定时间安全关机。
- **自动化裁判法庭**: 统一的评估与可视化模块，自动生成学术级箱线图、柱状图与量化指标 CSV，支持双盲公平测试。
- **工业级防死锁设计**: 针对劣质模型或引擎 Bug 内置完善的超时熔断机制与显存自动回收策略。

## 📁 项目架构 (Project Structure)

```text
RL_Foundation/
├── run_00_quick_test.py            # [测试] 全链路冒烟测试脚本 (一分钟验证代码健壮性)
├── run_01_collect_data.py          # [阶段一] 专家轨迹数据采集模块 (支持抖动增强与纯净模式)
├── run_02_train_pipeline.py        # [阶段二/三] 核心自动化训练调度器 (含通宵扫参模式)
├── run_03_evaluate.py              # [阶段四] 统一模型评估与可视化流水线
│
├── baseline_sac/                   # SAC 专家基线训练器 (分别对应 Highway 和 Merge)
│   ├── main_highway.py
│   └── main_merge.py
│
├── envs/                           # 环境工厂与包装器定义 (Environment Factory)
│   ├── __init__.py                 # 动态环境路由
│   ├── highway_wrapper.py          # 高速环境奖励塑形与状态展平
│   └── merge_wrapper.py            # 汇入环境奖励塑形 (包含底层源码缺陷修复补丁)
│
├── algorithms/                     # 算法大脑模块
│   ├── sac/                        # 经典 SAC 实现
│   └── diffusion_sac/              # 结合条件扩散模型的 Diff-SAC 实现
│
├── runners/                        # 子训练管线执行器
│   ├── train_offline_bc.py         # 纯离线扩散模型行为克隆 (Offline BC)
│   └── train_online_diff.py        # 在线真实环境强化学习微调 (Online RL Finetune)
│
├── core/                           # 核心基础组件
│   ├── replay_buffer.py            # 基础经验回放池
│   └── offline_buffer.py           # 混合经验回放池 (含在线/离线双区管理与数据归一化)
│
├── data/expert_data/               # [产出] 存放各环境采集的高质量专家数据集 (.npz)
└── outputs/                        # [产出] 实验结果分类归档
    ├── highway-v0/
    │   ├── logs/                   # TensorBoard 训练日志
    │   ├── models/                 # 网络权重文件 (.pth)
    │   ├── eval_results/           # 评估生成的图表、CSV 数据
    │   └── videos/                 # 评估生成的自动驾驶录像 (.mp4)
    └── merge-v0/                   # 同上，环境严格隔离
```

## 🚀 标准工作流 (Workflow)

### 1. 验证系统健壮性
每次修改底层代码后，请务必先运行冒烟测试：
```bash
python run_00_quick_test.py
```
该脚本将在极短时间内验证所有环境与算法的连通性，预期输出全绿 `ALL PASSED`。

### 2. 训练特定环境的 SAC 专家
以 Merge 环境为例，进入 `baseline_sac` 调整实验参数并训练：
```bash
python baseline_sac/main_merge.py
```

### 3. 采集高质量专家数据
运行数据采集脚本，选择对应的环境与模式：
```bash
python run_01_collect_data.py
```

### 4. 启动自动化实验流水线
修改 `run_02_train_pipeline.py` 中的 `experiment_configs` 参数矩阵，然后运行：
```bash
python run_02_train_pipeline.py
```
推荐选择 `[2] OVERNIGHT` 模式，设定次日早晨为截止时间，让显卡通宵完成消融实验。

### 5. 统一评估与出图
通宵结束后，运行评估脚本，自动生成对比图表：
```bash
python run_03_evaluate.py
```
生成的报告及图表可在 `outputs/{env_name}/eval_results/` 目录下查看。