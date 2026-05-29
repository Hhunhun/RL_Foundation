# Diff-SAC - 自动驾驶强化学习与扩散模型控制框架

本项目是一个面向自动驾驶连续控制任务的强化学习 (RL) 与生成式扩散模型 (Diffusion Models) 实验框架。系统涵盖了从“离线专家数据行为克隆 (Offline BC)”到“在线闭环微调 (Online RL)”的完整训练流程。

目前系统支持并重构了三大自动驾驶场景：**Highway (高速巡航)**、**Merge (匝道汇入)** 与 **Racetrack (赛道竞速)**。

---

## ✨ 核心特性

### 1. 算法架构
- **Diff-SAC (Asymmetric Diffusion RL)**: 引入非对称掩码更新机制 (Asymmetric Masking)，Actor 仅对专家数据执行行为克隆，Critic 吸收全局经验进行价值评估，结合了扩散模型的分布拟合能力与强化学习的寻优能力。
- **稳健的双重价值网络 (Robust Critic)**: 结合 Huber Loss、极小值截断与奖励截断 (Reward Clipping)，有效抑制连续控制任务与生成式策略中常见的 Q 值过估计与分布外 (OOD) 采样风险。
- **挤压高斯基准 (Squashed Gaussian SAC)**: 基准模型基于最大熵理论构建，部署了自适应温度系数优化 (Alpha Tuning) 与雅可比概率密度修正 (Jacobian Correction)。

### 2. 马尔可夫决策过程 (MDP) 优化
- **时空微扰 (Dynamic Jittering)**: 在 Merge 场景中引入初始状态的时空噪声，缓解确定性环境下的轨迹过拟合问题。
- **程序化域随机化 (Procedural Domain Randomization)**: 在 Racetrack 场景中通过路网拓扑检索，动态生成多模态的周边车辆遭遇场景，提升策略泛化性。
- **时序截断修正 (Absorbing State Correction)**: 严格区分环境终止 (Terminated) 与超时截断 (Truncated)，修正长周期任务评估中的时间差分 (TD) 误差。
- **平滑奖励函数 (Reward Shaping)**: 针对扩散模型初期探索不稳定的特性，简化复杂的一阶运动学惩罚，构建基于安全性与基础车速的稠密奖励体系。

### 3. 系统训练提效
- **计算图逻辑优化**: 通过拦截物理引擎底层的全网格搜寻结算逻辑，直接在张量层处理奖励信号，显著提升采样与训练速度。
- **显存自动回收**: 内置回合生存期监测机制，针对高频碰撞导致的计算图堆积问题，主动执行垃圾回收与显存清理。
- **环境参数动态调度**: 支持在训练过程中跨层修改 Gym Wrapper 惩罚权重，实现从探索期到稳定期的动力学参数平滑过渡。

---

## 📊 模型评估与可视化

项目内置了统一的评估流水线 (`run_03_evaluate.py` 与 `run_04` 系列脚本)，在测试期间关闭所有训练辅助惩罚，主要考核碰撞率与平均速度等物理指标。
系统支持自动生成以下标准化数据图表：
*   🌧️ **奖励核密度雨云图 (Raincloud Plots)**
*   🕸️ **五维综合性能雷达图 (Radar Charts)**
*   🫧 **安全-收益帕累托散点图 (Pareto Front Scatter)**
*   📊 **变异系数与平均奖励指标柱状图 (CV & Mean Reward)**
*   🧬 **动作分布 KDE 等高线图 (Action Manifold KDE)**

---

## 📁 系统文件结构

```text
Diff-SAC/
├── run_00_quick_test.py            # 冒烟测试脚本 (验证系统连通性)
├── run_01_collect_data.py          # 专家轨迹数据采集与存盘
├── run_02_train_pipeline.py        # 训练调度模块 (支持多组参数顺序执行)
├── run_03_evaluate.py              # 模型统一评估与结果图表生成
├── README.md                       # 项目主文档
├── requirements.txt                # 运行环境依赖包清单
│
├── algorithms/                     # 核心算法实现模块
│   ├── sac/
│   │   ├── sac_agent.py            # 最大熵 SAC 智能体控制逻辑
│   │   └── sac_nets.py             # 挤压高斯 Actor 与 Double-Q Critic 网络架构
│   └── diffusion_sac/
│       ├── diff_sac_agent.py       # Diff-SAC 混合更新与非对称掩码控制中枢
│       └── diffusion_model.py      # 条件扩散去噪 Actor 网络预演与调度模型
│
├── envs/                           # 物理引擎定制封装层 (MDP Wrappers)
│   ├── __init__.py                 # 环境路由构建工厂
│   ├── highway_wrapper.py          # 高速巡航状态展平与动力学正则化
│   ├── merge_wrapper.py            # 匝道汇入 TTC 时空微扰与安全边界硬截断
│   └── racetrack_wrapper.py        # 赛道竞速曲率寻迹强化与多模态程序化域随机化
│
├── baseline_sac/                   # SAC 专家基准模型训练中枢
│   ├── main_highway.py             # Highway 环境基线训练脚本 (含计算图逻辑优化)
│   ├── main_merge.py               # Merge 环境基线训练脚本 (含穿透式课程学习)
│   └── main_racetrack.py           # Racetrack 环境基线训练脚本 (含时序修正)
│
├── runners/                        # Diff-SAC 子训练管线
│   ├── train_offline_bc.py         # 第一阶段：纯离线行为克隆预训练 (Offline BC)
│   └── train_online_diff.py        # 第二阶段：在线非对称强化微调 (Online RL)
│
├── core/                           # 数据底座与经验回放模块
│   ├── replay_buffer.py            # 基础连续内存预分配环形经验池
│   └── offline_buffer.py           # 混合数据池 (含专家掩码派发与全局正态归一化)
│
├── utils/                          # 基础辅助工具
│   └── logger.py                   # 解决 Windows 平台实时刷新阻塞的 TensorBoard 封装
│
├── data/expert_data/               # [产出] 离线专家高质量数据集归档 (.npz)
└── outputs/                        # [产出] 实验工件归档 (模型权重、日志、评估图表与录像)
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