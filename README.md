# RL Foundation: Autonomous Driving Control via SAC and Reward Shaping & Diffusion

## 1. 项目概述 (Project Overview)
本项目构建了一个基于深度强化学习（Deep Reinforcement Learning, DRL）的自动驾驶规控基准平台（Baseline Framework）。核心算法早期采用连续动作空间的软演员-评论家算法（Soft Actor-Critic, SAC），在 `highway-v0` 仿真环境中进行高速公路多车道交互场景下的变道与速度控制训练。

在确立了纯 RL 的能力边界与安全瓶颈后，本项目目前已正式进入 **生成式强化学习阶段 (Generative RL)**。通过引入条件扩散模型（Conditional Diffusion Models）重构 Actor 网络，并结合专家数据增强（Demo Augmented RL），旨在打破传统反应式单步 RL 在“安全（Safety）与效率（Efficiency）”上的博弈困境。

## 2. 核心架构与“绝对冻结”评估体系 (Architecture & Evaluation)
项目不仅包含算法实现，更着重打造了工业级的自动化流水线与公平评估体系：

* **现代化工程架构**：采用高度解耦的模块化设计，将底层算法 (`algorithms/`)、核心数据流 (`core/`)、子训练循环 (`runners/`) 与顶层调度脚本 (`run_0x...`) 严格分离，支持通宵挂机跑参。
* **环境包装器 (`envs/highway_wrapper.py`)**：
  * **训练态 (`is_eval=False`)**：挂载 LQR 二次型转向惩罚、倒车重罚、龟速惩罚等严厉的先验引导奖励，用于解决早期模型钻物理引擎漏洞的问题。
  * **评估态 (`is_eval=True`)**：**冻结考场机制**。强制剥离所有人工塑形奖励，仅保留最纯净的客观物理碰撞与提速得分，确保所有阶段（SAC 与 Diff-SAC）的模型在同一标尺下进行绝对公平的量化对比。
* **双线程评估系统 (`run_03_evaluate.py`)**：支持智能路由，自动处理 Diffusion 模型的归一化数据流。分离视频录制（定性分析）与大样本全速蒙特卡洛统计（定量分析），自动计算存活率、均速、策略标准差，并一键输出学术级对比图表。

## 3. 策略演进与纯 RL 消融实验 (RL Ablation Study)
本项目完整记录了 6 个大版本的传统 RL 迭代历史，揭示了传统 RL 的核心痛点：

| 版本 (Version) | 核心约束机制 | 纯净考场实战表现 (Objective KPI) | 工程诊断与病理分析 |
| :--- | :--- | :--- | :--- |
| **v1.0 & v2.0** | 早期无约束或轻度线性约束 | 存活率 ~80%，均速 ~21m/s | 表现出疯狂原地摆头（Wobbling）和越野作弊，缺乏横向稳定性。 |
| **v3.0** | 引入 LQR 惩罚与出界死刑 | 存活率 **6.7%**，均速 25.6m/s | 极低容错率导致正向样本匮乏，模型产生严重欠拟合。 |
| **v4.0** | 改为 40k 总步数制训练 | - | 演化出极其危险的倒车苟活（Reverse Hacking）以规避转向惩罚。 |
| **v5.0<br>(Safe Baseline)** | **LQR + 绝对禁止倒车** | 存活率 **80%+**，均速 **~21.5m/s** | 传统 RL 最稳健状态，但陷入**局部最优**。为了极高安全性选择不敢超车的“保守策略”。 |
| **v6.0<br>(Efficiency Pro)** | **v5.0 + 绝对转向约束 + 严惩龟速** | 存活率 **<80%**，均速 **~22.5m/s** | 强行逼迫提速导致容错率急剧下降，揭示了单步 RL 在**安全与效率间的不可调和性**。 |

## 4. 扩散网络架构升级 (Diffusion-RL Framework)
综合上述数据，本项目确立 v5.0 为保守型专家基线，并在此基础上构建了全新的 **Diffusion-SAC** 混合架构，成功实现了从端到端 RL 向扩散生成控制的跨越：

* **Phase 1: 专家数据采集与清洗 (`run_01_collect_data.py`)**
  利用 v5.0 专家模型在纯净环境中运行。引入**概率性行为抖动 (Behavioral Jitter)** 强制专家展示纠偏动作，并严格丢弃任何包含撞车的瑕疵局，最终提取 50,000 步极其纯净的数据作为安全底座。
* **Phase 2: 离线行为克隆 (`runners/train_offline_bc.py`)**
  构建基于 Mish 激活函数和正弦位置编码的 Conditional Actor。在不接触真实环境的前提下，通过加噪/去噪过程让网络死记硬背专家的安全驾驶流形 (Manifold)。
* **Phase 3: 非对称掩码在线强化学习 (`runners/train_online_diff.py` & `diff_sac_agent.py`)**
  架构的核心突破点。在环境微调阶段，解决了分布偏移 (OOD) 与模型崩溃问题：
  1. **非对称更新掩码 (Asymmetric Masking)**：Actor 被严格限制只能克隆专家数据（禁止模仿自己探索时的撞车动作），但 Critic 评估全局数据（好坏经验兼收），实现“不忘本的创新”。
  2. **双重防爆装甲**：为 Critic 部署了 Reward Clipping (-10~10) 和 Huber Loss，防止严苛自动驾驶环境下的梯度爆炸。
  3. **100% EMA 权重同步**：修复了评估阶段影子网络随机化导致的 0% 存活率漏洞，保障了推理阶段动作的极度平滑。

## 5. 快速开始 (Quick Start)

### 依赖安装
```bash
pip install torch numpy gymnasium highway-env tensorboard matplotlib pandas