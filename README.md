# RL Foundation: Autonomous Driving Control via SAC and Reward Shaping & Diffusion

## 1. 项目概述 (Project Overview)
本项目构建了一个基于深度强化学习（Deep Reinforcement Learning, DRL）的自动驾驶规控基准平台（Baseline Framework）。核心算法早期采用连续动作空间的软演员-评论家算法（Soft Actor-Critic, SAC），在 `highway-v0` 仿真环境中进行高速公路多车道交互场景下的变道与速度控制训练。

在确立了纯 RL 的能力边界与安全瓶颈后，本项目目前已正式进入 **生成式强化学习阶段 (Generative RL)**。通过引入条件扩散模型（Conditional Diffusion Models）重构 Actor 网络，并结合专家数据增强（Demo Augmented RL），旨在打破传统反应式单步 RL 在“安全（Safety）与效率（Efficiency）”上的博弈困境。

## 2. 核心架构与“绝对冻结”评估体系 (Architecture & Evaluation)
项目不仅包含算法实现，更着重打造了工业级的公平评估流水线：

* **环境包装器 (`envs/highway_wrapper.py`)**：
  * **训练态 (`is_eval=False`)**：挂载 LQR 二次型转向惩罚、倒车重罚、龟速惩罚等严厉的先验引导奖励，用于解决早期模型钻物理引擎漏洞（Offroad Hacking, Reverse Hacking）的问题。
  * **评估态 (`is_eval=True`)**：**冻结考场机制**。在模型评估阶段，系统强制剥离所有人工塑形奖励，仅保留最纯净的客观物理碰撞与提速得分，确保所有历史版本乃至未来不同架构模型在同一标尺下进行绝对公平的量化对比。
* **双线程评估系统 (`evaluate_models.py`)**：分离视频录制（定性分析）与无头全速蒙特卡洛统计（定量分析），自动计算存活率、均速、策略标准差，并输出 CSV 数据与学术级箱线图。

## 3. 策略演进与纯 RL 消融实验 (RL Ablation Study)
本项目完整记录了 6 个大版本的传统 RL 迭代历史。在纯净物理环境下的量化测试暴露出传统 RL 的核心痛点：

| 版本 (Version) | 核心约束机制 | 纯净考场实战表现 (Objective KPI) | 工程诊断与病理分析 |
| :--- | :--- | :--- | :--- |
| **v1.0 & v2.0** | 早期无约束或轻度线性约束 | 存活率 ~80%，均速 ~21m/s | 表现出疯狂原地摆头（Wobbling）和越野作弊，缺乏横向稳定性。 |
| **v3.0** | 引入 LQR 惩罚与出界死刑 | 存活率 **6.7%**，均速 25.6m/s | 极低容错率导致正向样本匮乏，模型产生严重欠拟合。 |
| **v4.0** | 改为 40k 总步数制训练 | - | 演化出极其危险的倒车苟活（Reverse Hacking）以规避转向惩罚。 |
| **v5.0<br>(Safe Baseline)** | **LQR + 绝对禁止倒车** | 存活率 **90.0%**，均速 **21.13m/s** | 传统 RL 最稳健状态，但陷入**局部最优**。为了极高安全性选择不敢超车的“保守策略”。 |
| **v6.0<br>(Efficiency Pro)** | **v5.0 + 绝对转向约束 + 严惩龟速** | 存活率 **76.7%**，均速 **22.42m/s** | 强行逼迫提速导致容错率急剧下降，揭示了单步 RL 在**安全与效率间的不可调和性**。 |

## 4. 扩散网络架构升级 (Diffusion-RL Framework)
综合上述数据，本项目确立 v5.0 为保守型专家基线，并在此基础上构建了全新的 **Diffusion-SAC** 混合架构。该架构分为三个高度解耦的工程阶段：

* **Phase 1: 专家数据采集 (Expert Data Collection)**
  利用 v5.0 模型在纯净环境中运行，通过严格的过滤机制（剔除撞车局），采集并压缩了 **50,000 步** 极其纯净的安全驾驶过渡数据（Transitions），作为后续生成模型的“安全底线锚点”。
* **Phase 2: 离线行为克隆 (Offline Behavior Cloning)**
  构建了基于 Mish 激活函数和时间正弦位置编码（Sinusoidal Positional Embedding）的轻量级 Conditional Actor，配合 Noise Scheduler 进行离线加噪与去噪训练。网络成功将专家动作分布的 MSE Loss 降至 0.2，实现了对安全驾驶逻辑的刻印。
* **Phase 3: 专家混合在线强化学习 (Demo Augmented Online RL)**
  这是本架构的核心突破。在将 Diffusion Actor 接入在线环境由 Critic 指导微调时，解决了严重的**分布偏移（Distribution Shift）**与**灾难性遗忘（Catastrophic Forgetting）**难题：
  1. **DDIM 加速**：在在线交互和更新时，将 100 步去噪压缩至 5 步，保障控制实时性。
  2. **Q 值梯度缩放 (Q-Loss Scaling)**：大幅缩放 Critic 回传的梯度，避免巨大的 Q 值误差“锤碎”预训练的扩散流形。
  3. **Demo 注入机制**：强制将 50,000 步专家数据混入在线经验池。确保模型在在线探索时不越过安全底线，实现了高容错的平滑微调。

## 5. 快速开始 (Quick Start)

### 依赖安装
```bash
pip install torch numpy gymnasium highway-env tensorboard matplotlib pandas