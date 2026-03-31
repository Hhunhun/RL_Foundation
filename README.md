# RL Foundation: Autonomous Driving Control via SAC and Reward Shaping

## 1. 项目概述 (Project Overview)
本项目构建了一个基于深度强化学习（Deep Reinforcement Learning, DRL）的自动驾驶规控基准平台（Baseline Framework）。核心算法采用连续动作空间的软演员-评论家算法（Soft Actor-Critic, SAC），在 `highway-v0` 仿真环境中进行高速公路多车道交互场景下的变道与速度控制训练。

本阶段（RL Foundation）的核心研究目的在于：**探究 Model-free RL 在复杂规控任务中的能力边界**。通过多版本的奖励塑形（Reward Shaping）、动作空间绝对约束（Action Space Constraints）以及冻结物理评估环境的确立，系统性地揭示了单步反应式 RL 在“安全（Safety）与效率（Efficiency）”上的博弈困境，为下一阶段引入生成式扩散模型（Diffusion Models）进行长视野轨迹规划奠定基础与核心动机（Motivation）。

## 2. 核心架构与“绝对冻结”评估体系 (Architecture & Evaluation)
项目不仅包含算法实现，更着重打造了**工业级的公平评估流水线**：

* **环境包装器 (`envs/highway_wrapper.py`)**：
  * **训练态 (`is_eval=False`)**：挂载 LQR 二次型转向惩罚、倒车重罚、龟速惩罚等严厉的先验引导奖励，用于解决早期模型钻物理引擎漏洞（Offroad Hacking, Reverse Hacking）的问题。
  * **评估态 (`is_eval=True`)**：**冻结考场机制**。在模型评估阶段，系统强制剥离所有人工塑形奖励，仅保留最纯净的客观物理碰撞与提速得分，确保所有历史版本乃至未来不同架构模型在同一标尺下进行绝对公平的量化对比。
* **双线程评估系统 (`evaluate_models.py`)**：分离视频录制（定性分析）与无头全速蒙特卡洛统计（定量分析），自动计算存活率、均速、策略标准差，并输出 CSV 数据与学术级箱线图。

## 3. 策略演进与消融实验 (Ablation Study)
本项目完整记录了 6 个大版本的迭代历史。在纯净物理环境（冻结考场）下的量化测试暴露出传统 RL 的核心痛点：

| 版本 (Version) | 核心约束机制 | 纯净考场实战表现 (Objective KPI) | 工程诊断与病理分析 |
| :--- | :--- | :--- | :--- |
| **v1.0 & v2.0** | 早期无约束或轻度线性约束 | 存活率 ~80%，均速 ~21m/s | 虽然存活率尚可，但在录像中表现出疯狂原地摆头（Wobbling）和越野作弊（Offroad Hacking），缺乏横向稳定性。 |
| **v3.0** | 引入 LQR 惩罚与出界死刑 | 存活率 **6.7%**，均速 25.6m/s | 由于按局数（Episode）训练，极低容错率导致正向样本匮乏，模型产生**严重欠拟合**，只会踩死油门。 |
| **v4.0** | 改为 40k 总步数制训练 | - | 训练量达标，但演化出极其危险的**倒车苟活（Reverse Hacking）**以规避转向惩罚。 |
| **v5.0<br>(Safe Baseline)** | **LQR + 绝对禁止倒车** | 存活率 **90.0%**，均速 **21.13m/s** | 达到了传统 RL 的最稳健状态，但陷入**局部最优**。为了极高的安全性，模型选择不敢超车的“怂包策略”，通行效率低下。 |
| **v6.0<br>(Efficiency Pro)** | **v5.0 + 绝对转向约束 + 严惩龟速** | 存活率 **76.7%**，均速 **22.42m/s** | 强行逼迫模型提速（打破局部最优），导致在缺乏长程规划的情况下容错率急剧下降，揭示了单步 RL 在**安全与效率间的不可调和性**。 |

## 4. 结论与下一步研究计划 (Conclusion & Future Work)
综合客观评估数据，本项目正式确立 **v5.0 (Safety Conservative)** 作为后续研究的 Baseline。

**研究发现**：基于 MLP 的无模型强化学习（Model-free RL）由于仅依赖当前帧观测进行单步动作输出，缺乏时空多步推理与预测能力（Long-horizon Prediction）。当面对复杂交通流时，模型无法提前规划安全变道间隙，导致必须在“极度保守（v5.0）”或“危险激进（v6.0）”之间妥协。

**Next Phase (Diffusion-RL)**：
下一阶段，项目将从反应式决策（Reactive Decision）转向生成式规划（Generative Planning）。计划引入 **Diffusion Model (扩散模型)**，利用其卓越的多模态分布建模能力，直接在时间尺度上生成未来 $T$ 步的安全轨迹序列，旨在打破 v5.0 的效率瓶颈，实现安全与效率的双赢。

## 5. 快速开始 (Quick Start)

### 依赖安装
```bash
pip install torch numpy gymnasium highway-env tensorboard matplotlib pandas