# RL Foundation: Autonomous Driving Control via Soft Actor-Critic and Reward Shaping

## 1. 项目概述 (Project Overview)
本项目构建了一个基于深度强化学习（Deep Reinforcement Learning, DRL）的自动驾驶规控框架。核心算法采用连续动作空间的软演员-评论家算法（Soft Actor-Critic, SAC），并在 `highway-v0` 仿真环境中进行高速公路场景下的变道与速度控制训练。

本项目的核心研究重点在于**奖励塑形（Reward Shaping）**与**动作空间约束（Action Space Constraints）**。通过多版本的迭代与消融实验，系统性地解决了强化学习在自动驾驶领域常见的“奖励作弊（Reward Hacking）”问题（如越野行驶、倒车苟活等），最终实现符合交通法规与运动学平滑性的专家级策略。

## 2. 核心架构 (Architecture)
项目代码结构解耦为算法核心、环境封装、训练循环与量化评估四大模块：

* `algorithms/sac/`: SAC 算法的底层实现（Actor-Critic 网络架构、经验回放池、温度参数自适应调整）。
* `envs/highway_wrapper.py`: 环境包装器。负责状态空间降维（Flatten）、物理边界定义，以及核心的**AV Control (LQR 二次型平滑与速度约束)** 惩罚机制。
* `utils/logger.py`: 监控模块。集成了强制物理落盘（`flush`）机制的 TensorBoard 记录器，解决 Windows 平台下的 I/O 缓冲延迟问题。
* `main_highway.py`: 训练主循环。采用**基于环境步数（Step-based）**的训练逻辑（支持 120,000+ 步大规模交互），彻底解决早期欠拟合问题。
* `evaluate_models.py`: 双线程量化评估流水线。分离视频录制与大规模蒙特卡洛统计进程，自动计算均值、标准差与存活率，并输出学术级对比图表。

## 3. 奖励塑形与策略演进 (Reward Shaping Evolution)
为验证惩罚机制的有效性，本项目记录了 5 个核心版本的消融实验（Ablation Study）。环境的观测状态为运动学特征（Kinematics），动作空间为连续的 `[throttle, steering]`。

| 版本 (Version) | 约束机制 (Constraints & Wrappers) | 实战表现定性分析 (Qualitative Analysis) |
| :--- | :--- | :--- |
| **v1.0 Baseline** | 无约束 | 存活率高，但在车道内存在高频方向盘抖动（Wobbling），且由于原生变道惩罚导致策略极度保守。 |
| **v2.0 Linear Penalty** | 提高速度奖励，引入线性转向惩罚 | 惩罚过重导致拒绝变道；发现物理引擎漏洞，策略演化为通过**驶入草地（Offroad Hacking）**来规避碰撞并获取最高分。 |
| **v3.0 LQR Smoothing** | 开启草地出界终止（Offroad Terminal），引入二次型（Quadratic）Jerk惩罚 | 逻辑完善，但受限于按局数（Episode）训练的逻辑，早期出界暴毙导致经验池正向样本不足，存在严重**欠拟合**。 |
| **v4.0 Step-based** | 训练架构升级为按总步数（40k Steps）交互 | 训练充分，彻底学会规避草地。但策略演化出新的作弊行为：通过**高速倒车（Reverse Hacking）**以规避转向惩罚并保持存活。 |
| **v5.0 AV Regulated** | **LQR 转向惩罚 + 绝对速度约束（禁止 $v_x < 0$）** | 最终版本（120k Steps）。绝对遵循交通法规，方向盘轨迹平滑，遇慢车可稳定做出加速变道超车决策。 |

## 4. 评估基准与量化指标 (Evaluation Metrics)
运行评估流水线 `evaluate_models.py`，系统将在 `outputs/eval_results/` 目录下自动生成带有时间戳的专属实验档案，包含以下三个维度的评估：

1.  **安全性 (Safety)**：通过大样本（N=30/50）测试统计**存活率 (Survival Rate)**。
2.  **效率性 (Efficiency)**：实时探针提取自车平均纵向速度（Mean Speed $v_x$）与平均累计奖励（Mean Cumulative Reward）。
3.  **舒适性 (Comfort)**：通过视频渲染阶段启用的 `show_trajectories` 观测车辆轨迹平滑度，验证 LQR 二次型惩罚对抖动（Jerk）的抑制效果。

生成的统计数据会自动落盘为 `data/summary_metrics.csv`，并利用 Matplotlib 绘制用于学术展示的**累计奖励箱线图 (Boxplot)** 与**存活率柱状图 (Bar chart)**。

## 5. 快速开始 (Quick Start)

### 依赖安装 (Dependencies)
请确保在 Python 环境中安装了以下核心依赖：
```bash
pip install torch numpy gymnasium highway-env tensorboard matplotlib pandas