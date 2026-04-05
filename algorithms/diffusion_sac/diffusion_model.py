"""
条件扩散模型核心引擎 (Conditional Diffusion Model Core)

此模块实现了用于强化学习的扩散模型底座，主要包含四个关键组件：
1. NoiseScheduler (噪声调度器): 负责数学层面的前向加噪（破坏动作）和反向去噪（生成动作）过程。
2. SinusoidalPosEmb (位置编码): 将时间步 t 转化为神经网络能理解的高维特征。
3. ConditionalActor (条件去噪网络): 整个扩散模型的大脑，负责根据当前状态和时间步，预测并剔除动作中的噪声。
4. EMAModel (指数移动平均): 稳定器，用于平滑网络权重的更新，防止生成的动作发生剧烈抖动。
"""

import math
import copy
import torch
import torch.nn as nn


# ==========================================
# 模块一：核心数学引擎 (Noise Scheduler)
# ==========================================
class NoiseScheduler:
    def __init__(self, num_train_timesteps=100, beta_start=0.0001, beta_end=0.02):
        """
        线性噪声调度器。
        它决定了“破坏数据”的节奏：从第一步的微小噪声 (beta_start) 逐渐增加到最后一步的明显噪声 (beta_end)。
        传统的图像生成通常需要 1000 步，这里为了满足 RL 在线交互的实时性（不能让车等太久），我们将其压缩到了 100 步。
        """
        self.num_train_timesteps = num_train_timesteps

        # 预先计算好整套加噪过程所需的数学系数 (betas 和 alphas)
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # alpha 的累乘积，代表保留原始信息的比例

        # 预先计算出公式中常用的平方根部分，加速后续的前向与反向计算
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, original_action, noise, timesteps):
        """
        前向过程 (Forward Process)：给专家动作一步到位地添加特定时间步 t 的噪声。
        在训练 (BC) 阶段，我们用这个函数把干净的专家动作变脏，然后让网络去猜加了什么噪声。
        """
        device = original_action.device

        # 提取对应时间步的加噪系数
        # 🚨 修复核心：必须先将常量张量移动到 target device，然后再用 target device 上的 timesteps 去索引
        sqrt_alpha_prod = self.sqrt_alphas_cumprod.to(device)[timesteps].unsqueeze(1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps].unsqueeze(1)

        # 按照公式合成带噪动作: a_t = sqrt(alpha_bar) * a_0 + sqrt(1 - alpha_bar) * epsilon
        noisy_action = sqrt_alpha_prod * original_action + sqrt_one_minus_alpha_prod * noise
        return noisy_action

    @torch.no_grad()
    def sample(self, actor_net, state, action_dim, sample_steps=5, eta=0.0):
        """
        反向采样过程 (Reverse Sampling)：从纯随机噪声中，一步步雕刻出有意义的驾驶动作。
        这里使用了 DDIM 加速算法，允许我们跳步采样（例如从 100 步压缩到 5 步出动作）。

        :param actor_net: 去噪策略网络 (也就是下方的 ConditionalActor 或其 EMA 版本)
        :param state: 环境状态，作为去噪的“条件”告诉网络当前在什么样的路况下
        :param action_dim: 动作的维度 (比如油门、转向)
        :param sample_steps: 推理时的实际迭代步数 (越小越快，越大越准)
        :param eta: 探索系数。eta=0 代表绝对理性的最优解(评估用)，eta>0 会保留一点随机性用于探索新动作(训练用)
        """
        device = state.device
        batch_size = state.shape[0]

        # 1. 抓取一把纯随机的白噪声作为动作的起点
        action = torch.randn((batch_size, action_dim), device=device)

        # 2. 规划跳步采样的路径 (例如：[99, 79, 59, 39, 19])
        step_indices = torch.linspace(self.num_train_timesteps - 1, 0, sample_steps, dtype=torch.long, device=device)

        # 3. 开始循环去噪
        for i in range(sample_steps):
            t = step_indices[i]
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # 让神经网络看着当前的路况 (state)，猜一下当前的动作里包含多少噪声
            predicted_noise = actor_net(state, action, t_batch)

            alpha_bar_t = self.alphas_cumprod.to(device)[t]
            if i < sample_steps - 1:
                t_prev = step_indices[i + 1]
                alpha_bar_t_prev = self.alphas_cumprod.to(device)[t_prev]
            else:
                alpha_bar_t_prev = torch.tensor(1.0, device=device)

            # 剥离预测出的噪声，看一眼“完全干净的动作”大概长什么样 (pred_a0)
            pred_original_action = (action - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

            # 推理阶段使用边界截断，防止由于数学计算误差导致动作突破物理极限 (比如方向盘打过头)
            pred_original_action = torch.clamp(pred_original_action, -5.0, 5.0)

            # 如果 eta > 0，计算我们需要主动保留多少随机方差用于探索
            if eta > 0.0 and i < sample_steps - 1:
                variance = (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_t_prev)
                std_dev = eta * torch.sqrt(variance)
                noise = torch.randn_like(action)
            else:
                std_dev = 0.0
                noise = 0.0

            # 计算指向前一步 (更干净一点) 动作的方向向量
            dir_xt = torch.sqrt(1.0 - alpha_bar_t_prev - std_dev ** 2) * predicted_noise

            # 重新组装成上一步的动作，进入下一轮循环
            action = torch.sqrt(alpha_bar_t_prev) * pred_original_action + dir_xt + std_dev * noise

        return action


# ==========================================
# 模块二：时间位置编码 (Positional Embedding)
# ==========================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        生成正弦/余弦时间步特征。
        神经网络很难直接理解 "数字 5" 和 "数字 50" 的物理差异。
        这个模块将单纯的数字时间步 t，映射成一组像波浪一样交替的、具有丰富数学特征的向量，
        让网络清楚地知道自己目前正处于去噪过程的哪个阶段（初期、中期还是末期）。
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ==========================================
# 模块三：条件去噪神经网络 (Conditional Actor)
# ==========================================
class ConditionalActor(nn.Module):
    def __init__(self, state_dim, action_dim, time_emb_dim=64, hidden_dim=256):
        """
        接收环境状态、带噪动作与时间步的策略网络（相当于图像扩散模型里的 U-Net）。
        它的唯一工作就是输出：在当前时间步下，动作里混入了多少噪声。
        """
        super().__init__()

        # 处理时间特征的小型多层感知机 (MLP)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.Mish(),  # Mish 激活函数在扩散模型中通常比 ReLU 表现更平滑
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        input_dim = state_dim + action_dim + time_emb_dim

        # 主干网络：将状态、动作、时间三者融合，预测噪声
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)  # 输出层维度与动作维度一致（因为我们要预测的是加在动作上的噪声）
        )

    def forward(self, state, action, time):
        # 将时间步数值转化为高维特征
        t_emb = self.time_mlp(time)
        # 将状态条件、当前动作和时间特征拼接在一起作为输入
        x = torch.cat([state, action, t_emb], dim=-1)
        return self.net(x)


# ==========================================
# 模块四：指数移动平均模型 (EMA Wrapper)
# ==========================================
class EMAModel:
    """
    维护目标网络权重的指数移动平均版本。

    在训练过程中，神经网络的权重每次更新都会发生跳变。扩散模型对这种跳变非常敏感，
    容易导致每次生成的动作差异很大（车子疯狂打方向盘）。
    EMA 相当于一个反应迟钝的“影子网络”，它缓慢地吸收主网络的知识。我们在推理
    生成动作时，使用的是这个平滑的 EMA 模型，从而保证动作的连贯性和安全性。
    """

    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()  # EMA 模型只用于推理，关闭梯度相关的模式

    def update(self, source_model):
        """使用源模型(正在激烈训练的主网络)的参数，对 EMA 影子模型进行软更新"""
        with torch.no_grad():
            for ema_param, source_param in zip(self.model.parameters(), source_model.parameters()):
                # 按照 decay 比例混合：绝大部分保留过去的历史，吸收一小部分新知识
                ema_param.data.copy_(self.decay * ema_param.data + (1.0 - self.decay) * source_param.data)

    def __call__(self, *args, **kwargs):
        """魔法方法：让我们能像调用普通网络一样直接调用 EMA 对象 (如 ema_actor(state, action, t))"""
        return self.model(*args, **kwargs)