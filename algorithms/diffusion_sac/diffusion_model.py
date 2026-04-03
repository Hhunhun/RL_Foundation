import torch
import torch.nn as nn
import math


# ==========================================
# 模块一：核心数学引擎 (Noise Scheduler)
# ==========================================
class NoiseScheduler:
    def __init__(self, num_train_timesteps=100, beta_start=0.0001, beta_end=0.02, device="cpu"):
        """
        线性噪声调度器。
        注意：在 RL 中为了保证在线推理速度，扩散步数 T 通常设置较小（如 15 到 100 步），
        而不是图像生成里的 1000 步。
        """
        self.num_train_timesteps = num_train_timesteps
        self.device = device

        # 1. 生成 beta 序列 (线性时间表)
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)

        # 2. 计算 alpha 序列
        self.alphas = 1.0 - self.betas

        # 3. 计算 alpha 的累乘序列 (alpha_bar)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 4. 提前计算好前向加噪公式需要的两个核心“根号项”
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, original_action, noise, timesteps):
        """
        前向过程魔法：直接一步算出加噪后的动作 a_t
        公式: a_t = sqrt(alpha_bar_t) * a_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        # 提取对应时间步的系数，并调整形状以支持批量(Batch)矩阵乘法
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].unsqueeze(1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].unsqueeze(1)

        # 计算带噪动作 a_t
        noisy_action = sqrt_alpha_prod * original_action + sqrt_one_minus_alpha_prod * noise
        return noisy_action

    @torch.no_grad()
    def sample(self, actor_net, state, action_dim, sample_steps=5):
        """
        [新增] DDIM 加速反向采样引擎
        将原本需要 num_train_timesteps (100步) 的去噪过程，极致压缩到 sample_steps (例如 5步)。

        :param actor_net: 训练好的 ConditionalActor 网络
        :param state: 当前的路况状态特征 (Batch_size, state_dim)
        :param action_dim: 动作维度 (通常为 2：油门、转向)
        :param sample_steps: 推理时的极速采样步数 (越小越快，越大越准)
        :return: 去噪完成的干净动作 a_0
        """
        device = state.device
        batch_size = state.shape[0]

        # 1. 凭空捏造一个纯随机的高斯噪声序列 a_T
        action = torch.randn((batch_size, action_dim), device=device)

        # 2. 生成跳步的时间步序列 (比如 100步 缩减到 5步: [80, 60, 40, 20, 0])
        step_indices = torch.linspace(self.num_train_timesteps - 1, 0, sample_steps, dtype=torch.long, device=device)

        # 3. 开启反向去噪大循环
        for i in range(sample_steps):
            t = step_indices[i]
            # 扩展 t 以匹配 batch_size
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # a. 让网络看着路况 s，预测当前 a_t 里面的噪点
            predicted_noise = actor_net(state, action, t_batch)

            # 获取当前步 t 和 上一步 t_prev 的 alpha_bar
            alpha_bar_t = self.alphas_cumprod[t]
            if i < sample_steps - 1:
                t_prev = step_indices[i + 1]
                alpha_bar_t_prev = self.alphas_cumprod[t_prev]
            else:
                # 到了最后一步，上一步的 alpha_bar_prev 应该完全纯净 (=1.0)
                alpha_bar_t_prev = torch.tensor(1.0, device=device)

            # b. 利用网络预测的噪点，反推最原始的干净动作 a_0
            pred_original_action = (action - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

            # 【核心工程技巧】：强行把预测出来的动作卡在 [-1, 1] 的物理边界内！
            # 这能极大概率防止 RL 训练初期的动作崩溃
            pred_original_action = torch.clamp(pred_original_action, -1.0, 1.0)

            # c. DDIM 独家公式：利用 a_0 和 噪点，直接跳步计算出 a_{t_prev}
            dir_xt = torch.sqrt(1.0 - alpha_bar_t_prev) * predicted_noise
            action = torch.sqrt(alpha_bar_t_prev) * pred_original_action + dir_xt

        # 经历 sample_steps 次跳跃后，返回最终去噪完毕的动作
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
        将一维的时间步 t 转换为高维的正弦/余弦特征向量
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
        条件扩散策略网络（修图大师）。
        输入: 状态 s, 带噪动作 a_t, 时间步 t
        输出: 预测的纯噪声 epsilon
        """
        super().__init__()

        # 时间步编码器
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.Mish(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # 状态 s 和 动作 a_t 的特征融合网络
        # 输入维度 = 状态维度 + 动作维度 + 时间特征维度
        input_dim = state_dim + action_dim + time_emb_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)  # 最终输出与动作同维度的噪声预测
        )

    def forward(self, state, action, time):
        """
        前向传播计算预测噪声
        """
        # 1. 提取时间特征
        t_emb = self.time_mlp(time)

        # 2. 将 状态(s)、带噪动作(a_t)、时间特征(t_emb) 拼接在一起
        x = torch.cat([state, action, t_emb], dim=-1)

        # 3. 经过 MLP 预测噪声
        predicted_noise = self.net(x)
        return predicted_noise