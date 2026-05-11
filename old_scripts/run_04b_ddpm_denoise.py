import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from datetime import datetime

def add_ddpm_noise(image, t_ratio):
    """
    模拟真实 Diffusion Model (DDPM) 的前向加噪过程。
    
    :param image: 原始图像 (归一化到 0.0 - 1.0)
    :param t_ratio: 当前处于总时间步的比例 (0.0 表示无噪声，1.0 表示纯噪声)
    :return: 添加噪声后的图像
    """
    if t_ratio <= 0.0:
        return image
        
    # 模拟真实 DDPM 中的 alpha_bar 衰减机制
    # t_ratio 越大，alpha_bar 越趋近于 0，原图信号被衰减得越厉害
    # 使用余弦调度 (Cosine Schedule) 的平滑衰减曲线以获得更好的视觉效果
    alpha_bar = np.cos((t_ratio + 0.008) / 1.008 * np.pi / 2) ** 2
    
    # 1. 生成标准正态分布的纯白噪声
    noise = np.random.standard_normal(size=image.shape)
    
    # 2. 按照 DDPM 公式融合原图与噪声： x_t = sqrt(alpha_bar)*x_0 + sqrt(1 - alpha_bar)*noise
    noisy_image = np.sqrt(alpha_bar) * image + np.sqrt(1.0 - alpha_bar) * noise
    
    # 3. 将其裁剪回 [0, 1] 以便正确显示
    noisy_image = np.clip(noisy_image, 0.0, 1.0)
    
    return noisy_image

def generate_diffusion_schematic(image_path, output_path="diffusion_forward.png", individual_out_dir="diffusion_steps"):
    """
    读取图像并生成一张展示加噪过程的拼接图表。
    同时将每一步加噪后的单张图片保存到 individual_out_dir 目录下。
    """
    if not os.path.exists(image_path):
        print(f"❌ 找不到图片文件: {image_path}")
        return

    # 读取图像并将其从 BGR 转换为 RGB (Matplotlib 需要 RGB)
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 归一化到 [0, 1] 范围
    image_norm = image_rgb / 255.0

    # 如果指定了单张图片的输出目录，则创建它
    if individual_out_dir:
        os.makedirs(individual_out_dir, exist_ok=True)

    # 定义我们要展示的时间步比例 (0.0 = 原图, 1.0 = 纯噪声)
    t_ratios = [0.0, 0.15, 0.40, 0.70, 1.0]
    labels = ["$x_0$ (Original)", "$x_{t_1}$", "$x_{t_2}$", "$x_{t_3}$", "$x_T$ (Pure Noise)"]

    num_steps = len(t_ratios)
    fig, axes = plt.subplots(1, num_steps, figsize=(3 * num_steps, 3.5))

    for i, (t, label) in enumerate(zip(t_ratios, labels)):
        # 生成带噪图像
        noisy_img = add_ddpm_noise(image_norm, t)
        
        # 保存单张图片
        if individual_out_dir:
            # 将 [0, 1] 的 RGB 图像转换为 [0, 255] 的 BGR 图像供 OpenCV 保存
            save_img_bgr = cv2.cvtColor((noisy_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            step_filename = os.path.join(individual_out_dir, f"noisy_step_{i}_t_{t:.2f}.png")
            cv2.imwrite(step_filename, save_img_bgr)
        
        # 绘制图像
        axes[i].imshow(noisy_img)
        axes[i].axis('off')  # 隐藏坐标轴
        axes[i].set_title(label, fontsize=16, pad=10)

    # 减少图表周围的空白边缘
    plt.tight_layout()
    
    # 保存为高分辨率图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"✅ 扩散模型原理图已生成并保存至: {output_path}")
    if individual_out_dir:
        print(f"✅ 单张加噪图片已保存至文件夹: {individual_out_dir}/")
    
    # 如果你在 Jupyter Notebook 中，可以直接取消下方注释展示出来
    # plt.show()
    plt.close()

if __name__ == "__main__":
    # 使用说明：
    # 1. 准备一张输入图片 (例如 input_image.png)
    # 2. 修改下方的路径
    # 3. 运行脚本，即可得到完美排版的原理图素材
    
    # 使用 r 前缀防止 Windows 路径中的反斜杠被转义
    input_img = r"E:\Autol_Lab\RL_Foundation\old_scripts\output_plot\AAorigin\E11.jpg"  
    
    # 提取图片名称作为关键词，并生成时间戳
    keyword = os.path.splitext(os.path.basename(input_img))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder_name = f"denoise_{keyword}_{timestamp}"
    
    # 设置统一的输出保存目录，并在其下创建本次运行的专属子文件夹
    base_out_dir = r"E:\Autol_Lab\RL_Foundation\old_scripts\output_plot"
    run_out_dir = os.path.join(base_out_dir, subfolder_name)
    os.makedirs(run_out_dir, exist_ok=True)
    
    output_img = os.path.join(run_out_dir, "diffusion_forward_schematic.png")
    individual_dir = os.path.join(run_out_dir, "diffusion_steps")
    
    # 为了演示顺利运行，如果找不到图片，可以先用随机数组模拟一张（请在实际使用时删除）
    if not os.path.exists(input_img):
        print("未检测到输入图片，正在生成一张测试用渐变图...")
        x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
        test_img = np.stack([x, y, 1-x], axis=-1) * 255
        cv2.imwrite(input_img, cv2.cvtColor(test_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
        
    generate_diffusion_schematic(input_img, output_img, individual_dir)
