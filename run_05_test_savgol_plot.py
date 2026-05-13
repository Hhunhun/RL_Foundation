"""
测试脚本：Savgol 滤波平滑与双层折线图渲染 (基于 TensorBoard 数据)
结合了 run_04 的数据抽取、配色、排版规范，以及 SciPy Savitzky-Golay 滤波器的画图方式。
"""

import os
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.signal import savgol_filter  # 引入 Savitzky-Golay 平滑函数

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def set_publication_style():
    """
    全局学术期刊图表样式配置滤镜 (继承自 run_04)
    """
    custom_params = {
        "font.family": "serif",
        "font.serif": ["SimSun", "Times New Roman", "STSong", "Songti SC", "SimHei", "PingFang SC", "sans-serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
        "lines.linewidth": 2.0,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "axes.unicode_minus": False,
        "figure.figsize": (8.0, 6.0),
        "figure.dpi": 300,
        "savefig.bbox": "tight"
    }
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(custom_params)


def extract_tb_data(log_dir, target_tags, align_to_env_steps=False):
    """
    提取 TensorBoard 数据 (继承自 run_04)
    """
    if isinstance(target_tags, str):
        target_tags = [target_tags]
        
    try:
        ea = EventAccumulator(log_dir, size_guidance={'scalars': 0})
        ea.Reload()
        tb_tags = ea.Tags().get('scalars', [])
        
        found_tag = None
        for t in target_tags:
            if t in tb_tags:
                found_tag = t
                break
                
        if not found_tag:
            return None, None
            
        events = ea.Scalars(found_tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        
        if align_to_env_steps:
            len_tag = 'Train/Steps' if 'Train/Steps' in tb_tags else ('Metrics/Episode_Steps' if 'Metrics/Episode_Steps' in tb_tags else None)
            if len_tag:
                len_events = ea.Scalars(len_tag)
                mapping = {}
                cum = 0
                for e in len_events:
                    cum += e.value
                    mapping[e.step] = cum
                    
                is_already_steps = False
                if len(mapping) > 0:
                    max_map_key = max(mapping.keys())
                    if len(steps) > 0 and max(steps) > max_map_key * 5:
                        is_already_steps = True
                        
                if not is_already_steps:
                    steps = [mapping.get(s, mapping.get(min(mapping.keys(), key=lambda k: abs(k - s)) if mapping else s, s)) for s in steps]
                    
        return steps, vals
    except Exception as e:
        print(f"❌ 解析 {log_dir} 失败: {e}")
        return None, None


def plot_savgol_curves(models_dict, save_dir, tags_to_plot, align_to_env_steps=False, custom_limits=None):
    """
    使用 Savitzky-Golay 滤波器和透明底线样式的核心绘图函数
    """
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)
    
    # 采用《Nature》顶级期刊高对比度色系 (NPG Academic Palette)
    academic_colors = [
        "#8491B4", "#91D1C2", "#E64B35", "#4DBBD5", "#00A087", 
        "#3C5488", "#F39B7F", "#DC0000", "#7E6148"
    ]
    colors = sns.color_palette(academic_colors, n_colors=len(models_dict))

    for target_tags, y_label in tags_to_plot:
        plt.figure()
        global_max_step = 0
        has_data = False
        
        all_smoothed_vals = []
        
        for (label_name, log_paths), color in zip(models_dict.items(), colors):
            if isinstance(log_paths, str): log_paths = [log_paths]
            
            # 本测试脚本仅提取传入的第一个日志路径，以此演示基础的 Savgol 画法
            log_dir = os.path.join(PROJECT_ROOT, log_paths[0])
            if not os.path.exists(log_dir):
                continue
                
            steps, vals = extract_tb_data(log_dir, target_tags, align_to_env_steps)
            if steps is not None and vals is not None and len(steps) > 0:
                has_data = True
                global_max_step = max(global_max_step, steps[-1])
                
                # --- 使用 Savitzky-Golay 滤波器进行平滑处理 ---
                # 动态计算 window_length: 必须是奇数，且不能超过数据总长度
                wl = min(51, len(vals)) 
                if wl % 2 == 0: wl -= 1
                
                if wl > 3:
                    smoothed_vals = savgol_filter(vals, window_length=wl, polyorder=2)
                else:
                    smoothed_vals = vals # 数据太少时不平滑

                all_smoothed_vals.append(smoothed_vals)

                # [自定义画法] 1. 绘制原始折线图（透明度降低，细线）
                plt.plot(steps, vals, linestyle='-', color=color, alpha=0.3, linewidth=0.8)
                # [自定义画法] 2. 绘制平滑后的折线图（实线主体，带有图例）
                plt.plot(steps, smoothed_vals, linestyle='-', color=color, linewidth=2.0, label=label_name)
        
        if has_data:
            plt.ylabel(y_label)
            ax = plt.gca()
            
            # 强制限定奖励与步数类图表的 Y 轴区间从 0 开始
            if "奖励" in y_label or "步数" in y_label or "Q值" in y_label:
                ax.set_ylim(bottom=0)
                    
            # 智能美化最大横坐标显示，使用向上取整 (ceil) 确保不会裁剪末端最新数据
            if global_max_step >= 10000:
                display_max = int(np.ceil(global_max_step / 1000.0) * 1000)
            elif global_max_step >= 100:
                display_max = int(np.ceil(global_max_step / 10.0) * 10)
            else:
                display_max = int(global_max_step)
                
            ax.set_xlim(0, display_max)
            
            # 限制横纵坐标刻度标签数量不超过6个 (nbins=5 生成最多6个刻度)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            
            # --- 智能 X 轴动态格式化 ---
            if global_max_step >= 10000:
                x_label_text = "环境交互步数" if align_to_env_steps else "训练步数"
                plt.xlabel(f"{x_label_text} ($10^3$ Steps)")
                def thousands_formatter(x, pos):
                    val = x / 1000.0
                    if val.is_integer(): return f'{int(val)}'
                    else: return f'{val:.2f}'.rstrip('0').rstrip('.')
                ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
            else:
                x_label_text = "环境交互步数" if align_to_env_steps else "训练回合"
                plt.xlabel(f"{x_label_text} ({'Steps' if align_to_env_steps else 'Episodes'})")
                    
            # 应用个性化坐标轴裁剪窗口
            if custom_limits and y_label in custom_limits:
                lims = custom_limits[y_label]
                if lims.get("xlim") is not None: ax.set_xlim(lims["xlim"])
                if lims.get("ylim") is not None: ax.set_ylim(lims["ylim"])
                
            # 智能图例位置：优先左上角，若左侧数据高于右侧（如下降的Loss曲线），则放右上角避免遮挡
            try:
                left_val = np.mean([np.mean(v[:max(1, len(v)//3)]) for v in all_smoothed_vals if len(v) > 0])
                right_val = np.mean([np.mean(v[-max(1, len(v)//3):]) for v in all_smoothed_vals if len(v) > 0])
                legend_loc = "upper left" if left_val <= right_val else "upper right"
            except Exception:
                legend_loc = "upper left"
            plt.legend(loc=legend_loc, fontsize=12)
            
            # 增加网格 (保持与提供代码的半透明度一致)
            plt.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            save_filename = f"[Savgol测试] {y_label}"
            # 同时保存为 png 和 svg 格式
            plt.savefig(os.path.join(save_dir, f"{save_filename}.png"), dpi=300)
            plt.savefig(os.path.join(save_dir, f"{save_filename}.svg"), format='svg', bbox_inches='tight')
            print(f"✅ 生成完成: {save_filename}")
        
        plt.close()


if __name__ == "__main__":
    print("🤖 启动 Savgol 滤波测试脚本...")
    TARGET_ENV = "racetrack-v0"
    
    # 测试使用的模型 (直接从 TensorBoard 读取数据)
    models_to_plot = {
        #"SAC 专家底座": f"outputs/{TARGET_ENV}/logs/SAC_R05_SAC_Smooth_Racing_20260505_131614",
        "DR01 纯 BC 克隆": f"outputs/{TARGET_ENV}/logs/DiffSAC_DR01_Pure_BC_20260506_013340",
        "DR02 微引导": f"outputs/{TARGET_ENV}/logs/DiffSAC_DR02_Micro_Q_20260506_021118",
        #"DR03 标准引导": f"outputs/{TARGET_ENV}/logs/DiffSAC_DR03_Standard_Q_20260506_025209",
    }
    
    tags_to_plot = [
        (["Reward/Episode_Reward", "Train/Reward"], "全局环境交互奖励"),  
    ]

    CUSTOM_AXES_LIMITS = {
        "全局环境交互奖励": {"xlim": None, "ylim": None}, # 解除 30 分的硬编码限制，让 Y 轴根据实际数据完全自适应
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(PROJECT_ROOT, "outputs", TARGET_ENV, "train_results", f"Savgol_Test_{timestamp}")

    print("\n正在绘制 Savgol 滤波曲线...")
    plot_savgol_curves(models_to_plot, save_directory, tags_to_plot, align_to_env_steps=True, custom_limits=CUSTOM_AXES_LIMITS)
    print(f"\n📈 测试图表已保存至: {save_directory}")