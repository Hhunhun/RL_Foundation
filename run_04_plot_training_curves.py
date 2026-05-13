"""
阶段五：训练曲线高清渲染器 (Training Curves Plotting)

此模块专为撰写学术论文设计。
它可以直接解析 TensorBoard 生成的事件日志文件 (.tfevents)，
提取包括 Loss、Reward 在内的过程数据，并复刻 TensorBoard 的 EMA 平滑算法。
最终输出全矢量 (PDF) 与高分辨率 (PNG) 图表，支持多个模型在同一张图中对比。
"""

import os
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib.ticker import FuncFormatter, MaxNLocator

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def set_publication_style():
    """
    全局学术期刊图表样式配置滤镜，与 evaluate 脚本保持绝对统一。
    """
    # Matplotlib 和 Seaborn 的默认参数字典，用于统一图表外观
    custom_params = {
        # --- 字体配置 (中英混排：英文 Times New Roman，中文宋体) ---
        "font.family": "serif",           # 强制全局使用衬线字体
        "font.serif": ["SimSun", "Times New Roman", "STSong", "Songti SC", "SimHei", "PingFang SC", "sans-serif"], # 修复方块Bug：优先宋体，避免 TNR 阻断回退
        "mathtext.fontset": "stix",       # 确保公式中的 $10^3$ 也是类似 Times 的衬线体风格
        "font.size": 12,                  # 全局基础字号，所有文本元素的默认大小
        
        # --- 轴与标签字号配置 ---
        "axes.titlesize": 18,             # 图表主标题的字号 (调大以增强可读性)
        "axes.labelsize": 18,             # 坐标轴标签的字号 (调大以增强可读性)
        "xtick.labelsize": 16,            # X轴刻度标签的字号 (调大以增强可读性)
        "ytick.labelsize": 16,            # Y轴刻度标签的字号 (调大以增强可读性)
        "legend.fontsize": 12,            # 图例文字的字号
        
        # --- 线条与外框配置 ---
        "lines.linewidth": 2.0,           # 全局线宽，适用于所有 plt.plot 绘制的线条，但此脚本中平滑曲线会被单独设置
        "axes.edgecolor": "black",        # 强制显示坐标轴的黑色外框
        "axes.linewidth": 1.2,            # 坐标轴外框的线宽 (略微加粗，更符合双栏排版)
        "axes.unicode_minus": False,      # 解决负号在中文环境下可能显示为方块的乱码问题
        
        # --- 输出保存配置 ---
        "figure.figsize": (8.0, 6.0),     # 默认画幅：宽 8.0 英寸，高 6.0 英寸 (保持 4:3 比例，更适合 12 号字体的展示)
                                          # 常见学术论文单栏图宽 3.5 英寸，双栏图宽 7.0 英寸，可按需调整
        "figure.dpi": 300,                # 默认分辨率 (学术期刊通常要求 300 DPI 或 600 DPI)
        "savefig.bbox": "tight"           # 保存时自动裁剪图表周围多余的空白边距，确保紧凑排版
    }
    
    # 应用全局样式
    sns.set_theme(style="whitegrid")
    # 强制覆盖：Seaborn 有时会吞掉 font 列表导致 fallback 失效，直接更新 rcParams 最稳妥
    plt.rcParams.update(custom_params)


def smooth_curve(scalars, weight=0.95):
    """
    复刻 TensorBoard 的指数移动平均 (EMA) 平滑算法。
    weight 越大越平滑 (通常设为 0.8 ~ 0.95 之间)。
    """
    if not scalars:
        return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def extract_tb_data(log_dir, target_tags, align_to_env_steps=False):
    """
    从指定的 TensorBoard 文件夹中抽取标量数据。
    支持传入单一 Tag 或等价 Tag 列表。
    若 align_to_env_steps=True，则尝试将回合数(Episode)换算为环境物理步数(Environment Steps)。
    """
    if isinstance(target_tags, str):
        target_tags = [target_tags]
        
    try:
        # size_guidance=0 表示强制加载所有数据点，防止图表被截断失真
        ea = EventAccumulator(log_dir, size_guidance={'scalars': 0})
        ea.Reload()
        tb_tags = ea.Tags().get('scalars', [])
        
        found_tag = None
        for t in target_tags:
            if t in tb_tags:
                found_tag = t
                break
                
        if not found_tag:
            # 静默跳过，避免多模型不兼容时疯狂报错
            return None, None
            
        events = ea.Scalars(found_tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        
        # --- 核心对齐换算 ---
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
                    # 根据累计步数替换原本的 Episode X轴
                    steps = [mapping.get(s, mapping.get(min(mapping.keys(), key=lambda k: abs(k - s)) if mapping else s, s)) for s in steps]
                    
        return steps, vals
    except Exception as e:
        print(f"❌ 解析 {log_dir} 失败: {e}")
        return None, None


def plot_training_curves(models_dict, save_dir, tags_to_plot, smooth_weight=0.85, align_to_env_steps=False, custom_limits=None):
    """
    核心绘图函数。
    支持多随机种子(Multiple Seeds)聚合。若传入单个路径使用 EMA 滚动方差；若传入路径列表，自动计算 Median + IQR。
    """
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 采用《Nature》顶级期刊高对比度色系 (NPG Academic Palette) ---
    # 特点：色彩纯度高，半透明叠加时不会发灰变脏，依然保持极高的类别辨识度
    academic_colors = [
        "#8491B4", # 莫灰紫 (Slate Purple)
        "#91D1C2", # 薄荷青 (Mint)
        "#E64B35", # 胭脂红 (Carmine Red)
        "#4DBBD5", # 蔚蓝色 (Cerulean Blue)
        "#00A087", # 翠绿色 (Teal Green)
        "#3C5488", # 午夜蓝 (Midnight Blue)
        "#F39B7F", # 珊瑚粉 (Salmon Pink)
        "#DC0000", # 深红色 (Crimson)
        "#7E6148", # 咖啡褐 (Coffee Brown)
    ]
    colors = sns.color_palette(academic_colors, n_colors=len(models_dict))

    for target_tags, y_label in tags_to_plot:
        plt.figure()
        
        has_data = False
        global_max_step = 0 # 记录所有模型中的最大横坐标，用于统一 X 轴
        global_max_smoothed = 0 # 记录所有模型平滑后的最高均值，用于约束 Y 轴
        
        plot_cache = [] # 缓存池：为了能在全盘了解全局极值后再执行真正的绘制
        
        # ==========================================
        # 第一阶段：遍历所有模型，抽取数据，计算平滑和动态对齐参数
        # ==========================================
        for (label_name, log_paths), color in zip(models_dict.items(), colors):
            # 兼容单路径(str)与多路径列表(list)
            if isinstance(log_paths, str):
                log_paths = [log_paths]
                
            all_steps = []
            all_vals = []
            
            # 遍历该模型下的所有随机种子日志
            for rel_path in log_paths:
                log_dir = os.path.join(PROJECT_ROOT, rel_path)
                if not os.path.exists(log_dir):
                    print(f"⚠️ 找不到日志路径，跳过该种子: {log_dir}")
                    continue
                steps, vals = extract_tb_data(log_dir, target_tags, align_to_env_steps)
                if steps is not None and vals is not None and len(steps) > 0:
                    all_steps.append(steps)
                    all_vals.append(vals)
                    
            if not all_steps:
                continue
                
            has_data = True
            
            # --- 智能算法分支 ---
            if len(all_steps) == 1:
                # 【单种子模式】：使用我们之前的 EMA 极值压平滚动方差策略
                steps = all_steps[0]
                vals = all_vals[0]
                smoothed_main = smooth_curve(vals, weight=smooth_weight)
                
                squared_errors = [(v - s)**2 for v, s in zip(vals, smoothed_main)]
                smoothed_variances = smooth_curve(squared_errors, weight=0.99) # 强力压平
                smoothed_devs = [np.sqrt(max(0, var)) for var in smoothed_variances]
                
                lower_bound = [s - d for s, d in zip(smoothed_main, smoothed_devs)]
                upper_bound = [s + d for s, d in zip(smoothed_main, smoothed_devs)]
            else:
                # 【多种子模式】：顶会标配 Median + IQR 四分位距
                # 1. 寻找重叠的 X 轴公共空间，并插值对齐 (Interpolation)
                min_x = max([min(s) for s in all_steps])
                max_x = min([max(s) for s in all_steps])
                steps = np.linspace(min_x, max_x, 1000).tolist() # 建立 1000 个对齐的参考刻度
                
                interp_vals = []
                for s, v in zip(all_steps, all_vals):
                    interp_vals.append(np.interp(steps, s, v))
                interp_vals = np.array(interp_vals)
                
                # 2. 在各个刻度上计算跨种子的数学统计量，彻底免疫离群值
                median_vals = np.median(interp_vals, axis=0).tolist()
                p25_vals = np.percentile(interp_vals, 25, axis=0).tolist()
                p75_vals = np.percentile(interp_vals, 75, axis=0).tolist()
                
                # 3. 对统计线进行最后一道视觉平滑
                smoothed_main = smooth_curve(median_vals, weight=smooth_weight)
                lower_bound = smooth_curve(p25_vals, weight=smooth_weight)
                upper_bound = smooth_curve(p75_vals, weight=smooth_weight)
                
            global_max_step = max(global_max_step, steps[-1])
            global_max_smoothed = max(global_max_smoothed, max(smoothed_main))
            
            plot_cache.append({
                "label": label_name, "color": color, 
                "steps": steps, "smoothed_vals": smoothed_main, 
                "lower_bound": lower_bound, "upper_bound": upper_bound
            })
        
        # ==========================================
        # 第二阶段：利用全局极值参数作图，并执行“时空延展补齐”
        # ==========================================
        if has_data:
            for data in plot_cache:
                steps = data["steps"]
                smoothed_vals = data["smoothed_vals"]
                lower_bound = data["lower_bound"]
                upper_bound = data["upper_bound"]
                
                plt.fill_between(steps, 
                                 lower_bound, 
                                 upper_bound, 
                                 color=data["color"], alpha=0.10, linewidth=0)
                                 
                plt.plot(steps, smoothed_vals, color=data["color"], alpha=1.0, linewidth=1.5, label=data["label"])
        
            plt.ylabel(y_label)
            
            ax = plt.gca()
            
            # [核心修复 3] 强制限定奖励、步数以及Q值类图表的 Y 轴区间从 0 开始
            if "奖励" in y_label or "步数" in y_label or "Q值" in y_label:
                # 限定上限为“所有平滑曲线最高均值的 1.2 倍”，彻底拒绝被偶尔失控的背景阴影拉伸 Y 轴
                dynamic_top = global_max_smoothed * 1.20
                if dynamic_top > 0:
                    ax.set_ylim(bottom=0, top=dynamic_top)
                else:
                    ax.set_ylim(bottom=0)
                    
            # 智能美化最大横坐标显示
            if global_max_step >= 10000:
                display_max = int(round(global_max_step / 1000.0) * 1000)
            elif global_max_step >= 100:
                display_max = int(round(global_max_step / 10.0) * 10)
            else:
                display_max = int(global_max_step)
                
            ax.set_xlim(0, display_max)
            
            # 限制横纵坐标刻度标签数量不超过6个 (nbins=5 生成最多6个刻度)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            
            # --- 智能 X 轴动态格式化 ---
            if align_to_env_steps:
                if global_max_step >= 10000:
                    plt.xlabel("环境交互步数 ($10^3$ Steps)")
                    def thousands_formatter(x, pos):
                        val = x / 1000.0
                        if val.is_integer(): return f'{int(val)}'
                        else: return f'{val:.2f}'.rstrip('0').rstrip('.')
                    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
                else:
                    plt.xlabel("环境交互步数 (Steps)")
            else:
                if global_max_step >= 10000:
                    plt.xlabel("训练步数 ($10^3$ Steps)")
                    def thousands_formatter(x, pos):
                        val = x / 1000.0
                        if val.is_integer(): return f'{int(val)}'
                        else: return f'{val:.2f}'.rstrip('0').rstrip('.')
                    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
                else:
                    plt.xlabel("训练回合 (Episodes)")
                    
            # ==========================================
            # [新功能] 独立应用个性化坐标轴裁剪窗口
            # ==========================================
            if custom_limits and y_label in custom_limits:
                lims = custom_limits[y_label]
                # 支持传入 (min, max)，如果某一边不想设死，可以传 None，例如 (0, None)
                if lims.get("xlim") is not None:
                    ax.set_xlim(lims["xlim"])
                if lims.get("ylim") is not None:
                    ax.set_ylim(lims["ylim"])
                
            # 智能图例位置：优先左上角，若左侧数据高于右侧（如下降的Loss曲线），则放右上角避免遮挡
            try:
                left_val = np.mean([np.mean(d["smoothed_vals"][:max(1, len(d["smoothed_vals"])//3)]) for d in plot_cache if len(d["smoothed_vals"]) > 0])
                right_val = np.mean([np.mean(d["smoothed_vals"][-max(1, len(d["smoothed_vals"])//3):]) for d in plot_cache if len(d["smoothed_vals"]) > 0])
                legend_loc = "upper left" if left_val <= right_val else "upper right"
            except Exception:
                legend_loc = "upper left"
            plt.legend(loc=legend_loc)

            plt.tight_layout()
            # 增加文件名前缀区分
            save_filename = f"[时空对齐] {y_label}" if align_to_env_steps else y_label
            plt.savefig(os.path.join(save_dir, f"{save_filename}.png"), dpi=300)
            # 暂时关闭 PDF 导出，以提升出图速度，后续需要写论文时可取消注释
            # plt.savefig(os.path.join(save_dir, f"{save_filename}.pdf"), format='pdf', bbox_inches='tight')
            print(f"✅ 生成完成: {save_filename}")
        
        plt.close()


if __name__ == "__main__":
    print("🤖 启动训练曲线渲染器...")
    # ==========================================
    # 终端交互：选择评估环境
    # ==========================================
    print("==========================================")
    print("[H] Highway 环境 (highway-v0)")
    print("[M] Merge 环境 (merge-v0)")
    print("[R] Racetrack 环境 (racetrack-v0)")
    env_choice = input("👉 请选择评估环境 (H, M 或 R，默认 H): ").strip().upper()
    if env_choice == 'M':
        TARGET_ENV = "merge-v0"
    elif env_choice == 'R':
        TARGET_ENV = "racetrack-v0"
    else:
        TARGET_ENV = "highway-v0"
    print(f"✅ 已锁定评估环境: {TARGET_ENV}")
    print("==========================================")
    
    # ==========================================
    # 📍 配置待对比的模型及其 TensorBoard 日志路径
    # 注意：这里需要指向带有 events.out.tfevents.* 文件的底层 logs 目录
    # ==========================================
    if TARGET_ENV == "merge-v0":
        models_to_plot = {
            "SAC 稳健基准 (M4)": f"outputs/{TARGET_ENV}/logs/SAC_M4_Safety_First_20260420_170911",
            # "DM01 纯 BC 克隆": f"outputs/{TARGET_ENV}/logs/DiffSAC_DM01_Pure_BC_...",
            # "DM02 微引导": f"outputs/{TARGET_ENV}/logs/DiffSAC_DM02_Micro_Q_...",
            # "DM03 标准引导": f"outputs/{TARGET_ENV}/logs/DiffSAC_DM03_Standard_Q_...",
            # "DM04 强力干预": f"outputs/{TARGET_ENV}/logs/DiffSAC_DM04_Strong_Q_...",
        }
    elif TARGET_ENV == "racetrack-v0":
        models_to_plot = {
            # === SAC 消融矩阵 (挑选极具代表性的策略) ===
            "R01 基础 SAC": f"outputs/{TARGET_ENV}/logs/SAC_R01_SAC_Baseline_20260505_033212",
            "R03 安全优先": f"outputs/{TARGET_ENV}/logs/SAC_R03_SAC_Safety_Priority_20260505_083207",
            #"R04 极限漂移": f"outputs/{TARGET_ENV}/logs/SAC_R04_SAC_Extreme_Drift_20260505_110254",
            #"R05 平滑基准": f"outputs/{TARGET_ENV}/logs/SAC_R05_SAC_Smooth_Racing_20260505_131614",
            #"R08 专家底座": f"outputs/{TARGET_ENV}/logs/SAC_R08_SAC_Expert_Pro_20260505_184949",
            
            # === 第一期 diff-SAC 实验 ===
            "DR01 纯 BC 克隆": f"outputs/{TARGET_ENV}/logs/DiffSAC_DR01_Pure_BC_20260506_013340",
            "DR02 微引导": f"outputs/{TARGET_ENV}/logs/DiffSAC_DR02_Micro_Q_20260506_021118",
            #"DR03 标准引导": f"outputs/{TARGET_ENV}/logs/DiffSAC_DR03_Standard_Q_20260506_025209",
            #"DR04 强力干预": f"outputs/{TARGET_ENV}/logs/DiffSAC_DR04_Strong_Q_20260506_032647",
            
            # === 第二期 Diff-SAC 混合专家实验 ===
            #"DR05 混合纯BC": f"outputs/{TARGET_ENV}/logs/DiffSAC_DR05_Mixed_BC_20260506_144146",
            #"DR06 混合微引导": f"outputs/{TARGET_ENV}/logs/DiffSAC_DR06_Mixed_Micro_Q_20260506_150630",
            #"DR07 混合标引导": f"outputs/{TARGET_ENV}/logs/DiffSAC_DR07_Mixed_Standard_Q_20260506_152944",
            #"DR08 混合强干预": f"outputs/{TARGET_ENV}/logs/DiffSAC_DR08_Mixed_Strong_Q_20260506_154916",
        }
    else: # highway-v0
        models_to_plot = {
            "SAC 安全基准 (H5)": f"outputs/{TARGET_ENV}/logs/SAC_H5_20260330_135449",
            # "Diff-SAC 极微引导 (DH5)": f"outputs/{TARGET_ENV}/logs/DH5_20260406_023023",
            # "Diff-SAC 强力引导 (DH3)": f"outputs/{TARGET_ENV}/logs/DH3_20260405_101704",
        }
    
    # ==========================================
    # 📍 配置你想从 TensorBoard 里拔出来的图表 Tag 列表
    # 格式: ("TensorBoard内部标识名", "图表展示的Y轴中文标签")
    # ==========================================
    # 1. 原始各个指标 (保留原生单位)
    raw_tags_to_plot = [
        ("Reward/Episode_Reward", "回合累计奖励"),  # (Episode Reward)
        # ("Loss/Actor", "策略网络损失"),  # [已注释] 节省时间，论文不放
        # ("Loss/Critic", "价值网络损失"),  # [已注释] 节省时间，论文不放
        # ("Loss/Alpha", "温度系数损失"),  # [已注释] 节省时间，论文不放
        # ("Schedules/Jerk_Weight", "平滑惩罚权重调度"), # [已注释] (若存在)
        # ("Schedules/Learning_Rate", "学习率衰减调度"), # [已注释] (若存在)
        ("Metrics/Alpha_Value", "温度系数值"),  # (Alpha Value)
        ("Metrics/Episode_Steps", "回合步数"),  # (Episode Steps)
        # --- Diff-SAC 专属指标 ---
        ("Train/Reward", "在线微调奖励"),  # (Train Reward)
        # ("Train/Steps", "在线微调步数"),  # [已注释] 节省时间，论文不放
        ("Metric/Q_Value", "Q值评估"),  # (Q Value)
    ]
    
    # 2. 专门用于将异构算法同框对比的时空对齐指标
    aligned_tags_to_plot = [
        # 列表第一项为需要聚合搜寻的 Tag 数组
        (["Reward/Episode_Reward", "Train/Reward"], "全局环境交互奖励"),  
        (["Metrics/Episode_Steps", "Train/Steps"], "全局存活步数"), 
    ]

    # ==========================================
    # 🎨 独立图表坐标轴范围自定义 (个性化裁剪窗口)
    # ==========================================
    # 格式: "图表Y轴中文标签": {"xlim": (xmin, xmax), "ylim": (ymin, ymax)}
    # 如果不想限制某一张图或某一个轴，可以直接写 None。
    CUSTOM_AXES_LIMITS = {
        "全局环境交互奖励": {"xlim": None, "ylim": None},  # 示例: 无论阴影多高，强行把画幅顶部锁死在 30 分
        "全局存活步数": {"xlim": None, "ylim": (0, 250)},          # 示例: 不做干预，完全自适应
        "回合累计奖励": {"xlim": None, "ylim": None},      # 示例: 只看前 400 局的表现
        "Q值评估": {"xlim": None, "ylim": (0, None)},          # 示例: 锁定下限为0，上限自适应
    }

    # ==========================================
    # 动态构建保存路径：train_results/[短代号_短代号]_时间戳
    # ==========================================
    short_names = []
    for key in models_to_plot.keys():
        # 智能提取短代号：优先提取括号内的内容，若无括号则按空格分割取第一部分 (例如从 "DR01 纯 BC 克隆" 提取 "DR01")
        match = re.search(r'\((.*?)\)', key)
        short_names.append(match.group(1) if match else key.split(" ")[0])
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"[{'_'.join(short_names)}]_{timestamp}"
    save_directory = os.path.join(PROJECT_ROOT, "outputs", TARGET_ENV, "train_results", folder_name)

    print("\n[1/2] 正在绘制各模型原生训练曲线...")
    plot_training_curves(models_to_plot, save_directory, raw_tags_to_plot, smooth_weight=0.90, align_to_env_steps=False, custom_limits=CUSTOM_AXES_LIMITS)
    
    print("\n[2/2] 正在绘制 SAC 与 Diff-SAC 统一 X 轴的时空对齐对比曲线...")
    plot_training_curves(models_to_plot, save_directory, aligned_tags_to_plot, smooth_weight=0.90, align_to_env_steps=True, custom_limits=CUSTOM_AXES_LIMITS)
    
    print(f"\n📈 所有原生及对齐对比图表已保存至: {save_directory}")