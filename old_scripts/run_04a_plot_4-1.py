"""
第四章 动机图表：帕累托困境 (Pareto Dilemma)

核心目的：痛批传统强化学习在自动驾驶中的“要命还是要速度”的绝对对立。
呈现方式：柱线混合双轴图 (Bar-Line Dual Axis)
"""

import os
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def set_publication_style():
    """配置顶刊级别的图表视觉规范"""
    custom_params = {
        # --- 字体配置 (Font Configuration) ---
        "font.family": "serif",           # 强制全局使用衬线体
        "font.serif": ["SimSun", "Times New Roman", "STSong", "SimHei", "sans-serif"], # 中英文字体回退栈
        "mathtext.fontset": "stix",       # 数学公式字体 (与 Times 风格匹配)
        "font.size": 16,                  # 全局基础字号
        
        # --- 字号层级控制 (Font Sizes) ---
        "axes.titlesize": 18,             # 图表主标题字号
        "axes.labelsize": 18,             # 坐标轴(X/Y)标题字号
        "xtick.labelsize": 16,            # X 轴刻度数字字号
        "ytick.labelsize": 16,            # Y 轴刻度数字字号
        "legend.fontsize": 14,            # 图例文字字号
        
        # --- 线条与外框 (Lines & Spines) ---
        "lines.linewidth": 2.5,           # 全局线宽 (影响折线图粗细)
        "axes.edgecolor": "black",        # 图表外框颜色
        "axes.linewidth": 1.5,            # 图表外框粗细
        "axes.unicode_minus": False,      # 解决负号显示为方块的乱码问题
        
        # --- 画幅与导出 (Figure & Output) ---
        "figure.figsize": (12.0, 6.5),    # 画幅大小 (宽, 高) - 更宽的画幅容纳三大环境
        "figure.dpi": 300,                # 导出图片分辨率 (学术标准通常为300或600)
        "savefig.bbox": "tight"           # 保存时自动裁剪边缘多余留白
    }
    sns.set_theme(style="white", rc=custom_params) # 去掉默认的灰色网格，双轴图背景越干净越好

# ==========================================
# 🎨 图表元素定制参数 (Visual Element Configuration)
# ==========================================
PLOT_STYLE = {
    # --- 柱状图 (Bar Chart) ---
    "bar_alpha": 0.6,               # 柱体填充透明度 (0.0~1.0，1.0 为完全不透明)
    "bar_width": 0.7,               # 柱体宽度 (调大使得组内柱子更紧密，调小则间距增大)
    "bar_edgecolor": "none",       # 柱体外框颜色 (设为 'none' 或 None 可取消边框)
    "bar_linewidth": 1.2,           # 柱体外框粗细

    # --- 间距控制 (Spacing) ---
    "intra_spacing": 1.0,           # 组内相邻柱子的中心间距
    "group_spacing": 1.5,           # 组与组之间的间距 (上一组最后一个柱子到下一组首个柱子的距离)

    # --- 折线图 (Line Chart) ---
    "line_color": "black",          # 存活率折线颜色
    "line_width": 1.0,              # 存活率折线粗细
    "line_style": "-",              # 存活率折线样式 (实线'-', 虚线'--', 点划线'-.')

    # --- 标记点 (Marker) ---
    "marker_style": "o",            # 数据点形状 ('o'圆点, 's'方块, '^'三角, 'D'菱形)
    "marker_size": 6,               # 数据点大小
    "marker_facecolor": "#EEAF49",     # 数据点内部填充色
    "marker_edgecolor": "none",    # 数据点外框颜色
    "marker_edgewidth": 1.5,        # 数据点外框粗细

    "marker_text_fontsize": 14,     # 折线图数据点文字字号
    "marker_text_color": "black",   # 折线图数据点文字颜色
    "marker_text_offset_x": -2,      # 折线图数据点文字的水平像素偏移量
    "marker_text_offset_y": -10,    # 折线图数据点文字的垂直像素偏移量 (负数表示在点下方)

    # --- 文本标注 (Text Annotation) ---
    "bar_text_fontsize": 14,        # 柱状图顶部文字字号
    "bar_text_offset": 0.3,         # 柱状图顶部文字距离柱体上边框的绝对偏移量
    "bar_text_color": "black",      # 柱状图顶部文字颜色
    "env_text_fontsize": 16,        # 底部环境大标签(如 Highway-v0)字号
    
    # --- 坐标轴范围 (Axis Limits) ---
    "y_speed_min": 0,              # 左轴(平均速度)最小值
    "y_speed_max": 32,              # 左轴(平均速度)最大值
    "y_surv_min": 0,                # 右轴(存活率)最小值
    "y_surv_max": 105,              # 右轴(存活率)最大值
}

# ==========================================
# 核心实验设计数据结构
# ==========================================
ENV_CONFIGS = [
    {
        "env": "merge-v0",
        "display_name": "匝道汇入（merge-v0）",
        "models": ["M04", "M01", "M03"]
    },
    {
        "env": "racetrack-v0",
        "display_name": "赛道竞速（racetrack-v0）",
        "models": ["R05", "R01", "R02"]
    },
    {
        "env": "highway-v0",
        "display_name": "高速巡航（highway-v0）",
        "models": ["H02", "H01", "H03"]
    }
]

# 组内统一标签
GROUP_LABELS = ['保守策略', '基线策略', '激进策略']

# 采用 run03 中柔和高级的 Nature 风格配色
COLOR_SAFE = '#00A087'  # 翠绿色：让人安心的安全色
COLOR_BASE = '#8491B4'  # 莫灰紫：中规中矩的基线
COLOR_EFF  = '#E64B35'  # 胭脂红：具备警示感且视觉舒适
BAR_COLORS = [COLOR_SAFE, COLOR_BASE, COLOR_EFF]

# ==========================================
# 直接指定数据文件路径，取消全盘扫描以节省时间
# ==========================================
HARDCODED_PKL_PATHS = {
    "merge-v0": r"E:\Autol_Lab\RL_Foundation\outputs\merge-v0\eval_results\[M01_M02_M03_M04_M05_M06_M07_M08_DM01_DM02_DM03_DM04_DM05_DM06_DM07_DM08]_20260516_021146\data\all_results.pkl",
    "racetrack-v0": r"E:\Autol_Lab\RL_Foundation\outputs\racetrack-v0\eval_results\[R01_R02_R03_R04_R05_R06_R07_R08_DR01_DR02_DR03_DR04_DR05_DR06_DR07_DR08]_20260516_125952\data\all_results.pkl",
    "highway-v0": r"E:\Autol_Lab\RL_Foundation\outputs\highway-v0\eval_results\[H01_H02_H03_H04_DH01_DH02_DH03_DH04_DH05_DH06_DH07_DH08]_20260516_152551\data\all_results.pkl"
}

def get_eval_data():
    """
    智能数据搜集器：去 outputs 目录下搜刮最新的 all_results.pkl
    如果对应模型缺失（尚未跑评估），则填充一套展示“帕累托困境”的默认演示数据。
    """
    all_data = {}
    
    # [演示用底线数据]：完美契合你“立靶子”的逻辑
    dummy_data = {
        "M01": {"speed": 16.5, "survival": 92.0}, "M03": {"speed": 21.0, "survival": 55.0}, "M04": {"speed": 14.8, "survival": 99.0},
        "R01": {"speed": 18.0, "survival": 85.0}, "R02": {"speed": 24.5, "survival": 28.0}, "R05": {"speed": 17.2, "survival": 96.0},
        "H01": {"speed": 22.0, "survival": 94.0}, "H03": {"speed": 27.5, "survival": 42.0}, "H02": {"speed": 20.5, "survival": 98.5},
    }

    for cfg in ENV_CONFIGS:
        env = cfg["env"]
        models_needed = set(cfg["models"])
        
        pkl_file = HARDCODED_PKL_PATHS.get(env, "")
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                env_results = pickle.load(f)
            
            found_in_this_file = []
            for mid in models_needed:
                if mid in env_results:
                    all_data[mid] = {
                        "speed": env_results[mid]["mean_speed"],
                        "survival": env_results[mid]["survival_rate"]
                    }
                    found_in_this_file.append(mid)
                    
            for mid in found_in_this_file:
                models_needed.remove(mid)
                        
        # 如果依然有没找全的模型，直接用 dummy 数据补齐，保证画图不崩
        for mid in models_needed:
            all_data[mid] = dummy_data[mid]
            print(f"⚠️ 未找到 [{mid}] 的真实评估数据，已自动填充演示数据。")
                
    return all_data

def plot_pareto_dilemma():
    set_publication_style()
    data = get_eval_data()
    
    # ==========================================
    # 新增：将数据落盘为 CSV 并重新读取
    # ==========================================
    out_dir = os.path.join(PROJECT_ROOT, "old_scripts", "output_plot", "plot4_1")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pareto_data.csv")
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Speed", "Survival"])
        for mid, vals in data.items():
            writer.writerow([mid, vals["speed"], vals["survival"]])
            
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = {row["Model"]: {"speed": float(row["Speed"]), "survival": float(row["Survival"])} for row in reader}

    fig, ax_speed = plt.subplots()
    ax_surv = ax_speed.twinx() # 建立第二 Y 轴
    
    # ==========================================
    # 1. 坐标与数据组装
    # ==========================================
    # 动态计算分组 x 坐标
    intra_spacing = PLOT_STYLE["intra_spacing"]
    group_spacing = PLOT_STYLE["group_spacing"]
    
    x_positions = []
    group_centers = []
    current_x = 1.0
    
    for i in range(len(ENV_CONFIGS)):
        group_centers.append(current_x + intra_spacing)
        for j in range(3):
            x_positions.append(current_x)
            if j < 2: current_x += intra_spacing
        if i < len(ENV_CONFIGS) - 1:
            current_x += group_spacing
    
    speeds = []
    survivals = []
    for cfg in ENV_CONFIGS:
        for mid in cfg["models"]:
            speeds.append(data[mid]["speed"])
            survivals.append(data[mid]["survival"])
            
    # ==========================================
    # 2. 绘制左轴：平均速度 (柱状体)
    # ==========================================
    bars = ax_speed.bar(x_positions, speeds, width=PLOT_STYLE["bar_width"], 
                        color=BAR_COLORS * 3, alpha=PLOT_STYLE["bar_alpha"],
                        edgecolor=PLOT_STYLE["bar_edgecolor"], linewidth=PLOT_STYLE["bar_linewidth"], 
                        zorder=3)
    
    ax_speed.set_ylabel('平均速度 (m/s)', fontsize=16, fontweight='bold')
    ax_speed.tick_params(axis='y', labelsize=14)
    
    # 左侧主Y轴范围锁死，让所有柱子从底线升起
    ax_speed.set_ylim(PLOT_STYLE["y_speed_min"], PLOT_STYLE["y_speed_max"])
    
    # 在柱子上标注具体数值
    for bar in bars:
        yval = bar.get_height()
        ax_speed.text(bar.get_x() + bar.get_width()/2, yval + PLOT_STYLE["bar_text_offset"], 
                      f'{yval:.1f}', ha='center', va='bottom', fontsize=PLOT_STYLE["bar_text_fontsize"], fontweight='bold', color=PLOT_STYLE["bar_text_color"])

    # ==========================================
    # 3. 绘制右轴：存活率 (折线图)
    # ==========================================
    # 采用分段连线，不在环境与环境的空白之间画线
    for i in range(3):
        idx = slice(i*3, i*3+3)
        ax_surv.plot(x_positions[idx], survivals[idx], 
                     color=PLOT_STYLE["line_color"], linewidth=PLOT_STYLE["line_width"], 
                     linestyle=PLOT_STYLE["line_style"], zorder=4)
        # 叠加圆点
        ax_surv.plot(x_positions[idx], survivals[idx], 
                     marker=PLOT_STYLE["marker_style"], markersize=PLOT_STYLE["marker_size"], 
                     color=PLOT_STYLE["line_color"], markerfacecolor=PLOT_STYLE["marker_facecolor"], 
                     markeredgecolor=PLOT_STYLE["marker_edgecolor"], markeredgewidth=PLOT_STYLE["marker_edgewidth"], 
                     linewidth=0, zorder=5)
                     
    ax_surv.set_ylabel('存活率 (%)', fontsize=16, fontweight='bold')
    ax_surv.tick_params(axis='y', labelsize=14)
    
    # [绝对核心] 存活率轴锁死
    ax_surv.set_ylim(PLOT_STYLE["y_surv_min"], PLOT_STYLE["y_surv_max"]) 
    
    # 为暴跌的折线点添加数据标注
    for i, (x, y) in enumerate(zip(x_positions, survivals)):
        # 统一将数值放置在数据点的下方，通过参数区调整间距
        ax_surv.annotate(f'{y:.0f}', xy=(x, y), 
                         xytext=(PLOT_STYLE["marker_text_offset_x"], PLOT_STYLE["marker_text_offset_y"]), textcoords='offset points',
                         ha='center', va='top', fontsize=PLOT_STYLE["marker_text_fontsize"], 
                         fontweight='bold', color=PLOT_STYLE["marker_text_color"])

    # ==========================================
    # 4. 设计底部 X 轴的层级标签
    # ==========================================
    ax_speed.set_xticks(x_positions)
    ax_speed.set_xticklabels(GROUP_LABELS * 3, fontsize=11, rotation=0, ha='center')
    
    # 隐藏常规的 X 轴外框线，准备手动画分割线
    ax_speed.spines['bottom'].set_visible(False)
    ax_speed.tick_params(axis='x', length=0) # 隐藏刻度线

    # 在 X 轴更下方画出各环境的大标签 (移除干扰横线)
    y_transform = ax_speed.get_xaxis_transform()
    for center, cfg in zip(group_centers, ENV_CONFIGS):
        ax_speed.text(center, -0.07, cfg["display_name"], 
                      ha='center', va='top', fontsize=PLOT_STYLE["env_text_fontsize"], fontweight='bold', transform=y_transform)

    # ==========================================
    # 5. 设计分离式高级图例
    # ==========================================
    # [核心修复] 将图例统一横向排列并顶置，绝对不遮挡任何数据
    legend_elements = [
        Patch(facecolor=COLOR_SAFE, edgecolor=PLOT_STYLE["bar_edgecolor"], alpha=PLOT_STYLE["bar_alpha"], label='保守策略'),
        Patch(facecolor=COLOR_BASE, edgecolor=PLOT_STYLE["bar_edgecolor"], alpha=PLOT_STYLE["bar_alpha"], label='基线基线'),
        Patch(facecolor=COLOR_EFF, edgecolor=PLOT_STYLE["bar_edgecolor"], alpha=PLOT_STYLE["bar_alpha"], label='激进策略'),
        Line2D([0], [0], color=PLOT_STYLE["line_color"], 
               marker=PLOT_STYLE["marker_style"], markerfacecolor=PLOT_STYLE["marker_facecolor"], 
               markeredgecolor=PLOT_STYLE["marker_edgecolor"], markersize=PLOT_STYLE["marker_size"], 
               linewidth=PLOT_STYLE["line_width"], label='存活率 (%)')
    ]
    ax_speed.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.02),
                    ncol=4, frameon=False, columnspacing=1.0)

    # 给图表加个横向引导网格线，仅依附于左侧速度轴
    ax_speed.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    # ==========================================
    # 6. 保存出图
    # ==========================================
    plt.tight_layout() # 强制收缩边缘白边
    save_path = os.path.join(out_dir, "04_pareto_dilemma.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n📊 帕累托图表对应的数据已保存至: {csv_path}")
    print(f"\n🎉 帕累托困境靶向图绘制完成！请查看: {save_path}")

if __name__ == "__main__":
    plot_pareto_dilemma()