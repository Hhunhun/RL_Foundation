"""
图 4-2：传统单峰 RL 的帕累托困境 (2D 散点前沿图)
核心目的：揭示传统 SAC 无法同时兼顾“安全”与“效率”，
所有基线模型都在一条“向右下倾斜的边界线”内苦苦挣扎。
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle

# ---------------------------------------------------------
# 0. 全局路径与样式配置
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 高级视觉控制面板 (Visual Style Control Panel)
PLOT_STYLE = {
    "scatter_alpha": 0.6,          # 散点的不透明度
    "scatter_size": 150,            # 散点的大小
    "pareto_line_alpha": 0.8,       # 帕累托边界虚线的不透明度
    "pareto_fill_alpha": 0.1,       # 帕累托边界下方“能力域”填充区域的透明度 (调大颜色变深)
    "target_circle_alpha": 0.7,     # 右上角理想目标区虚线圆圈的透明度
    "legend_loc": "upper left",     # 图例锚点对齐方式
    "legend_bbox": (0.02, 0.98),    # 图例的相对位置偏移 (X, Y)，调整为左上角
}

def set_publication_style():
    """全局学术期刊图表样式配置滤镜 (继承自 run_03)"""
    custom_params = {
        # --- 字体配置 ---
        "font.family": "serif",           # 强制使用衬线字体
        "font.serif": ["SimSun", "Times New Roman", "STSong", "Songti SC", "SimHei", "sans-serif"], # 中英文字体回退栈
        "mathtext.fontset": "stix",       # 数学公式字体样式，与 Times New Roman 协调
        "font.size": 14,                  # 全局基础字号
        
        # --- 字号层级控制 ---
        "axes.titlesize": 18,             # 图表主标题字号
        "axes.labelsize": 16,             # 坐标轴(X/Y)标签字号
        "xtick.labelsize": 14,            # X轴刻度数字字号
        "ytick.labelsize": 14,            # Y轴刻度数字字号
        "legend.fontsize": 12,            # 图例文字字号
        
        # --- 线条与外框 ---
        "axes.edgecolor": "black",        # 图表外框颜色
        "axes.linewidth": 1.5,            # 图表外框粗细
        "axes.unicode_minus": False,      # 解决负号显示为方块的乱码问题
        
        # --- 画幅与导出 ---
        "figure.figsize": (9.0, 6.5),     # 画幅大小 (宽, 高)
        "figure.dpi": 300,                # 导出图片的高清分辨率
        "savefig.bbox": "tight"           # 保存时自动裁剪多余的空白边距
    }
    sns.set_theme(style="whitegrid", rc=custom_params)

# ---------------------------------------------------------
# 1. 数据来源与视觉映射配置
# ---------------------------------------------------------
# 硬编码内置提取出的性能数据 [存活率, 平均速度]
HARDCODED_DATA = {
    "merge-v0": [
        {"id": "M04", "survival": 100.0, "speed": 18.6, "role": "保守"},
        {"id": "M01", "survival": 100.0, "speed": 19.0, "role": "基线"},
        {"id": "M03", "survival": 98.0,  "speed": 19.7, "role": "激进"}
    ],
    "racetrack-v0": [
        {"id": "R05", "survival": 27.0,  "speed": 15.7, "role": "保守"},
        {"id": "R01", "survival": 45.0,  "speed": 19.3, "role": "基线"},
        {"id": "R02", "survival": 30.0,  "speed": 18.4, "role": "激进"}
    ],
    "highway-v0": [
        {"id": "H02", "survival": 98.0,  "speed": 20.5, "role": "保守"},
        {"id": "H01", "survival": 94.0,  "speed": 22.0, "role": "基线"},
        {"id": "H03", "survival": 42.0,  "speed": 27.5, "role": "激进"}
    ]
}

# 专属角色颜色映射
ROLE_COLORS = {
    "保守": "#009688",  # 深青色
    "基线": "#78909c",  # 灰蓝色
    "激进": "#d32f2f"   # 红色
}

# 环境场景形状映射
ENV_MARKERS = {
    "merge-v0": "o",       # 圆形 (Merge)
    "racetrack-v0": "^",   # 正三角形 (Racetrack)
    "highway-v0": "s"      # 正方形 (Highway)
}

ENV_NAMES_CN = {
    "merge-v0": "匝道汇入",
    "racetrack-v0": "极限赛道",
    "highway-v0": "高速巡航"
}

def plot_traditional_pareto():
    set_publication_style()
    fig, ax = plt.subplots()
    
    all_points = [] # 用于计算帕累托前沿
    
    print("==================================================")
    print("📊 正在渲染图 4-2：传统 SAC 帕累托散点图...")
    
    # 2. 遍历内置字典绘制散点
    for env, models in HARDCODED_DATA.items():
        for m in models:
            mid = m["id"]
            x = m["survival"]
            y = m["speed"]
            color = ROLE_COLORS[m["role"]]
            marker = ENV_MARKERS[env]
            
            # 绘制带黑色边框的散点
            ax.scatter(x, y, s=PLOT_STYLE["scatter_size"], c=color, marker=marker, edgecolors='black', linewidths=1.5, alpha=PLOT_STYLE["scatter_alpha"], zorder=4)
            
            # 添加避免重叠的文本标签
            ax.annotate(mid, (x, y), xytext=(0, 12), textcoords='offset points', 
                        ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=5)
            
            all_points.append((x, y))

    # 3. 核心计算：提取纯物理意义上的“帕累托前沿” (Pareto Frontier)
    # 按存活率从大到小排序，若存活率相同按速度从大到小
    all_points.sort(key=lambda p: (p[0], p[1]), reverse=True) 
    pareto_front = []
    max_y = -float('inf')
    
    for p in all_points:
        if p[1] >= max_y:
            pareto_front.append(p)
            max_y = p[1]
            
    pareto_front.reverse() # 翻转回按 x 轴从小到大排序
    px = [p[0] for p in pareto_front]
    py = [p[1] for p in pareto_front]
    
    # 向左向右拓展出完整的包络线
    px_env = [0] + px + [105]
    # 计算最后一段自然的倾斜率进行右下延伸
    tail_slope = (py[-1] - py[-2]) / (px[-1] - px[-2]) if len(px) > 1 else -0.1
    py_tail = py[-1] + tail_slope * (105 - px[-1])
    py_env = [py[0]] + py + [py_tail]

    # 4. 绘制“能力天花板”边界线与灰底阴影挣扎区
    ax.plot(px_env, py_env, linestyle='--', color='dimgray', linewidth=2.0, alpha=PLOT_STYLE["pareto_line_alpha"], zorder=2)
    ax.fill_between(px_env, 10, py_env, color='gray', alpha=PLOT_STYLE["pareto_fill_alpha"], zorder=1) # 底边填到10
    
    # 边界线学术说明
    ax.text(70, 16.5, "传统策略能力域", color='dimgray', fontsize=16, fontweight='bold', alpha=0.8, ha='center', zorder=2)

    # 5. 右上角：理想目标区圆圈
    target_circle = Circle((98, 28), radius=3.5, edgecolor='gray', facecolor='none', linestyle='--', linewidth=2, alpha=PLOT_STYLE["target_circle_alpha"], zorder=2)
    ax.add_patch(target_circle)
    ax.text(98, 28, "理想目标区\n(安全且高效)", ha='center', va='center', color='dimgray', fontsize=12, fontweight='bold', zorder=3)

    # 6. 图例与坐标轴严格限制
    ax.set_xlabel("存活率（%）", fontweight='bold', labelpad=10)
    ax.set_ylabel("平均速度（m/s）", fontweight='bold', labelpad=10)
    # ax.set_title("传统单峰 SAC 算法的帕累托极限困境", pad=15, fontweight='bold') # 应要求移除大标题
    
    ax.set_xlim(0, 105)
    ax.set_ylim(10, 32)

    # 构建分离式的精致图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=ENV_NAMES_CN["merge-v0"], markerfacecolor='gray', markersize=10, markeredgecolor='black', alpha=PLOT_STYLE["scatter_alpha"]),
        Line2D([0], [0], marker='^', color='w', label=ENV_NAMES_CN["racetrack-v0"], markerfacecolor='gray', markersize=10, markeredgecolor='black', alpha=PLOT_STYLE["scatter_alpha"]),
        Line2D([0], [0], marker='s', color='w', label=ENV_NAMES_CN["highway-v0"], markerfacecolor='gray', markersize=10, markeredgecolor='black', alpha=PLOT_STYLE["scatter_alpha"]),
        Line2D([0], [0], color='w', label='  '), # 空白占位符
        Line2D([0], [0], marker='o', color='w', label='保守策略', markerfacecolor=ROLE_COLORS["保守"], markersize=10, markeredgecolor='black', alpha=PLOT_STYLE["scatter_alpha"]),
        Line2D([0], [0], marker='o', color='w', label='基线策略', markerfacecolor=ROLE_COLORS["基线"], markersize=10, markeredgecolor='black', alpha=PLOT_STYLE["scatter_alpha"]),
        Line2D([0], [0], marker='o', color='w', label='激进策略', markerfacecolor=ROLE_COLORS["激进"], markersize=10, markeredgecolor='black', alpha=PLOT_STYLE["scatter_alpha"]),
    ]
    # 放在左中偏上位置，避开图表右上方的理想目标区和点群
    ax.legend(handles=legend_elements, loc=PLOT_STYLE["legend_loc"], bbox_to_anchor=PLOT_STYLE["legend_bbox"], fontsize=11, framealpha=0.9, edgecolor='black')

    # 7. 保存导出
    out_dir = os.path.join(PROJECT_ROOT, "old_scripts", "output_plot", "plot4_2")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "Figure_4-2_Traditional_SAC_Pareto.png")
    plt.savefig(save_path)
    print(f"✅ 图 4-2 (传统 SAC 帕累托散点图) 已成功生成：\n📁 {save_path}")

if __name__ == "__main__":
    plot_traditional_pareto()