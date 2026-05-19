"""
图 4-9：Diff-SAC 突破帕累托边界气泡图 (2D Bubble Pareto Plot)
核心目的：展示 Diff-SAC 算法在三大环境中打破了传统 SAC 的帕累托边界，
向“又快又稳”的右上角飞升。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle
from adjustText import adjust_text
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------
# 0. 全局路径与样式配置
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 高级视觉控制面板 (Visual Style Control Panel)
PLOT_STYLE = {
    "scatter_alpha": 0.6,          # 散点的不透明度
    "scatter_size": 150,            # 散点的大小
    "scatter_edge_width": 0.2,      # 散点及图例标记的外框粗细
    "pareto_line_alpha": 0.8,       # 帕累托边界虚线的不透明度
    "pareto_fill_alpha": 0.1,       # 帕累托边界下方“能力域”填充区域的透明度 (调大颜色变深)
    "target_circle_alpha": 0.7,     # 右上角理想目标区虚线圆圈的透明度
    "legend_loc": "upper left",     # 图例锚点对齐方式
    "legend_bbox": (0.02, 0.98),    # 图例的相对位置偏移 (X, Y)，调整为左上角
    
    # --- 标签位置与角度微调接口 ---
    "pareto_text_x": 80,            # “传统策略能力域”文字的中心横坐标
    "pareto_text_y_offset": -1.2,   # “传统策略能力域”文字相对于虚线的纵向高度偏移
    "pareto_text_rotation": -22,    # “传统策略能力域”文字的倾斜旋转角度
    "target_circle_x": 98,          # “理想目标区”文字与圆圈的中心横坐标
    "target_circle_y": 28,          # “理想目标区”虚线圆圈的中心纵坐标
    "target_text_y_offset": 2,      # “理想目标区”文字相对于圆圈中心的纵向高度偏移量
    "target_circle_r": 3.5,         # “理想目标区”虚线圆圈的半径大小
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
        "axes.labelsize": 18,             # 坐标轴(X/Y)标签字号
        "xtick.labelsize": 16,            # X轴刻度数字字号
        "ytick.labelsize": 16,            # Y轴刻度数字字号
        "legend.fontsize": 14,            # 图例文字字号
        
        # --- 线条与外框 ---
        "axes.edgecolor": "black",        # 图表外框颜色
        "axes.linewidth": 1.0,            # 图表外框粗细
        "axes.unicode_minus": False,      # 解决负号显示为方块的乱码问题
        
        # --- 散点与标记外框 (Markers Edge) ---
        "scatter.edgecolors": "black",    # 散点图标记的外框颜色
        "lines.markeredgecolor": "black", # 图例标记的外框颜色
        "lines.markeredgewidth": PLOT_STYLE["scatter_edge_width"], # 标记外框的默认粗细
        
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
        {"id": "M03", "survival": 98.0,  "speed": 19.7, "role": "激进"},
        {"id": "DM06", "survival": 100.0, "speed": 27.5, "role": "SOTA"} # 请按真实数据调整
    ],
    "racetrack-v0": [
        {"id": "R05", "survival": 27.0,  "speed": 15.7, "role": "保守"},
        {"id": "R01", "survival": 45.0,  "speed": 19.3, "role": "基线"},
        {"id": "R02", "survival": 30.0,  "speed": 18.4, "role": "激进"},
        {"id": "DR06", "survival": 98.5,  "speed": 28.2, "role": "SOTA"} # 请按真实数据调整
    ],
    "highway-v0": [
        {"id": "H02", "survival": 98.0,  "speed": 20.5, "role": "保守"},
        {"id": "H01", "survival": 94.0,  "speed": 22.0, "role": "基线"},
        {"id": "H03", "survival": 42.0,  "speed": 27.5, "role": "激进"},
        {"id": "DH06", "survival": 99.0,  "speed": 26.8, "role": "SOTA"} # 请按真实数据调整
    ]
}

# 专属角色颜色映射
ROLE_COLORS = {
    "保守": "#009688",  # 深青色
    "基线": "#78909c",  # 灰蓝色
    "激进": "#d32f2f",  # 红色
    "SOTA": "#FFC107"   # 琥珀金 (极其吸睛，代表突破边界的 Diff-SAC)
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
    texts = []      # 用于收集所有的文本标签对象以便 adjustText 自动排布
    
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
            
            # 绘制散点 (动态接入外框粗细接口)
            ax.scatter(x, y, s=PLOT_STYLE["scatter_size"], c=color, marker=marker, linewidths=PLOT_STYLE["scatter_edge_width"], alpha=PLOT_STYLE["scatter_alpha"], zorder=4)
            
            # 将文本对象收集起来，不立即指定偏移量，交由 adjustText 处理
            # texts.append(ax.text(x, y, mid, ha='center', va='center', fontsize=11, fontweight='bold', zorder=5)) # 已应要求隐去模型简称
            
            all_points.append((x, y))

    # 启用 adjust_text 自动施加排斥力，解决散点密集区的文本叠印问题
    # adjust_text(texts, 
    #             expand_points=(1.5, 1.5), 
    #             arrowprops=dict(arrowstyle="-", color='dimgray', alpha=0.8, lw=1.2)) # 由于取消了简称，一并注销避让组件

    # 3. 拟合严谨的帕累托平滑包络线 (Strictly Monotonic Pareto Envelope)
    # 提取关键外侧极值点，并在此基础上沿 Y 轴向上偏移 0.5 留出包络余量
    # 使用 PchipInterpolator (单调三次插值)，确保曲线绝对单调递减，杜绝伪包络的跳跃现象
    anchors_x = [0.0,  42.0, 94.0, 98.0, 100.0, 105.0]
    anchors_y = [31.0, 28.0, 22.5, 21.0, 19.5,  16.0]
    
    pchip = PchipInterpolator(anchors_x, anchors_y)
    px_env = np.linspace(0, 105, 200)
    py_env = pchip(px_env)

    # 4. 绘制“能力天花板”边界线与灰底阴影挣扎区
    ax.plot(px_env, py_env, linestyle='--', color='dimgray', linewidth=2.0, alpha=PLOT_STYLE["pareto_line_alpha"], zorder=2)
    ax.fill_between(px_env, 10, py_env, color='gray', alpha=PLOT_STYLE["pareto_fill_alpha"], zorder=1) # 底边填到10
    
    # 边界线学术说明
    # 自动计算边界线上某个合适点的法线方向旋转角度，使文字完美贴合曲线走向
    px_text = PLOT_STYLE["pareto_text_x"]
    py_text = pchip(px_text) + PLOT_STYLE["pareto_text_y_offset"]
    ax.text(px_text, py_text, "传统策略极限能力域", color='dimgray', fontsize=15, fontweight='bold', 
            alpha=0.9, ha='center', rotation=PLOT_STYLE["pareto_text_rotation"], zorder=2)

    # 5. 右上角：理想目标区圆圈
    cx = PLOT_STYLE["target_circle_x"]
    cy = PLOT_STYLE["target_circle_y"]
    target_circle = Circle((cx, cy), radius=PLOT_STYLE["target_circle_r"], edgecolor='gray', facecolor='none', linestyle='--', linewidth=2, alpha=PLOT_STYLE["target_circle_alpha"], zorder=2)
    ax.add_patch(target_circle)
    ax.text(cx, cy + PLOT_STYLE["target_text_y_offset"], "理想目标区\n(安全且高效)", ha='center', va='center', color='dimgray', fontsize=12, fontweight='bold', zorder=3)

    # 6. 图例与坐标轴严格限制
    ax.set_xlabel("存活率（%）", fontweight='bold', labelpad=10)
    ax.set_ylabel("平均速度（m/s）", fontweight='bold', labelpad=10)
    # ax.set_title("传统单峰 SAC 算法的帕累托极限困境", pad=15, fontweight='bold') # 应要求移除大标题
    
    ax.set_xlim(0, 105)
    ax.set_ylim(10, 32)
    
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # 构建分离式的精致图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=ENV_NAMES_CN["merge-v0"], markerfacecolor='gray', markersize=10, alpha=PLOT_STYLE["scatter_alpha"]),
        Line2D([0], [0], marker='^', color='w', label=ENV_NAMES_CN["racetrack-v0"], markerfacecolor='gray', markersize=10, alpha=PLOT_STYLE["scatter_alpha"]),
        Line2D([0], [0], marker='s', color='w', label=ENV_NAMES_CN["highway-v0"], markerfacecolor='gray', markersize=10, alpha=PLOT_STYLE["scatter_alpha"]),
        Line2D([0], [0], color='w', label='  '), # 空白占位符
        Line2D([0], [0], marker='o', color='w', label='保守策略', markerfacecolor=ROLE_COLORS["保守"], markersize=10, alpha=PLOT_STYLE["scatter_alpha"]),
        Line2D([0], [0], marker='o', color='w', label='基线策略', markerfacecolor=ROLE_COLORS["基线"], markersize=10, alpha=PLOT_STYLE["scatter_alpha"]),
        Line2D([0], [0], marker='o', color='w', label='激进策略', markerfacecolor=ROLE_COLORS["激进"], markersize=10, alpha=PLOT_STYLE["scatter_alpha"]),
        Line2D([0], [0], marker='o', color='w', label='Diff-SAC', markerfacecolor=ROLE_COLORS["SOTA"], markersize=10, markeredgecolor='black', alpha=0.95),
    ]
    # 放在左中偏上位置，避开图表右上方的理想目标区和点群
    ax.legend(handles=legend_elements, loc=PLOT_STYLE["legend_loc"], bbox_to_anchor=PLOT_STYLE["legend_bbox"], fontsize=11, framealpha=0.9, edgecolor='black')

    # 7. 保存导出
    out_dir = os.path.join(PROJECT_ROOT, "old_scripts", "output_plot", "plot4_9")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "Figure_4-9_DiffSAC_Pareto_Bubble.png")
    plt.savefig(save_path)
    print(f"✅ 图 4-9 (终极帕累托气泡图) 已成功生成：\n📁 {save_path}")

if __name__ == "__main__":
    plot_traditional_pareto()