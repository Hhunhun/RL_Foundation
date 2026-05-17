"""
提取 表 4-2：基准算法综合性能汇总表
用于第 4.2 节立靶子：揭示传统单峰 RL 的帕累托困境
"""
import os
import pickle
import csv

# ---------------------------------------------------------
# 硬编码的数据路径 (与 run_03 和 run_04a 保持一致)
# ---------------------------------------------------------
HARDCODED_PKL_PATHS = {
    "merge-v0": r"E:\Autol_Lab\RL_Foundation\outputs\merge-v0\eval_results\[M01_M02_M03_M04_M05_M06_M07_M08_DM01_DM02_DM03_DM04_DM05_DM06_DM07_DM08]_20260516_021146\data\all_results.pkl",
    "racetrack-v0": r"E:\Autol_Lab\RL_Foundation\outputs\racetrack-v0\eval_results\[R01_R02_R03_R04_R05_R06_R07_R08_DR01_DR02_DR03_DR04_DR05_DR06_DR07_DR08]_20260516_125952\data\all_results.pkl",
    "highway-v0": r"E:\Autol_Lab\RL_Foundation\outputs\highway-v0\eval_results\[H01_H02_H03_H04_DH01_DH02_DH03_DH04_DH05_DH06_DH07_DH08]_20260516_152551\data\all_results.pkl"
}

# ---------------------------------------------------------
# 定义表 4-2 需要出场的基准模型及学术标签
# ---------------------------------------------------------
TARGET_MODELS = {
    "merge-v0": [
        {"id": "M04", "role": "保守/核心老师", "name": "SAC-安全约束 (M04)"},
        {"id": "M01", "role": "基础/原生锚点", "name": "SAC-标准基线 (M01)"},
        {"id": "M03", "role": "激进/反面教材", "name": "SAC-效率导向 (M03)"}
    ],
    "racetrack-v0": [
        {"id": "R05", "role": "保守/核心老师", "name": "SAC-平顺专家 (R05)"},
        {"id": "R01", "role": "基础/原生锚点", "name": "SAC-标准基线 (R01)"},
        {"id": "R02", "role": "激进/反面教材", "name": "SAC-效率导向 (R02)"}
    ],
    "highway-v0": [
        {"id": "H02", "role": "保守/核心老师", "name": "SAC-安全约束 (H02)"},
        {"id": "H01", "role": "基础/原生锚点", "name": "SAC-标准基线 (H01)"},
        {"id": "H03", "role": "激进/反面教材", "name": "SAC-效率导向 (H03)"}
    ]
}

def extract_table_data():
    print("==================================================")
    print("📊 正在提取 表 4-2：基准算法综合性能汇总表 数据...")
    print("==================================================")
    
    extracted_data = []
    
    for env, models in TARGET_MODELS.items():
        pkl_path = HARDCODED_PKL_PATHS.get(env)
        if not pkl_path or not os.path.exists(pkl_path):
            print(f"⚠️ 找不到环境 {env} 的数据文件: {pkl_path}")
            continue
            
        with open(pkl_path, 'rb') as f:
            env_results = pickle.load(f)
            
        for model_info in models:
            mid = model_info["id"]
            if mid in env_results:
                res = env_results[mid]
                
                # 动态计算 CV，兼容老版本未保存 cv 字段的 pkl 数据
                mean_r = res.get("mean_reward", 0.0)
                std_r = res.get("std_reward", 0.0)
                cv_val = res.get("cv", None)
                if cv_val is None:
                    cv_val = std_r / max(1e-3, abs(mean_r)) if mean_r != 0 else 0.0
                    
                extracted_data.append({
                    "Environment": env.split('-')[0].capitalize(), # 例如：Merge
                    "Model_ID": mid,
                    "Role": model_info["role"],
                    "Academic_Name": model_info["name"],
                    "Mean_Reward": round(mean_r, 2),
                    "Survival_Rate": round(res.get("survival_rate", 0), 1),
                    "Mean_Speed": round(res.get("mean_speed", 0), 2),
                    "CV": round(cv_val, 3)
                })
            else:
                print(f"⚠️ 在 {env} 的数据文件中未找到模型 {mid} 的记录！")

    # --- 打印对齐的表格到终端 ---
    header = f"{'环境 (Env)':<14} | {'模型名称 (Model)':<22} | {'奖励 (Reward)':<12} | {'存活率 (Surv)':<14} | {'速度 (Speed)':<14} | {'变异系数 (CV)':<12}"
    print(header)
    print("-" * 110)
    for d in extracted_data:
        row_str = f"{d['Environment']:<16} | {d['Academic_Name']:<22} | {d['Mean_Reward']:<16} | {str(d['Survival_Rate'])+'%':<18} | {str(d['Mean_Speed'])+' m/s':<18} | {d['CV']:<12}"
        print(row_str)
    print("-" * 110)

    # --- 导出到 CSV 文件 ---
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plot", "table4-1")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "Table_4_2_Baseline_Metrics.csv")
    
    # 使用 utf-8-sig 编码，防止 Excel 打开中文出现乱码
    with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["评估环境", "模型角色定位", "模型学术名称", "平均累计奖励", "存活率 (%)", "平均纵向速度 (m/s)", "变异系数 (CV)"])
        for d in extracted_data:
            writer.writerow([d["Environment"], d["Role"], d["Academic_Name"], d["Mean_Reward"], d["Survival_Rate"], d["Mean_Speed"], d["CV"]])
            
    print(f"\n✅ 表格数据已成功导出至 CSV: {csv_path}")
    print("💡 您可以直接使用 Excel 打开此 CSV 文件，复制粘贴到论文的 Word 文档中，直接套用三线表格式！")

if __name__ == "__main__":
    extract_table_data()