import os
import shutil
import re
from datetime import datetime

def migrate():
    # 锁定绝对路径
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OLD_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    NEW_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, f"outputs_migrated_{timestamp}")

    # 🚨 约束：本次只处理 merge-v0 环境的数据
    OLD_MERGE_DIR = os.path.join(OLD_OUTPUTS_DIR, "merge-v0")
    NEW_MERGE_DIR = os.path.join(NEW_OUTPUTS_DIR, "merge-v0")

    if not os.path.exists(OLD_MERGE_DIR):
        print(f"❌ 找不到 merge-v0 文件夹: {OLD_MERGE_DIR}")
        return
        
    print(f"🚀 开始执行 outputs 文件夹结构迁移 (专精 Merge 环境)...")
    print(f"📁 源目录: {OLD_MERGE_DIR}")
    print(f"📁 新目录: {NEW_MERGE_DIR}\n")
    
    # 预定义 DiffSAC 第一期实验代号与全称的映射关系
    DM_NAMES = {
        "DM1": "DM1_Zero_Q",
        "DM2": "DM2_Micro_Q",
        "DM3": "DM3_Gentle_Q",
        "DM4": "DM4_Standard_Q"
    }
    
    # 1. 扫描您手动整理的 models 文件夹，建立精准的关联映射
    dm_mapping = {}  # 格式: { '旧文件夹名': ('新父文件夹名', '子目录名') }
    models_dir = os.path.join(OLD_MERGE_DIR, 'models')
    
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            # 寻找您手动创建的 diffSAC_DMx 父文件夹 (忽略大小写)
            if re.match(r'(?i)diffsac_dm[1-4]', item):
                dm_path = os.path.join(models_dir, item)
                if os.path.isdir(dm_path):
                    bc_item, online_item = None, None
                    ts_online = "00000000_000000" # 兜底时间戳
                    
                    # 识别内部的 bc 和 online 文件夹
                    for sub in os.listdir(dm_path):
                        if sub.startswith('diffusion_bc_'):
                            bc_item = sub
                        elif sub.startswith('DiffSAC_'):
                            online_item = sub
                            # 提取在线阶段的时间戳作为整个实验的唯一时间锚点
                            ts_online = sub.replace('DiffSAC_', '')
                            
                    # 提取 DM 编号并获取带有实验属性的全名
                    dm_id = re.search(r'(?i)dm[1-4]', item).group(0).upper()
                    dm_name = DM_NAMES.get(dm_id, dm_id)
                    
                    # 构建标准化的聚合父文件夹名: DiffSAC_DM1_Zero_Q_20260422_020015
                    new_parent = f"DiffSAC_{dm_name}_{ts_online}"
                    
                    # 存入字典备用
                    if bc_item: dm_mapping[bc_item] = (new_parent, 'bc_pretrain')
                    if online_item: dm_mapping[online_item] = (new_parent, 'online_finetune')

    migrated_count = 0
    categories = ['logs', 'models', 'eval_results', 'videos']
    
    # 2. 遍历四大分类，开始外科手术式分拣
    for cat in categories:
        cat_dir = os.path.join(OLD_MERGE_DIR, cat)
        if not os.path.exists(cat_dir): continue
        
        for item in os.listdir(cat_dir):
            item_path = os.path.join(cat_dir, item)
            
            # [特判 A]: 针对 models 目录下您手动建好的 diffSAC_DMx 文件夹
            if cat == 'models' and re.match(r'(?i)diffsac_dm[1-4]', item):
                for sub in os.listdir(item_path):
                    sub_path = os.path.join(item_path, sub)
                    if not os.path.isdir(sub_path): continue
                    
                    if sub in dm_mapping:
                        new_parent, sub_dir = dm_mapping[sub]
                        dest = os.path.join(NEW_MERGE_DIR, cat, new_parent, sub_dir)
                        
                        print(f"📦 复制: [models] {item}/{sub}")
                        print(f"   └──> {cat}/{new_parent}/{sub_dir}")
                        
                        shutil.copytree(sub_path, dest, dirs_exist_ok=True)
                        migrated_count += 1
                continue
            
            if not os.path.isdir(item_path): continue
            
            # [特判 B]: 对于 logs, eval_results, videos 中的散落 DiffSAC 文件夹
            if item in dm_mapping:
                new_parent, sub_dir = dm_mapping[item]
                
                # 🚨 核心逻辑: 仅有 models 才保留子目录，其余分类直接合并在聚合名称下
                if cat != 'models':
                    sub_dir = "" 
                    
                dest = os.path.join(NEW_MERGE_DIR, cat, new_parent, sub_dir) if sub_dir else os.path.join(NEW_MERGE_DIR, cat, new_parent)
                
                print(f"📦 复制: [{cat}] {item}")
                print(f"   └──> {cat}/{new_parent}" + (f"/{sub_dir}" if sub_dir else ""))
                
                shutil.copytree(item_path, dest, dirs_exist_ok=True)
                migrated_count += 1
                
            # [特判 C]: 对于常规的 SAC 实验
            else:
                clean_name = item
                
                # 剥离潜在的环境前缀
                clean_name = re.sub(r'^(?i)merge-v0_', '', clean_name)
                clean_name = re.sub(r'^(?i)merge_', '', clean_name)
                
                # 🚨 核心逻辑: 剥离结尾重复的 _SAC_时间戳 
                clean_name = re.sub(r'_SAC_\d{8}_\d{6}$', '', clean_name)
                
                dest = os.path.join(NEW_MERGE_DIR, cat, clean_name)
                
                print(f"📦 复制: [{cat}] {item}")
                print(f"   └──> {cat}/{clean_name}")
                
                shutil.copytree(item_path, dest, dirs_exist_ok=True)
                migrated_count += 1
                
    print(f"\n✅ 迁移完美结束！共重新整理了 {migrated_count} 个实验记录。")
    print(f"⚠️ 请前往查看 {NEW_OUTPUTS_DIR} 确认无误后，将其覆盖回旧的 outputs 即可。")

if __name__ == "__main__":
    migrate()
