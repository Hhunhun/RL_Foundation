import os
import shutil
from datetime import datetime

def get_env_and_clean_name(folder_name):
    """
    根据旧文件夹名称推断所属的环境，并清理冗余的环境前缀。
    """
    name_lower = folder_name.lower()
    
    if 'merge' in name_lower:
        env_name = 'merge-v0'
    else:
        # 因为项目前期未做区分的数据都是 highway 的，所以默认归类为 highway-v0
        env_name = 'highway-v0'
        
    # 清理名称：如果旧文件夹名以环境名开头，则去除该前缀，让新结构更清爽
    # 例如: highway-v0_SAC_20260330 -> SAC_20260330
    clean_name = folder_name
    prefixes_to_strip = ['highway-v0_', 'highway_', 'merge-v0_', 'merge_']
    for prefix in prefixes_to_strip:
        if clean_name.lower().startswith(prefix):
            # 切片去除前缀，保留原名称的大小写
            clean_name = clean_name[len(prefix):]
            break
            
    return env_name, clean_name

def migrate():
    # 锁定路径
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OLD_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    NEW_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, f"outputs_migrated_{timestamp}")

    if not os.path.exists(OLD_OUTPUTS_DIR):
        print(f"❌ 找不到旧的 outputs 文件夹: {OLD_OUTPUTS_DIR}")
        return
        
    print(f"🚀 开始执行 outputs 文件夹结构迁移...")
    print(f"📁 源目录: {OLD_OUTPUTS_DIR}")
    print(f"📁 新目录: {NEW_OUTPUTS_DIR}\n")
    
    # 需要遍历的旧分类
    categories = ['logs', 'models', 'eval_results', 'videos']
    migrated_count = 0
    
    for category in categories:
        old_cat_dir = os.path.join(OLD_OUTPUTS_DIR, category)
        if not os.path.exists(old_cat_dir):
            continue
            
        for item in os.listdir(old_cat_dir):
            old_item_path = os.path.join(old_cat_dir, item)
            
            # 只迁移目录（跳过可能存在的零散游离文件）
            if not os.path.isdir(old_item_path):
                continue
                
            env_name, new_item_name = get_env_and_clean_name(item)
            
            # 构建新结构下的目标路径: outputs_migrated/highway-v0/models/SAC_2026...
            new_cat_dir = os.path.join(NEW_OUTPUTS_DIR, env_name, category)
            os.makedirs(new_cat_dir, exist_ok=True)
            
            new_item_path = os.path.join(new_cat_dir, new_item_name)
            
            print(f"📦 复制: [{category}] {item}")
            print(f"   └──> {env_name}/{category}/{new_item_name}")
            shutil.copytree(old_item_path, new_item_path, dirs_exist_ok=True)
            migrated_count += 1
            
    print(f"\n✅ 迁移完美结束！共重新整理了 {migrated_count} 个实验记录。")
    print(f"⚠️ 请检查 {NEW_OUTPUTS_DIR} 确认无误后，您可以手动删除旧的 outputs 文件夹，并将新文件夹重命名为 outputs。")

if __name__ == "__main__":
    migrate()