from .highway_wrapper import create_highway_env
from .merge_wrapper import create_merge_env

def create_environment(env_name: str, is_eval: bool = False, algo: str = "sac"):
    """
    环境工厂函数，根据名称动态创建并配置指定的环境实例。

    Args:
        env_name (str): 环境的名称，例如 "highway-v0" 或 "merge-v0"。
        is_eval (bool): 如果为 True，则创建用于评估的纯净环境（关闭辅助惩罚）。
        algo (str): 算法类型，"sac" 或 "diff"，用于环境包装器中的奖励塑形。

    Returns:
        gym.Env: 配置好的 Gymnasium 环境实例。

    Raises:
        ValueError: 如果提供了不支持的环境名称。
    """
    if env_name == "highway-v0":
        print(f"🏭 环境工厂：正在创建 Highway 环境 (is_eval={is_eval}, algo={algo})...")
        return create_highway_env(env_name, is_eval=is_eval, algo=algo)
    elif env_name == "merge-v0":
        print(f"🏭 环境工厂：正在创建 Merge 环境 (is_eval={is_eval}, algo={algo})...")
        return create_merge_env(env_name, is_eval=is_eval, algo=algo)
    else:
        raise ValueError(f"不支持的环境名称: {env_name}. 请使用 'highway-v0' 或 'merge-v0'.")

# 可以在这里添加其他环境的创建函数，并注册到工厂中