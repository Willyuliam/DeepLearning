import os
import pickle
import hashlib
import time

def get_cache_path(data_path, method):
    """
    生成缓存文件路径。
    参数:
        data_path (str): 用于生成缓存键的数据文件路径（或其组合）。
        method (str): 数据处理方法（用于区分不同处理方式的缓存）。
    返回:
        str: 缓存文件的完整路径。
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True) # 确保缓存目录存在

    # 基于数据文件路径和处理方法生成唯一的缓存文件名
    cache_key = f"{data_path}_{method}"
    hash_key = hashlib.md5(cache_key.encode()).hexdigest()[:8] # 取MD5哈希的前8位
    return os.path.join(cache_dir, f"datasets_{hash_key}.pkl")


def save_cache(data, cache_path):
    """
    保存数据到缓存文件。
    参数:
        data (any): 要缓存的数据对象。
        cache_path (str): 缓存文件的路径。
    """
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f) # 使用pickle序列化数据
        print(f"数据已缓存到: {cache_path}")
    except Exception as e:
        print(f"缓存保存失败: {e}")


def load_cache(cache_path):
    """
    从缓存文件加载数据。
    参数:
        cache_path (str): 缓存文件的路径。
    返回:
        any: 加载的数据对象，如果文件不存在或加载失败则返回None。
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                print(f"从缓存加载数据: {cache_path}")
                return pickle.load(f) # 使用pickle反序列化数据
        except Exception as e:
            print(f"缓存加载失败: {e}")
    return None


def check_cache_freshness(cache_path, data_files, max_age_seconds):
    """
    检查缓存文件是否新鲜（未过期且源文件未更改）。
    参数:
        cache_path (str): 缓存文件的路径。
        data_files (list): 缓存所依赖的原始数据文件路径列表。
        max_age_seconds (int): 缓存的最大有效期（秒）。
    返回:
        bool: 如果缓存新鲜则返回True，否则返回False。
    """
    if not os.path.exists(cache_path):
        return False # 缓存文件不存在

    cache_mtime = os.path.getmtime(cache_path) # 缓存文件的最后修改时间

    # 检查缓存是否过期
    if time.time() - cache_mtime > max_age_seconds:
        print(f"缓存 {cache_path} 已过期。")
        return False

    # 检查源数据文件是否有更新
    for data_file in data_files:
        if not os.path.exists(data_file):
            print(f"源数据文件 {data_file} 不存在。")
            return False # 源文件不存在
        if os.path.getmtime(data_file) > cache_mtime:
            print(f"源数据文件 {data_file} 已更新，缓存 {cache_path} 失效。")
            return False # 源文件比缓存新

    return True # 缓存新鲜