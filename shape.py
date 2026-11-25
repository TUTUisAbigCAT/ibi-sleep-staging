import numpy as np
import os
import glob
from typing import List, Dict

# --- 配置参数 ---
# WatchSleepNet IBI Epoch 的标准特征维度 D
TARGET_FEATURE_DIM = 750  
# ----------------------

# --- 请在这里配置您的实际数据路径 ---
# 统一的 SHHS + MESA IBI 数据文件夹路径
AGGREGATED_IBI_PATH = '/home/yxtao/scratch/WatchSleepProcessedDataset/SHHS_MESA_IBI'
# ------------------------------------

def load_and_print_raw_ibi_shapes(folder_path: str, feature_dim: int):
    """
    加载指定文件夹中的所有 IBI 特征文件，并打印其原始的 NumPy 数组形状 (.shape)，
    同时计算并显示潜在的 Epoch 数量 (总样本数 / 750)。
    不进行实际的修剪、重塑或处理。
    """
    total_files = 0
    
    print(f"--- 正在检查数据文件夹: {folder_path} 的 IBI 原始形状 ---")
    
    if not os.path.isdir(folder_path):
        print(f"错误: 数据文件夹未找到: {folder_path}. 无法继续。")
        return
            
    file_list = glob.glob(os.path.join(folder_path, '*.npz'))
    total_files = len(file_list)
    print(f"发现文件总数: {total_files}")
        
    # 1. 遍历文件，加载原始数据
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        try:
            npzfile = np.load(file_path)
            
            if 'data' not in npzfile:
                print(f"警告: {file_name} 中未找到 'data' 键，跳过。")
                continue
                
            # 读取原始数组，不进行任何处理
            ibi_data_raw = npzfile['data']
            fs = npzfile["fs"]
            
            total_samples = ibi_data_raw.size

            downsample_rate = 1
            
            num_samples = int((fs / downsample_rate) * 30) # for 30 second segments
            num_segments = len(ibi_data_raw) // num_samples
    
            # Reshape the ibi and labels array
            ibi_data_raw = ibi_data_raw[:num_samples*num_segments]
            ibi_data_raw = ibi_data_raw.reshape(num_segments, num_samples)
            
            # 打印原始形状和计算结果
            print(f"  文件: {file_name}")
            print(f"    - 频率 .shape: {fs}")
            print(f"    - 处理结束 .shape: {ibi_data_raw.shape}")
            print(f"    - 元素总数 (.size): {total_samples}")
            
            if total_samples % feature_dim != 0:
                 print(f"      **警告**: 无法整除! 完整 Epoch 数量为 {potential_epochs_int}")


        except Exception as e:
            print(f"处理文件 {file_name} 时发生错误: {e}")
            continue

    print("\n------------------------------------")
    print(f"共检查 {total_files} 个文件。")
    print("------------------------------------")


if __name__ == "__main__":
    load_and_print_raw_ibi_shapes(AGGREGATED_IBI_PATH, TARGET_FEATURE_DIM)