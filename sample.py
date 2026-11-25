import mne.io
import os
import glob
import numpy as np

# --- 配置参数 ---
# 注意：您需要将 'your_edf_file.edf' 替换为您实际的文件路径
# 示例文件路径，请根据您的实际数据进行修改！
DATA_PATH_TEMPLATE = '/home/yxtao/scratch/shhs/polysomnography/edfs/shhs2/shhs2-203186.edf' 

# 假设您只需要提取 ECG 通道的信息
SELECT_CHANNELS = ["ECG"] 

# 如果您想排除某些通道，可以在这里指定
EXCLUDE_CHANNELS = [] 

def get_original_sampling_frequency(data_path: str, select_chs: list, exclude_chs: list = None):
    """
    读取 EDF 文件，选择指定通道，并返回原始采样率 (sfreq)。

    Args:
        data_path (str): EDF 文件的完整路径。
        select_chs (list): 要选择的通道名称列表（例如 ["ECG"]）。
        exclude_chs (list, optional): 要排除的通道名称列表。
        
    Returns:
        float: 原始采样率 (sfreq)，如果文件读取失败则返回 None。
    """
    try:
        # 1. 读取原始 EDF 数据
        # verbose=0: 抑制 MNE 输出的冗余信息
        # exclude=exclude_chs: 排除不需要的通道
        # infer_types=False: 不自动推断通道类型
        raw_data = mne.io.read_raw_edf(
            data_path, 
            verbose=0, 
            exclude=exclude_chs, 
            infer_types=False
        )

        # 2. 选择需要的通道 (例如 "ECG", "Pleth" 等)
        raw_data.pick(picks=select_chs)
        
        # 3. 提取原始采样率
        original_fs = raw_data.info["sfreq"]
        
        # 4. 打印结果
        print(f"成功读取文件: {os.path.basename(data_path)}")
        print(f"选取的通道: {raw_data.ch_names}")
        print(f"原始采样率 (sfreq): {original_fs} Hz")
        
        return original_fs
        
    except FileNotFoundError:
        print(f"错误：文件未找到于 {data_path}")
        return None
    except ValueError as e:
        print(f"错误：无法选取通道或文件结构问题。{e}")
        return None
    except Exception as e:
        print(f"处理文件时发生未知错误: {e}")
        return None


if __name__ == "__main__":
    # --- 实际运行示例 (请修改 DATA_PATH_TEMPLATE) ---
    
    # 示例: 假设您要读取 MESA 目录下的某个文件，并提取 "Pleth" (PPG) 信号的采样率
    # 注意: 您必须安装 MNE-Python 库才能运行此代码!
    # pip install mne
    
    # 请将此路径替换为您实际的 EDF 文件路径
    data_file = DATA_PATH_TEMPLATE 
    
    # 运行函数
    sampling_rate = get_original_sampling_frequency(
        data_path=data_file,
        select_chs=SELECT_CHANNELS,
        exclude_chs=EXCLUDE_CHANNELS
    )