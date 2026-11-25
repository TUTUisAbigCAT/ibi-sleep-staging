import os
import numpy as np
import logging
import pandas as pd
from scipy.signal import resample_poly, butter, filtfilt

logging.basicConfig(level=logging.INFO)

def preprocess_ecg_waveform(signal, fs):
    """
    针对 Mamba 模型处理 ECG 原生波形:
    1) 带通滤波 (0.5Hz - 45Hz) 去除基线漂移和高频噪声
    2) Z-Score 归一化 (使数值分布平稳，防止 Mamba 梯度爆炸)
    """
    # 1. 滤波处理
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 45.0 / nyq
    b, a = butter(4, [low, high], btype='band')
    # 使用 filtfilt 进行零相位滤波，防止信号相位偏移
    filtered_signal = filtfilt(b, a, signal)

    # 2. Z-Score 归一化 (Critical for Mamba)
    mean_val = np.mean(filtered_signal)
    std_val = np.std(filtered_signal)
    normalized_signal = (filtered_signal - mean_val) / (std_val + 1e-8)
    
    return normalized_signal

def process_file(
    filename, in_dir, out_dir, info_df, id_col, ahi_col, fs_col, dataset_name
):
    file_path = os.path.join(in_dir, filename)

    try:
        data = np.load(file_path)
        signal = data["data"].flatten()  # 原始 ECG 波形
        original_fs = int(data[fs_col].item()) 
        stages = data["stages"]

        # --- 核心修改：由计算 IBI 改为 预处理波形 ---
        processed_signal = preprocess_ecg_waveform(signal, original_fs)

        # Parse subject ID
        if dataset_name == "SHHS":
            file_id = filename.split("-")[1].split(".npz")[0].lstrip("0")
        else:
            file_id = filename[len("mesa-") : -len(".npz")] if filename.startswith("mesa-") else filename.split(".npz")[0]

        # Retrieve AHI
        matching_rows = info_df[info_df[id_col] == file_id.lstrip("0")]
        if matching_rows.empty:
            logging.warning(f"AHI not found for {filename}.")
            return
        ahi = float(matching_rows[ahi_col].values[0])

        # 下采样到适合 Mamba 处理的频率
        # 注意：ECG 波形建议保留较高频率，25Hz 对 IBI 够用，但对 ECG 波形稍低
        # 如果模型支持，建议 target_fs 设为 100 或 125。这里暂按原逻辑设为 25
        target_fs = 25 
        if original_fs % target_fs == 0:
            factor = int(original_fs // target_fs)
            data_ds = processed_signal[::factor]
            stages_ds = stages[::factor]
        else:
            data_ds = resample_poly(processed_signal, target_fs, original_fs)
            stages_ds = resample_poly(stages, target_fs, original_fs)

        out_path = os.path.join(out_dir, filename)
        np.savez(
            out_path,
            data=data_ds.astype(np.float32), # 使用 float32 节省显存
            fs=target_fs,
            stages=stages_ds,
            ahi=ahi
        )
        logging.info(f"{dataset_name} Waveform processed => {filename}")

    except Exception as e:
        logging.exception(f"Error processing file: {filename}")

def process_dataset(dataset_name, in_dir, out_dir, info_path, id_col, ahi_col, fs_col):
    # 此处逻辑基本保持不变，删除了 method 参数
    logging.info(f"=== Processing {dataset_name} ECG Waveform ===")
    info_df = pd.read_csv(info_path, dtype={id_col: str})
    info_df[id_col] = info_df[id_col].str.lstrip("0")

    os.makedirs(out_dir, exist_ok=True)
    to_process = [f for f in os.listdir(in_dir) if f.endswith(".npz")]
    
    # 简单的跳过逻辑
    existing_files = set(os.listdir(out_dir))
    to_process = [f for f in to_process if f not in existing_files]

    for f in to_process:
        process_file(f, in_dir, out_dir, info_df, id_col, ahi_col, fs_col, dataset_name)

if __name__ == "__main__":
    # 修改输出路径，明确这是波形数据集
    output_dir = "/home/yxtao/scratch/WatchSleepProcessedDataset/SHHS_MESA_ECG_Waveform" 
    
    # SHHS
    process_dataset(
        dataset_name="SHHS",
        in_dir="/home/yxtao/scratch/SHHS_ECG_1/",
        out_dir=output_dir,
        info_path="/home/yxtao/scratch/shhs/datasets/shhs-harmonized-dataset-0.21.0.csv",
        id_col="nsrrid",
        ahi_col="nsrr_ahi_hp3r_aasm15",
        fs_col="fs"
    )

    # # MESA (如果是 PPG 波形，预处理逻辑是一样的)
    # process_dataset(
    #     dataset_name="MESA",
    #     in_dir="/home/yxtao/scratch/MESA_PPG_raw_2/",
    #     out_dir=output_dir,
    #     info_path="/home/yxtao/scratch/mesa/datasets/mesa-sleep-harmonized-dataset-0.7.0.csv",
    #     id_col="mesaid",
    #     ahi_col="nsrr_ahi_hp3u",
    #     fs_col="fs"
    # )