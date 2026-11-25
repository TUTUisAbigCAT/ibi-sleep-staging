import os
import numpy as np
import biosppy
import neurokit2 as nk
import logging
from multiprocessing import Pool, cpu_count
from scipy.signal import find_peaks
import pandas as pd
from scipy.signal import resample_poly

logging.basicConfig(level=logging.INFO)


def calculate_ibi_ecg(signal, fs):
    """
    Faithfully reproduces the old "biosppy" ECG approach:
      1) Run biosppy ECG pipeline.
      2) Compute IBI between consecutive R-peaks.
      3) Assign those IBIs to each sample between peaks (float64).
      4) Zero out IBIs >= 2.0 seconds.
    """
    out = biosppy.signals.ecg.ecg(signal, sampling_rate=fs, show=False)
    r_peaks = out["rpeaks"]  # indices of R-peaks
    ibi_values = np.diff(r_peaks) / fs  # IBI in seconds

    # Create an array to hold IBI (float64)
    ibi = np.zeros(signal.shape, dtype=np.float64)

    # Fill in IBI values between consecutive peaks
    for i in range(len(r_peaks) - 1):
        ibi[r_peaks[i] : r_peaks[i + 1]] = ibi_values[i]

    # Zero out improbable IBIs
    ibi[ibi >= 2.0] = 0.0
    return ibi


def calculate_ibi_ppg(signal, fs):
    """
    Faithfully reproduces the old "neurokit" PPG approach:
      1) Use neurokit2 PPG pipeline.
      2) Compute IBI between consecutive PPG peaks.
      3) Assign those IBIs to each sample between peaks (float64).
      4) Zero out IBIs >= 2.0 seconds.
    """
    signals, info = nk.ppg_process(signal, sampling_rate=fs)
    peaks = info["PPG_Peaks"]  # indices of PPG peaks
    ibi_values = np.diff(peaks) / fs

    # Create an array to hold IBI (float64)
    ibi = np.zeros(signal.shape, dtype=np.float64)

    # Fill in IBI values between consecutive peaks
    for i in range(len(peaks) - 1):
        ibi[peaks[i] : peaks[i + 1]] = ibi_values[i]

    # Zero out improbable IBIs
    ibi[ibi >= 2.0] = 0.0
    return ibi


def process_file(
    filename,
    in_dir,
    out_dir,
    info_df,
    id_col,
    ahi_col,
    fs_col,
    method,
    dataset_name
):
    """
    Loads .npz file from 'in_dir', computes continuous IBI array, 
    downsamples to 25 Hz and saves.
    """
    file_path = os.path.join(in_dir, filename)

    try:
        data = np.load(file_path)
        signal = data["data"].flatten()  # original waveform
        original_fs = int(data[fs_col].item())  # e.g., 125, 250, 256, etc.
        stages = data["stages"]

        # Compute IBI using the old code approach
        if method == "biosppy":  # ECG for SHHS
            ibi = calculate_ibi_ecg(signal, original_fs)
        elif method == "neurokit":  # PPG for MESA
            ibi = calculate_ibi_ppg(signal, original_fs)
        else:
            raise ValueError(f"Unknown method: {method}")

        if ibi is None:
            return

        # Parse subject ID from filename
        if dataset_name == "SHHS":
            file_id = filename.split("-")[1].split(".npz")[0].lstrip("0")
        else:  # MESA or other
            if filename.startswith("mesa-"):
                file_id = filename[len("mesa-") : -len(".npz")]
            else:
                file_id = filename.split(".npz")[0]

        # Retrieve AHI from info_df
        matching_rows = info_df[info_df[id_col] == file_id.lstrip("0")]
        if matching_rows.empty:
            logging.warning(f"AHI not found for {filename} in dataset {dataset_name}.")
            return
        ahi = float(matching_rows[ahi_col].values[0])

        # Smarter downsampling to 25 Hz
        target_fs = 25
        if original_fs % target_fs == 0:
            factor = int(original_fs // target_fs)
            ibi_ds = ibi[::factor]
            stages_ds = stages[::factor]
        else:
            ibi_ds = resample_poly(ibi, target_fs, original_fs)
            stages_ds = resample_poly(stages, target_fs, original_fs)

        # Construct the output filename
        if dataset_name == "MESA":
            out_filename = f"mesa-{file_id}.npz"
        else:
            out_filename = filename

        # Save the processed file
        out_path = os.path.join(out_dir, out_filename)
        np.savez(
            out_path,
            data=ibi_ds.astype(np.float64),
            fs=target_fs,
            stages=stages_ds,
            ahi=ahi
        )
        logging.info(f"{dataset_name} processed & saved => {out_filename}")

    except Exception as e:
        logging.exception(f"Error processing file: {filename} in {dataset_name}")


def process_dataset(
    dataset_name,
    in_dir,
    out_dir,
    info_path,
    id_col,
    ahi_col,
    fs_col,
    method
):
    """
    Main function with caching logic: skips files that already exist in out_dir.
    """
    try:
        logging.info(f"=== Processing {dataset_name} dataset ===")

        info_df = pd.read_csv(info_path, dtype={id_col: str})
        info_df[id_col] = info_df[id_col].str.lstrip("0")

        os.makedirs(out_dir, exist_ok=True)
        file_list = [f for f in os.listdir(in_dir) if f.endswith(".npz")]
        
        # --- CACHE LOGIC START ---
        existing_files = set(os.listdir(out_dir))
        to_process = []
        
        for f in file_list:
            # 模拟 process_file 中的文件名生成逻辑
            if dataset_name == "MESA":
                fid = f[len("mesa-") : -len(".npz")] if f.startswith("mesa-") else f.split(".npz")[0]
                out_f = f"mesa-{fid}.npz"
            else:
                out_f = f
            
            if out_f not in existing_files:
                to_process.append(f)
        
        skip_count = len(file_list) - len(to_process)
        if skip_count > 0:
            logging.info(f"{dataset_name}: Skipped {skip_count} files (already processed).")
        
        if not to_process:
            logging.info(f"{dataset_name}: No new files to process.")
            return
      # --- CACHE LOGIC END ---

        logging.info(f"{dataset_name}: Found {len(to_process)} files to process in {in_dir}")

        # 改为直接循环处理，不再使用多进程
        for f in to_process:
            try:
                # 直接调用函数，传入与原 args_list 相同的参数
                process_file(f, in_dir, out_dir, info_df, id_col, ahi_col, fs_col, method, dataset_name)
            except Exception as e:
                logging.error(f"Error processing file {f} in {dataset_name}: {e}")

        logging.info(f"=== Completed processing {dataset_name} ===\n")

    except Exception as e:
        logging.exception(f"Error processing dataset: {dataset_name}")

if __name__ == "__main__":
    logging.info("Starting dataset processing...")
    output_dir = "/home/yxtao/scratch/WatchSleepProcessedDataset/SHHS_MESA_IBI_1" 
    
    # SHHS
    shhs_input_dir = "/home/yxtao/scratch/SHHS_ECG_1/" 
    shhs_info_path = "/home/yxtao/scratch/shhs/datasets/shhs-harmonized-dataset-0.21.0.csv" 
    process_dataset(
        dataset_name="SHHS",
        in_dir=shhs_input_dir,
        out_dir=output_dir,
        info_path=shhs_info_path,
        id_col="nsrrid",
        ahi_col="nsrr_ahi_hp3r_aasm15",
        fs_col="fs",
        method="biosppy",
    )

    # MESA
    mesa_input_dir = "/home/yxtao/scratch/MESA_PPG_raw_2/" 
    mesa_info_path = "/home/yxtao/scratch/mesa/datasets/mesa-sleep-harmonized-dataset-0.7.0.csv" 
    process_dataset(
        dataset_name="MESA",
        in_dir=mesa_input_dir,
        out_dir=output_dir,
        info_path=mesa_info_path,
        id_col="mesaid",
        ahi_col="nsrr_ahi_hp3u",
        fs_col="fs",
        method="neurokit",
    )

    logging.info("All done! SHHS + MESA IBI files are now in the same folder.")