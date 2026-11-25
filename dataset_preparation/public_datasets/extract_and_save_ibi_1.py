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
    out = biosppy.signals.ecg.ecg(signal, sampling_rate=fs, show=False)
    r_peaks = out["rpeaks"]
    ibi_values = np.diff(r_peaks) / fs
    ibi = np.zeros(signal.shape, dtype=np.float64)
    for i in range(len(r_peaks) - 1):
        ibi[r_peaks[i] : r_peaks[i + 1]] = ibi_values[i]
    ibi[ibi >= 2.0] = 0.0
    return ibi


def calculate_ibi_ppg(signal, fs):
    signals, info = nk.ppg_process(signal, sampling_rate=fs)
    peaks = info["PPG_Peaks"]
    ibi_values = np.diff(peaks) / fs
    ibi = np.zeros(signal.shape, dtype=np.float64)
    for i in range(len(peaks) - 1):
        ibi[peaks[i] : peaks[i + 1]] = ibi_values[i]
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
    # ── 先确定输出文件名，存在则跳过 ──────────────────────────────────────
    if dataset_name == "MESA":
        if filename.startswith("mesa-"):
            file_id_for_name = filename[len("mesa-") : -len(".npz")]
        else:
            file_id_for_name = filename.split(".npz")[0]
        out_filename = f"mesa-{file_id_for_name}.npz"
    else:
        out_filename = filename

    out_path = os.path.join(out_dir, out_filename)
    if os.path.exists(out_path):
        logging.info(f"Skipping already processed file: {filename}")
        return
    # ─────────────────────────────────────────────────────────────────────

    file_path = os.path.join(in_dir, filename)

    try:
        data = np.load(file_path)
        signal = data["data"].flatten()
        original_fs = int(data[fs_col].item())
        stages = data["stages"]

        if method == "biosppy":
            ibi = calculate_ibi_ecg(signal, original_fs)
        elif method == "neurokit":
            ibi = calculate_ibi_ppg(signal, original_fs)
        else:
            raise ValueError(f"Unknown method: {method}")

        if ibi is None:
            return

        if dataset_name == "SHHS":
            file_id = filename.split("-")[1].split(".npz")[0].lstrip("0")
        else:
            if filename.startswith("mesa-"):
                file_id = filename[len("mesa-") : -len(".npz")]
            else:
                file_id = filename.split(".npz")[0]

        matching_rows = info_df[info_df[id_col] == file_id.lstrip("0")]
        if matching_rows.empty:
            logging.warning(f"AHI not found for {filename} in dataset {dataset_name}.")
            return
        ahi = float(matching_rows[ahi_col].values[0])

        target_fs = 25
        factor = int(original_fs // target_fs)
        ibi_ds = ibi[::factor]
        stages_ds = stages[::factor]

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
    try:
        logging.info(f"=== Processing {dataset_name} dataset ===")
        info_df = pd.read_csv(info_path, dtype={id_col: str})
        info_df[id_col] = info_df[id_col].str.lstrip("0")
        os.makedirs(out_dir, exist_ok=True)
        file_list = [f for f in os.listdir(in_dir) if f.endswith(".npz")]
        logging.info(f"{dataset_name}: Found {len(file_list)} .npz files in {in_dir}")
        num_processes = max(1, cpu_count() // 2)
        logging.info(f"{dataset_name}: Using {num_processes} processes")
        args_list = [
            (f, in_dir, out_dir, info_df, id_col, ahi_col, fs_col, method, dataset_name)
            for f in file_list
        ]
        with Pool(num_processes) as pool:
            pool.starmap(process_file, args_list)
        logging.info(f"=== Completed processing {dataset_name} ===\n")
    except Exception as e:
        logging.exception(f"Error processing dataset: {dataset_name}")


if __name__ == "__main__":
    logging.info("Starting dataset processing...")

    output_dir = "/home/yxtao/scratch/WatchSleepProcessedDataset/MESA_PIBI_20260502_naive"

    shhs_input_dir = "/home/yxtao/scratch/SHHS_ECG_1/"
    shhs_info_path = "/home/yxtao/scratch/shhs/datasets/shhs-harmonized-dataset-0.21.0.csv"
    shhs_id_col = "nsrrid"
    shhs_ahi_col = "nsrr_ahi_hp3r_aasm15"
    shhs_fs_col = "fs"
    shhs_method = "biosppy"

    mesa_input_dir = "/home/yxtao/scratch/MESA_PPG_raw/"
    mesa_info_path = "/home/yxtao/scratch/mesa/datasets/mesa-sleep-harmonized-dataset-0.7.0.csv"
    mesa_id_col = "mesaid"
    mesa_ahi_col = "nsrr_ahi_hp3u"
    mesa_fs_col = "fs"
    mesa_method = "neurokit"

    process_dataset(
        dataset_name="MESA",
        in_dir=mesa_input_dir,
        out_dir=output_dir,
        info_path=mesa_info_path,
        id_col=mesa_id_col,
        ahi_col=mesa_ahi_col,
        fs_col=mesa_fs_col,
        method=mesa_method,
    )

    logging.info("All done! SHHS + MESA IBI files are now in the same folder.")