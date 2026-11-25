import os
from pathlib import Path
from tqdm import tqdm
# 注意：您的 utils.py 中已经包含了 numpy，但如果您在主脚本中直接使用它，仍需要引入
import numpy as np 
from utils import read_edf_data, save_to_npz

out_dir = Path("/home/yxtao/scratch/SHHS_ECG_1/") # 您的输出路径
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------
# !!! 注意: N1/N2 合并逻辑已在 utils.py 中完成，此处无需再定义和执行 !!!
# ------------------------------------------------

def process_shhs(out_dir):
    # TO fill in: path to the SHHS dataset EDF raw data files
    shhs_dirs = [
        "/home/yxtao/scratch/shhs/polysomnography/edfs/shhs1/",
        "/home/yxtao/scratch/shhs/polysomnography/edfs/shhs2/"
    ]
    
    processed_count = 0 
    
    for shhs_dir in shhs_dirs:
        dir_label = os.path.basename(os.path.normpath(shhs_dir))
        files = [f for f in os.listdir(shhs_dir) if f.endswith(".edf")]
        for file in tqdm(files):
            sid = file.split("-")[1].split(".")[0]
            data_path = os.path.join(shhs_dir, file)
            label_path = f"/home/yxtao/scratch/shhs/polysomnography/annotations-events-profusion/{dir_label}/{file.split('.')[0]}-profusion.xml"
            
            try:
                # 1. 读取数据、采样率和分期标签
                # N1/N2 合并已在 read_edf_data 内部完成
                data, fs, stages = read_edf_data(data_path, label_path, dataset="SHHS", select_chs=["ECG"])
                
                # 2. 直接保存已处理的数据
                save_to_npz(f"{str(out_dir)}/shhs-{sid}.npz", data, stages, fs)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {sid}: {e}")
                
    print(f"\nProcessing finished. Total successful subjects: {processed_count}")

if __name__ == "__main__":
    process_shhs(out_dir)