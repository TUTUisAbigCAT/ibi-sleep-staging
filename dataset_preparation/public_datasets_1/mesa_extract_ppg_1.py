import os
from pathlib import Path
from tqdm import tqdm
from utils import read_edf_data, save_to_npz

out_dir = Path("/home/yxtao/scratch/MESA_PPG_raw_2/") # TO fill in: path to save the MESA extract raw PPG data, this folder needs to be created before running this file
out_dir.mkdir(parents=True, exist_ok=True)

def process_mesa(out_dir):
    mesa_dir = "/home/yxtao/scratch/mesa/polysomnography/edfs/" # TO fill in: path to the MESA dataset EDF raw data files
    files = [f for f in os.listdir(mesa_dir) if f.endswith(".edf")]
    for file in tqdm(files):
        sid = file.split("-")[-1].split(".")[0]
        data_path = os.path.join(mesa_dir, file)
        # TO fill in: path to the MESA dataset XML annotation files
        label_path = f"/home/yxtao/scratch/mesa/polysomnography/annotations-events-profusion/{file.split('.')[0]}-profusion.xml"
        try:
            data, fs, stages = read_edf_data(data_path, label_path, dataset="MESA", select_chs=["Pleth"])
            save_to_npz(f"{str(out_dir)}/mesa-{sid}.npz", data, stages, fs)
        except Exception as e:
            print(f"Error processing {sid}: {e}")

if __name__ == "__main__":
    process_mesa(out_dir)
