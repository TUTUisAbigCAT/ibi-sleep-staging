#!/bin/bash
#SBATCH --account=def-plago       # 换成你的账户名 (def-xxxx)
#SBATCH --cpus-per-task=4         # CPU 数量
#SBATCH --mem=60G                 # 内存
#SBATCH --time=24:00:00            # 最大运行时间 (时:分:秒)
# -------------------------------


# -------------------------------
# 运行程序
# -------------------------------

python dataset_preparation/public_datasets_1/extract_and_save_ibi_1.py