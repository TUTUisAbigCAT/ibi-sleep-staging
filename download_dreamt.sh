#!/bin/bash
#SBATCH --account=def-plago       # 换成你的账户名 (def-xxxx)
#SBATCH --cpus-per-task=2         # CPU 数量
#SBATCH --mem=60G                 # 内存
#SBATCH --time=24:00:00            # 最大运行时间 (时:分:秒)
# -------------------------------

# -------------------------------
# 运行程序
# -------------------------------
cd ~/scratch/confirmed_datasets

wget -r -N -c -np --user="${PHYSIONET_USER:?set PHYSIONET_USER}" --password="${PHYSIONET_PASSWORD:?set PHYSIONET_PASSWORD}" https://physionet.org/files/dreamt/2.1.0/
