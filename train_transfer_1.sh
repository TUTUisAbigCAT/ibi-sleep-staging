#!/bin/bash
#SBATCH --account=def-plago       # 换成你的账户名 (def-xxxx)
#SBATCH --cpus-per-task=4         # CPU 数量
#SBATCH --mem=60G                 # 内存
#SBATCH --gpus=h100:1             # GPU 数量 (去掉这一行表示不用 GPU)
#SBATCH --time=24:00:00            # 最大运行时间 (时:分:秒)
# -------------------------------

# -------------------------------
# 运行程序
# -------------------------------
module load python/3.11
module load rust/1.76.0
module load opencv/4.12.0
module load cuda/12.2

pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.
org/whl/cu121

pip install -r requirements.txt

python modeling/train_transfer_1.py 
