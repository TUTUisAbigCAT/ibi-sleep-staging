import os
import torch
import argparse
import warnings
import random
import numpy as np
import re
import time

# 修改点 1: 确保引入了 K-fold 相关的函数
from data_setup import create_dataloaders_kfolds 
from config import (
    WatchSleepNetConfig,
    InsightSleepNetConfig,
    SleepConvNetConfig,
    SleepPPGNetConfig,
    dataset_configurations,
)
# 修改点 2: 引入 train_and_evaluate 处理 K-fold 逻辑
from engine import train_and_evaluate, setup_model_and_optimizer

# --- 基础配置 ---
warnings.filterwarnings("ignore", category=UserWarning)
start_time = time.time()
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DATASET = "dreamt_pibi" 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="watchsleepnet", choices=["watchsleepnet", "insightsleepnet", "sleepconvnet", "sleepppgnet"])
# 注意：K-fold 模式下通常不使用 --testing，因为需要跑完所有折叠并评估
args = parser.parse_args()

train_config = dataset_configurations.get(TRAIN_DATASET, None)
if train_config is None:
    raise ValueError(f"Invalid dataset configuration for {TRAIN_DATASET}.")

# 模型配置类选择
if args.model == "watchsleepnet":
    model_config_class = WatchSleepNetConfig
elif args.model == "insightsleepnet":
    model_config_class = InsightSleepNetConfig
elif args.model == "sleepconvnet":
    model_config_class = SleepConvNetConfig
elif args.model == "sleepppgnet":
    model_config_class = SleepPPGNetConfig
else:
    raise ValueError(f"Unknown model: {args.model}")

model_config_dict = model_config_class.to_dict()  
loss_fn = model_config_class.LOSS_FN

# 设定模型保存路径（作为各折叠 checkpoint 的基准路径）
model_save_path = train_config["get_model_save_path"](
    model_name=args.model, 
    dataset_name=TRAIN_DATASET, 
    version="shhs_mesa_kfold"
)

# --- K-Fold 核心逻辑修改 ---

# 修改点 3: 创建 K-Fold Dataloaders
print(f"Creating 3-fold dataloaders for {TRAIN_DATASET}...")
dataloader_folds = create_dataloaders_kfolds(
    dir=train_config["directory"],
    dataset=TRAIN_DATASET,
    num_folds=3,              # 设定为 5 折
    val_ratio=0.2,
    batch_size=model_config_class.BATCH_SIZE,
    num_workers=NUM_WORKERS,
    multiplier=train_config["multiplier"],
    downsampling_rate=train_config["downsampling_rate"],
)

# 修改点 4: 调用 train_and_evaluate 代替原本的单次 train 过程
print("Starting cross-validation training...")
best_results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc = train_and_evaluate(
    model_name=args.model,
    model_params=model_config_dict,  
    dataloader_folds=dataloader_folds,
    saved_model_path=None,           # 重要：设为 None 表示从零开始训练，不做迁移学习
    loss_fn=loss_fn,
    num_epochs=model_config_class.NUM_EPOCHS,
    patience=model_config_class.PATIENCE,
    device=DEVICE,
    checkpoint_path=model_save_path, # 系统会自动在文件名中加入 _fold1, _fold2 等
    learning_rate=model_config_class.LEARNING_RATE,
    weight_decay=model_config_class.WEIGHT_DECAY,
    freeze_layers=False,
)

# 修改点 5: 直接输出 K-Fold 汇总后的最终指标
print("\n" + "="*30 + " SHHS/MESA K-Fold Final Results " + "="*30)
print(f"Overall Acc: {overall_acc:.4f}")
print(f"Overall F1:  {overall_f1:.4f}")
print(f"Overall Kappa: {overall_kappa:.4f}")
print(f"Overall AUROC: {auroc:.4f}")
print(f"REM F1 Score: {rem_f1:.4f}")
print("="*80)

end_time = time.time()
print(f"Total elapsed time: {end_time - start_time:.2f} seconds")