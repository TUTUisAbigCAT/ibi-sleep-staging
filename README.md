# WatchSleepNet

![Static Badge](https://img.shields.io/badge/License%20-%20MIT%20-%20blue)

<img src="./media/WatchSleepNet Logo.png" height="100"/>

## Introduction

WatchSleepNet is a novel deep learning model designed to improve sleep staging using wrist-worn wearables. The model integrates ResNet, Temporal Convolutional Networks (TCN), and LSTM with attention to enhance temporal feature extraction. A key innovation is the pretraining strategy, which first trains on large ECG-derived Inter-Beat Interval (IBI) datasets and then fine-tunes on smaller wrist-worn PPG-derived IBI datasets, significantly improving generalization. WatchSleepNet achieves state-of-the-art performance on the DREAMT dataset, surpassing existing wearable sleep staging models. This repository provides the full implementation and benchmarking tools to support reproducible research and future advancements in sleep monitoring.

## Directory Structure

```bash
.
├── dataset_preparation
│   ├── dreamt
│   │   ├── dreamt_extract_ibi_se.py # Run program to extract IBI from downloaded DREAMT dataset
│   ├── public_datasets
│   │   ├── mesa_extract_ppg.py # Run program to extract PPG from MESA dataset
│   │   ├── shhs_extract_ecg.py # Run program to extract ECG from SHHS dataset
│   │   ├── extract_and_save_ibi.py # Run program to save IBI extracted from both MESA and SHHS datasets.
│   │   └── utils.py
├── modeling
│   ├── config.py # Configure parameters related to dataset and models
│   ├── data_setup.py
│   ├── engine.py
│   ├── insightsleepnet_hpt.py # Run program to perform hyperparameter tuning for InsightSleepNet
│   ├── sleepconvnet_hpt.py # Run program to perform hyperparameter tuning for SleepConvNet
│   ├── watchsleepnet_hpt.py # Run program to perform hyperparameter tuning for WatchSleepNet
│   ├── models # Model architectures
│   │   ├── insightsleepnet.py
│   │   ├── sleepconvnet.py
│   │   └── watchsleepnet.py
│   ├── train_cv.py
│   ├── train_transfer.py # Run program to perform transfer learning experiments
│   ├── watchsleepnet_cv_ablation.py # Run program to perform ablation experiments (DREAMT) on WatchSleepNet
│   └── watchsleepnet_transfer_ablation.py # Run program to perform ablation experiments (Transfer Learning) on WatchSleepNet
├── README.md
└── requirements.txt
```
> [!TIP]
> The files without description contain utility functions that are used in the main programs.

## Installation 

1. Clone the repository

```
git clone [redacted]/WatchSleepNet_public.git
```

2. Setup virtual environment (recommended)
```
conda create -n watchsleepnet python=3.10
conda activate watchsleepnet
```
> [!Note]
> Develpment was conducted in an conda environment, but you may use another package manager of your choice.

2. Install required dependencies
```
pip install -r requirements.txt
```
> [!TIP]
> The versions in `requirements.txt` have been tested with our compute setup. Please update the versions if they are not compatible with your CUDA setup.


## Usage

### Dataset Preperation

1. Download the open source datasets ([DREAMT](https://physionet.org/content/dreamt/2.0.0/), [MESA](https://sleepdata.org/datasets/mesa), [SHHS](https://sleepdata.org/datasets/shhs)) to your designated directory.

2. Extract IBI from DREAMT dataset
```
cd dataset_preparation/dreamt
python dreamt_extract_ibi_se.py
```
> [!TIP]
> Update the `root_path` variable in the file to where you stored the DREAMT dataset. The directory should be named `DREAMT_PIBI_SE`.

3. Extract PPG from MESA and ECG from SHHS
```
cd dataset_preparation/public_datasets
python mesa_extract_ppg.py
python shhs_extract_ecg.py
```
> [!TIP]
> Update the `out_dir` path in the files to where you stored the MESA and SHHS datasets.

4. Extract IBI from MESA and SHHS
```
python dataset_preperation/public_dataset/extract_and_save_ibi.py
```
> [!TIP]
> Update `shhs_input_dir` and `mesa_input_dir` to where you stored the extracted PPG and ECG datasets. Update `output_dir` to where you want to store the combined IBI. The directory should be named `SHHS_MESA_IBI`.

5. Confirm that the dataset root directory contains the following subdirectories with the same names:

```bash
.
├── DREAMT_PIBI_SE
├── MESA_PIBI
└── SHHS_MESA_IBI
```

6. Set the `DATASET_DIR` variable in `modeling/config.py` to your dataset root directory:
```
DATASET_DIR = # Enter your dataset root directory
```
Enter the modeling folder for running experiments
```
cd modeling/
```

### Experiment 1: Transfer Learning

You can perform transfer learning experiments (pre-train on IBI from SHHS+MESA and test on DREAMT IBI) using the `modeling/train_transfer.py`. Run the experiment with WatchSleepNet:
```
python train_transfer.py
```
To perform the experiment with other benchmark models (i.e. InsightSleepNet, SleepConvNet), indicate selected model using the `--model` parser argument:
```
python train_transfer.py --model=insightsleepnet
```
```
python train_transfer.py --model=sleepconvnet
```

### Experiment 2: WatchSleepNet Ablation Study

You can perform ablation experiments on WatchSleepNet using `watchsleepnet_cv_ablation.py`. Run WatchSleepNet without the TCN and Attention, for only pretraining without finetuning
```
python watchsleepnet_cv_ablation.py
```
or with finetuning
```
python watchsleepnet_transfer_ablation.py
```
> [!TIP]
> `watchsleepnet_cv_ablation.py` tests performance on using only DREAMT while `watchsleepnet_transfer_ablation.py` performs transfer learning on SHHS+MESA and then tests on DREAMT.

Utilize the argument flags to run either/both the TCN and Attention layers
```
python watchsleepnet_cv_ablation.py --use_tcn --use_attention
```
Utilize the argument flags to run without the TCN and Attention layers
```
python watchsleepnet_cv_ablation.py
```

### Hyperparameter Tuning

You can perform hyperparameter tuning for WatchSleepNet, InsightSleepNet, and SleepConvNet. For example, to tune WatchSleepNet run
```
python watchsleepnet_hpt.py
```
## DREAMT IBI 分期替换实验（本仓库新增）

本仓库在 WatchSleepNet_public_1 代码基础上增加了：

- **MambaSleepNet**（`modeling/models/mamba_sleepnet.py`）：CNN 提取每个 30 秒窗口的特征 + 双向 Mamba 建模整晚序列 + 分类层，没有对比学习部分；需要 `mamba_ssm`（CUDA）。`--model mambasleepnet`，设置见 `config.MambaSleepNetConfig`。
- **4 类**：W 0、Light（N1+N2）1、Deep 2、REM 3（所有模型）。`compute_metrics` 按模型输出的类别数计算，某个集合缺少某一类时 REM F1 和 AUROC 仍然正确。
- `experiments/`：替换实验的脚本（只读输入，输出写到新目录）。

```bash
# Trillium 上，仓库根目录（/home/yxtao/links/projects/def-plago/yxtao/ibi-sleep-staging）。环境与 WatchSleepNet_public 相同：
module load StdEnv/2023 gcc/12.3 opencv/4.9.0 python/3.11
source /home/yxtao/links/projects/def-plago/yxtao/WatchSleepNet_public/venv/bin/activate
D=/home/yxtao/links/scratch/ibi-sleep-staging
REAL=/scratch/yxtao/dataset-generation-ppg50/nights/ppg50_1393726_e146/ibi_elgendi_local_fix/night_npz_real
python experiments/summarize_sync.py                                   # E4 与 PSG 是否同步（读 PPG 提取报告）
python experiments/make_folds.py --nights $REAL --output $D/folds.json  # 按人 5 折，平衡 AHI 和深睡
python experiments/make_replacement_sets.py --folds $D/folds.json --real $REAL --generated GEN_NIGHTS \
  --output-dir $D/sets --arms real gen:Deep gen:REM gen:Deep,REM gen:random0.2 shuffled:Deep shuffled:REM shuffled:random0.2
sbatch experiments/submit_run_folds.sh mambasleepnet $D/folds.json $D/sets $D/results real 0   # 每个模型、arm、seed 一个 job
```

- `GEN_NIGHTS`：每折生成器生成的整夜 IBI（WatchSleepNet NPZ 格式），用 `{fold}` 代表折号；只有基线 `real` 时可以随便填。
- 训练只用每折的 `train` 参与者，`val` 用于 early stopping，`test` 全部是真实数据；结果在 `$D/results/<model>/summary.csv`（各组与 `real` 的差）。
- 类别加权：`submit_run_folds.sh ... real 0 --class-weights sqrt`（或 `inverse`）。每折按自己训练集的标签算交叉熵的类别权重（∝ 占比^-1/2 或 ^-1，训练 epoch 上的平均权重为 1，验证 loss 用同样的权重），结果写到 `$D/results/<model>_cw-sqrt/`，和不加权的结果分开；每折的权重和各类 epoch 数记在 `results.json` 的 `settings.class_weights`。
- `python modeling/train_cv.py --model mambasleepnet --train_dataset dreamt_pibi` 等原有入口也可以选 MambaSleepNet。
- `experiments/submit_run_folds.sh` 在 job 里加载同样的 module 和 WatchSleepNet_public 的 venv（路径在脚本的 `VENV`）。
- 测试（CPU，用一个替身代替 CUDA 的 Mamba 内核）：`python -m pytest -q tests`
