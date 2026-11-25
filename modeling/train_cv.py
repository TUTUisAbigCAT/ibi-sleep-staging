import os
import torch
import argparse
import warnings
import random
import numpy as np
import pandas as pd #add results

from data_setup import create_dataloaders_kfolds
from models.watchsleepnet import WatchSleepNet
from models.insightsleepnet import InsightSleepNet
from models.sleepconvnet import SleepConvNet
from config import (
    WatchSleepNetConfig,
    InsightSleepNetConfig,
    SleepConvNetConfig,
    SleepPPGNetConfig,
    MambaSleepNetConfig,
    dataset_configurations,
)
from engine import train_cross_validate 

def set_seed(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Sleep Classification Models")
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="dreamt_pibi",
        choices=[
            "shhs_ibi",
            "mesa_ppg",
            "mesa_eibi",
            "mesa_pibi",
            "dreamt_pibi",
            "dreamt_ppg",
            "shhs_mesa_ibi",
        ],
        help="Dataset to train on."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sleep_staging",
        choices=["sleep_staging"],
        help="Task to perform."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="watchsleepnet",
        choices=["watchsleepnet", "insightsleepnet", "sleepconvnet", "sleepppgnet", "mambasleepnet"],
        help="Model architecture to use."
    )
    return parser.parse_args()


def main():
    # Suppress UserWarnings
    warnings.filterwarnings("ignore", category=UserWarning)
    set_seed(seed=0)
    device = get_device()
    args = parse_arguments()
    
    # Retrieve configuration based on the dataset argument
    train_config = dataset_configurations.get(args.train_dataset)
    if train_config is None:
        raise ValueError(f"Configuration for dataset '{args.train_dataset}' not found.")
    
    # Model initialization based on user argument
    model_config_class = None
    if args.model == "watchsleepnet":
        model_config_class = WatchSleepNetConfig
    elif args.model == "insightsleepnet":
        model_config_class = InsightSleepNetConfig
    elif args.model == "sleepconvnet":
        model_config_class = SleepConvNetConfig
    elif args.model == "sleepppgnet":
        model_config_class = SleepPPGNetConfig
    elif args.model == "mambasleepnet":
        model_config_class = MambaSleepNetConfig
    else:
        raise ValueError("Model not recognized.")
    
    # Convert model config to dict
    model_config = model_config_class.to_dict()
    
    # Generate dynamic save path for model and analysis
    model_save_path = train_config["get_model_save_path"](
        model_name=args.model,
        dataset_name=args.train_dataset,
        version="vtrial"
    )
    
    print(f"Model Save Path: {model_save_path}")
    
    # Create dataloaders with cross-validation
    print(f"Performing cross-validation on dataset: {args.train_dataset}")
    dataloader_folds = create_dataloaders_kfolds(
        dir=train_config["directory"],
        dataset=args.train_dataset,
        num_folds=5,
        val_ratio=0.2,
        batch_size=model_config.get("BATCH_SIZE", 16),
        num_workers=os.cpu_count() // 2,
        multiplier=train_config.get("multiplier", 1),
        downsampling_rate=train_config.get("downsampling_rate", 1),
    )
    
    print(f"Model save path: {model_save_path}")
    
    loss_fn = model_config.get('LOSS_FN')

    # 修改了checkpoints_
    saved_model_path='/home/yxtao/project/WatchSleepNet_public/checkpoints/watchsleepnet/shhs_mesa_ibi/best_saved_model_ablation_separate_pretraining.pt'

    # Perform cross-validation without transfer learning
    results, overall_acc, overall_f1, overall_kappa, rem_f1, auroc = train_cross_validate(
        model_name=args.model,
        model_params=model_config,  # Contains 'use_attention', 'use_tcn', 'num_classes', etc.
        dataloader_folds=dataloader_folds,
        # TODO: change the NONE to the best model path
        saved_model_path=saved_model_path,      # If training from scratch
        learning_rate=model_config.get('LEARNING_RATE', 1e-3),
        weight_decay=model_config.get('WEIGHT_DECAY', 1e-4),
        loss_fn=loss_fn,
        num_epochs=model_config.get('NUM_EPOCHS', 100),
        patience=model_config.get('PATIENCE', 10),
        device=device,
        checkpoint_path=train_config["get_model_save_path"](
            model_name=args.model,
            dataset_name=args.train_dataset, 
            version="cv_no_transfer"
        ),
        freeze_layers=False,
    )
    
    # Optionally, print or log the results
    print("Cross-Validation Results:")
    print(f"Overall Accuracy: {overall_acc}")
    print(f"Overall F1 Score: {overall_f1}")
    print(f"Overall Kappa: {overall_kappa}")
    print(f"REM F1 Score: {rem_f1}")
    print(f"AUROC: {auroc}")
   
    
    # === Save results to CSV ===
    os.makedirs(model_save_path, exist_ok=True)
    results_dict = {
        "Accuracy": overall_acc,
        "F1": overall_f1,
        "Kappa": overall_kappa,
        "REM_F1": rem_f1,
        "AUROC": auroc
    }
    df = pd.DataFrame([results_dict])
    csv_path = os.path.join(model_save_path, "cv_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Results saved to {csv_path}")


if __name__ == "__main__":
    main()