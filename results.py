import pandas as pd
import os

# os.makedirs(model_save_path, exist_ok=True)
results_dict = {
    "Accuracy": [0,1,2,3],
    "F1": [0,1,2,3],
    "Kappa": [0,1,2,3],
    "REM_F1": [0,1,2,3],
    "AUROC": [0,1,2,3]
}

df = pd.DataFrame([results_dict])
csv_path = os.path.join("cv_results.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Results saved to {csv_path}")
