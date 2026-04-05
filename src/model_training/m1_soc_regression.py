import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure src modules can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    PROCESSED_FILES,
    WEIGHTS_DIR,
    ESA_TARGET_COL,
    MODULE1_PARAMS,
    RANDOM_SEED
)

def train_module_1():
    print("="*60)
    print("🚀 Initiating Module 1: ESA SOC Regressor Training")
    print("="*60)

    # 1. Load Processed Data
    data_path = PROCESSED_FILES["esa_train"]
    if not data_path.exists():
        raise FileNotFoundError(f"Processed ESA train data not found at {data_path}. Run 07_data_processing.ipynb first.")
    
    print(f"Loading data from {data_path.name}...")
    df = pd.read_csv(data_path)
    
    # 2. Split Features & Target
    X = df.drop(columns=[ESA_TARGET_COL])
    y = df[ESA_TARGET_COL]
    
    print(f"Feature Matrix: {X.shape[0]} rows, {X.shape[1]} principal components")

    # 3. Train-Validation Split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"Training split: {len(X_train)} samples\nValidation split: {len(X_val)} samples\n")

    # 4. Initialize & Train XGBoost Model
    xgb_params = MODULE1_PARAMS.get("xgb", {})
    print("Initializing XGBoost Regressor with hyperparams:")
    for k, v in xgb_params.items():
        print(f"  - {k}: {v}")
    
    # Force evaluation metric to RMSE 
    kwargs = {**xgb_params, "n_jobs": -1}
    model = XGBRegressor(**kwargs)

    print("\nTraining model... (This may take a minute)")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )

    # 5. Evaluate Performance
    print("\n" + "-"*60)
    print("📈 Validation Performance")
    print("-" * 60)
    y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Calculate a custom "Accuracy Margin" (e.g. % of predictions within +/- 0.5 log units of actuals)
    margin = 0.5
    accuracy_within_margin = np.mean(np.abs(y_val - y_pred) <= margin) * 100
    
    print(f"RMSE (Root Mean Sq Error): {rmse:.4f}")
    print(f"MAE  (Mean Abs Error):     {mae:.4f}")
    print(f"R²   (Explained Variance): {r2:.4f}")
    print(f"Accuracy (Within ±{margin}):     {accuracy_within_margin:.2f}%\n")

    # 6. Save the Model
    model_out_path = WEIGHTS_DIR / "esa_soc_model.pkl"
    joblib.dump(model, model_out_path)
    print(f"✅ Trained model successfully saved to: {model_out_path.name}")

if __name__ == "__main__":
    train_module_1()
