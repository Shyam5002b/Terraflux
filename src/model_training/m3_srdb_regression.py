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
    PREPROCESS_DIR,
    WEIGHTS_DIR,
    RANDOM_SEED
)

def train_srdb_model():
    print("="*60)
    print("🚀 Initiating Module 3: SRDB Temporal Soil Respiration Regressor")
    print("="*60)

    # 1. Load Processed Data
    data_path = PROCESSED_FILES["srdb_temporal"]
    if not data_path.exists():
        raise FileNotFoundError(f"Processed SRDB data not found at {data_path}")

    print(f"Loading data from {data_path.name}...")
    df = pd.read_csv(data_path)
    
    # Drop out incomplete targets
    df = df.dropna(subset=['Rs_annual'])
    print(f"Data shape after dropping missing Rs_annual: {df.shape}")

    # 2. Prevent Data Leakage & Setup Target
    # We must eliminate seasonal outputs and ratios which are essentially components of the target
    leakage_columns = [
        'Record_number', 'Rs_annual', 'Rs_spring', 'Rs_summer', 'Rs_autumn',
        'Rs_winter', 'Rs_growingseason', 'spring_ratio', 'summer_ratio',
        'autumn_ratio', 'winter_ratio', 'Decade'
    ]
    
    # Features
    X_raw = df.drop(columns=[col for col in leakage_columns if col in df.columns])
    
    # Apply Log Transformation on Target (As noted in our EDA phase)
    y = np.log1p(df['Rs_annual'])
    print("Applied natural log transform: log1p(Rs_annual)")

    # 3. One-hot Encode Categorical Text
    X = pd.get_dummies(X_raw, columns=['Biome', 'Ecosystem_type'], drop_first=True)
    
    # We also have to handle any accidental NA inputs generated from bad splits
    X = X.fillna(X.median())

    print(f"Feature Matrix: {X.shape[0]} rows, {X.shape[1]} columns")

    # 4. Train-Validation Split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # 5. Initialize & Train XGBoost Model
    print("\nInitializing Baseline XGBoost Regressor...")
    model = XGBRegressor(
        n_estimators=300, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=RANDOM_SEED, 
        n_jobs=-1
    )

    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    # 6. Evaluate Performance
    print("\n" + "-"*60)
    print("📈 Validation Performance (Log Scale)")
    print("-" * 60)
    y_pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    margin = 0.5
    accuracy_within_margin = np.mean(np.abs(y_val - y_pred) <= margin) * 100

    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"Accuracy (Within ±{margin}): {accuracy_within_margin:.2f}%")

    print("\n" + "-"*60)
    print("📈 Validation Performance (Real Scale - Carbon Flux)")
    print("-" * 60)
    y_val_real = np.expm1(y_val)
    y_pred_real = np.expm1(y_pred)
    rmse_real = np.sqrt(mean_squared_error(y_val_real, y_pred_real))
    mae_real = mean_absolute_error(y_val_real, y_pred_real)
    
    print(f"Original RMSE : {rmse_real:.2f} gC/m2/yr")
    print(f"Original MAE  : {mae_real:.2f} gC/m2/yr")

    # 7. Save the Model and Featured Columns
    model_out_path = WEIGHTS_DIR / "m3_srdb_regression_model.pkl"
    features_out_path = PREPROCESS_DIR / "m3_srdb_features.pkl"
    joblib.dump(model, model_out_path)
    joblib.dump(X.columns.tolist(), features_out_path)
    
    print(f"\n✅ Trained model saved to: {model_out_path.name}")
    print(f"✅ Feature columns saved to: {features_out_path.name}")

if __name__ == "__main__":
    train_srdb_model()