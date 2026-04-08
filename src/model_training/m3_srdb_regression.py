import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.base import clone
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure src modules can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    CV_FOLDS,
    HYPERPARAM_SEARCH_ITERS,
    MAX_REGRESSION_RMSE_LOSS,
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

    # 4. Hyperparameter Search with CV
    print("\nRunning hyperparameter search with cross-validation...")
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=2,
        tree_method="hist",
    )

    param_dist = {
        "n_estimators": [200, 300, 500, 700],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.3],
        "reg_alpha": [0.0, 0.01, 0.1],
        "reg_lambda": [1.0, 1.5, 2.0],
    }

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=HYPERPARAM_SEARCH_ITERS,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=2,
        random_state=RANDOM_SEED,
        verbose=1,
    )
    search.fit(X, y)
    best_model = search.best_estimator_
    baseline_rmse = float(-search.best_score_)
    print(f"Best CV RMSE (log scale): {baseline_rmse:.4f}")

    # 5. Feature Reduction Sweep
    importances = np.asarray(getattr(best_model, "feature_importances_", np.zeros(X.shape[1])))
    ordered_idx = np.argsort(importances)[::-1]
    ordered_features = X.columns[ordered_idx].tolist()

    candidate_counts = sorted({
        5,
        8,
        12,
        16,
        24,
        min(32, X.shape[1]),
        X.shape[1],
    })

    reduction_curve: list[dict[str, float]] = []
    for k in candidate_counts:
        selected = ordered_features[:k]
        candidate_model = clone(best_model)
        scores = cross_val_score(
            candidate_model,
            X[selected],
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=2,
        )
        mean_rmse = float(-scores.mean())
        reduction_curve.append({"n_features": float(k), "cv_rmse_log": mean_rmse})

    selected_features = ordered_features
    rmse_limit = baseline_rmse * (1 + MAX_REGRESSION_RMSE_LOSS)
    for row in reduction_curve:
        if row["cv_rmse_log"] <= rmse_limit:
            selected_features = ordered_features[: int(row["n_features"])]
            break

    print(f"Selected {len(selected_features)} / {X.shape[1]} features")

    # 6. Final Train + Holdout Evaluation
    X_selected = X[selected_features]
    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = clone(best_model)
    print("Training final model on selected features...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    # 7. Evaluate Performance
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

    # 8. Save the Model and Feature Columns
    model_out_path = WEIGHTS_DIR / "m3_srdb_regression_model.pkl"
    features_out_path = PREPROCESS_DIR / "m3_srdb_features.pkl"
    report_out_path = WEIGHTS_DIR / "m3_training_report.json"
    joblib.dump(model, model_out_path)
    joblib.dump(selected_features, features_out_path)

    report = {
        "baseline_cv_rmse_log": baseline_rmse,
        "max_allowed_rmse_loss_ratio": MAX_REGRESSION_RMSE_LOSS,
        "selected_feature_count": len(selected_features),
        "total_feature_count": X.shape[1],
        "reduction_curve": reduction_curve,
        "holdout_rmse_log": float(rmse),
        "holdout_mae_log": float(mae),
        "holdout_r2_log": float(r2),
        "selected_features": selected_features,
    }
    with open(report_out_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    
    print(f"\n✅ Trained model saved to: {model_out_path.name}")
    print(f"✅ Feature columns saved to: {features_out_path.name}")
    print(f"✅ Training report saved to: {report_out_path.name}")

if __name__ == "__main__":
    train_srdb_model()