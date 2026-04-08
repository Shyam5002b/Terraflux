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

    # 3. Hyperparameter Search with CV
    print("Running hyperparameter search with cross-validation...")
    xgb_params = MODULE1_PARAMS.get("xgb", {})
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=2,
        tree_method="hist",
    )

    param_dist = {
        "n_estimators": [xgb_params.get("n_estimators", 300), 500, 700],
        "max_depth": [4, 6, 8, xgb_params.get("max_depth", 6)],
        "learning_rate": [0.03, 0.05, 0.08, xgb_params.get("learning_rate", 0.1)],
        "subsample": [0.7, 0.8, xgb_params.get("subsample", 0.8), 1.0],
        "colsample_bytree": [0.6, 0.8, xgb_params.get("colsample_bytree", 0.8), 1.0],
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
    print(f"Best CV RMSE: {baseline_rmse:.4f}")

    # 4. Feature Reduction Sweep over PCA components
    ordered_features = sorted(
        X.columns.tolist(),
        key=lambda name: int(name.split("_")[1]) if "_" in name and name.split("_")[1].isdigit() else 9999,
    )
    candidate_counts = sorted({
        10,
        15,
        20,
        30,
        40,
        50,
        min(60, X.shape[1]),
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
        reduction_curve.append({"n_features": float(k), "cv_rmse": mean_rmse})

    selected_features = ordered_features
    rmse_limit = baseline_rmse * (1 + MAX_REGRESSION_RMSE_LOSS)
    for row in reduction_curve:
        if row["cv_rmse"] <= rmse_limit:
            selected_features = ordered_features[: int(row["n_features"])]
            break

    print(f"Selected {len(selected_features)} / {X.shape[1]} PCA features")

    # 5. Train-Validation Split (80/20)
    X_selected = X[selected_features]
    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"Training split: {len(X_train)} samples\nValidation split: {len(X_val)} samples\n")

    # 6. Final Train
    model = clone(best_model)
    print("\nTraining model... (This may take a minute)")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    # 7. Evaluate Performance
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

    # 8. Save the Model and report
    model_out_path = WEIGHTS_DIR / "esa_soc_model.pkl"
    report_out_path = WEIGHTS_DIR / "m1_training_report.json"
    joblib.dump(model, model_out_path)

    report = {
        "baseline_cv_rmse": baseline_rmse,
        "max_allowed_rmse_loss_ratio": MAX_REGRESSION_RMSE_LOSS,
        "selected_feature_count": len(selected_features),
        "total_feature_count": X.shape[1],
        "selected_features": selected_features,
        "reduction_curve": reduction_curve,
        "validation_rmse": float(rmse),
        "validation_mae": float(mae),
        "validation_r2": float(r2),
    }
    with open(report_out_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print(f"✅ Trained model successfully saved to: {model_out_path.name}")
    print(f"✅ Training report saved to: {report_out_path.name}")

if __name__ == "__main__":
    train_module_1()
