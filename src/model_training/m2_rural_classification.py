import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import CV_FOLDS, HYPERPARAM_SEARCH_ITERS, MAX_CLASSIFICATION_F1_DROP, RANDOM_SEED

# Paths
DATA_PATH = "data/processed/rural_carbon_processed.csv"
WEIGHTS_DIR = "models/weights"
PREPROCESS_DIR = "models/preprocessors"
MODEL_PATH = os.path.join(WEIGHTS_DIR, "m2_rural_risk_model.pkl")
ENCODER_PATH = os.path.join(PREPROCESS_DIR, "m2_risk_encoder.pkl")
SCALER_PATH = os.path.join(PREPROCESS_DIR, "rural_scaler.pkl")
FEATURES_PATH = os.path.join(PREPROCESS_DIR, "m2_selected_features.pkl")
REPORT_PATH = os.path.join(WEIGHTS_DIR, "m2_training_report.json")


def _is_binary_series(series: pd.Series) -> bool:
    unique_vals = set(series.dropna().unique().tolist())
    return unique_vals.issubset({0, 1, False, True})


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder, StandardScaler]:
    cols_to_drop = ["Emission_Risk"]
    if "Carbon_Emission_tCO2" in df.columns:
        cols_to_drop.append("Carbon_Emission_tCO2")

    X = df.drop(columns=cols_to_drop).copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    y_raw = df["Emission_Risk"].copy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    scale_cols = [c for c in numeric_cols if not _is_binary_series(X[c])]

    scaler = StandardScaler()
    if scale_cols:
        X.loc[:, scale_cols] = scaler.fit_transform(X[scale_cols])
    else:
        # Keep a valid artifact contract for inference even if there are no scale columns.
        scaler.fit(np.zeros((1, 1)))
        scaler.feature_names_in_ = np.array([], dtype=object)

    return X, y, encoder, scaler


def _tune_classifier(X: pd.DataFrame, y: np.ndarray) -> tuple[XGBClassifier, float]:
    base = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y)),
        eval_metric="mlogloss",
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

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=HYPERPARAM_SEARCH_ITERS,
        scoring="f1_weighted",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=2,
        verbose=1,
    )
    search.fit(X, y)
    return search.best_estimator_, float(search.best_score_)


def _reduce_features(
    estimator: XGBClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    baseline_f1: float,
) -> tuple[list[str], list[dict[str, float]]]:
    importances = estimator.feature_importances_
    ordered_idx = np.argsort(importances)[::-1]
    ordered_features = X.columns[ordered_idx].tolist()

    candidate_counts = sorted({
        8,
        12,
        16,
        24,
        32,
        48,
        64,
        min(96, X.shape[1]),
        min(128, X.shape[1]),
        X.shape[1],
    })

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    reduction_curve: list[dict[str, float]] = []

    for k in candidate_counts:
        selected = ordered_features[:k]
        candidate_model = clone(estimator)
        scores = cross_val_score(candidate_model, X[selected], y, cv=cv, scoring="f1_weighted", n_jobs=2)
        mean_f1 = float(scores.mean())
        reduction_curve.append({"n_features": float(k), "cv_weighted_f1": mean_f1})

    selected_features = ordered_features
    for row in reduction_curve:
        if row["cv_weighted_f1"] >= baseline_f1 - MAX_CLASSIFICATION_F1_DROP:
            selected_features = ordered_features[: int(row["n_features"])]
            break

    return selected_features, reduction_curve


def main() -> None:
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    X, y, encoder, scaler = _prepare_features(df)
    class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print(f"Class mapping: {class_mapping}")
    print(f"Prepared features shape: {X.shape}")

    print("Running hyperparameter search with cross-validation...")
    best_model, baseline_f1 = _tune_classifier(X, y)
    print(f"Best CV weighted-F1: {baseline_f1:.4f}")

    print("Running feature reduction sweep...")
    selected_features, reduction_curve = _reduce_features(best_model, X, y, baseline_f1)
    print(f"Selected {len(selected_features)} / {X.shape[1]} features")

    # Final train on selected features
    final_model = clone(best_model)
    final_model.fit(X[selected_features], y)

    # Holdout sanity check for user-facing report
    X_train, X_test, y_train, y_test = train_test_split(
        X[selected_features], y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    sanity_model = clone(best_model)
    sanity_model.fit(X_train, y_train)
    y_pred = sanity_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nHoldout Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(PREPROCESS_DIR, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(selected_features, FEATURES_PATH)

    report = {
        "baseline_cv_weighted_f1": baseline_f1,
        "selected_feature_count": len(selected_features),
        "total_feature_count": X.shape[1],
        "max_allowed_f1_drop": MAX_CLASSIFICATION_F1_DROP,
        "reduction_curve": reduction_curve,
        "holdout_accuracy": float(accuracy),
        "selected_features": selected_features,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Label Encoder saved to {ENCODER_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
    print(f"Selected feature list saved to {FEATURES_PATH}")
    print(f"Training report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
