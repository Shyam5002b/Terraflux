"""
TerraFlux - Shared Utilities

Reusable helpers for loading data, evaluating models, plotting, and
serialising artifacts. Used by all three modules and notebooks.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# --------------------------------------------------------------------------
# 1. Data Loading
# --------------------------------------------------------------------------

def load_csv_safe(
    file_path: Union[str, Path],
    encodings: Optional[List[str]] = None,
    **read_kwargs,
) -> pd.DataFrame:
    """
    Try multiple encodings to load a CSV.  Raises on total failure.

    Parameters
    ----------
    file_path : path-like
        Path to the CSV file.
    encodings : list[str], optional
        Ordered list of encodings to attempt.  Defaults to
        ``['utf-8', 'latin1', 'cp1252']``.
    **read_kwargs
        Extra keyword arguments forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    encodings = encodings or ["utf-8", "latin1", "cp1252"]

    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, **read_kwargs)
            print(f"  Loaded: {file_path.name:<30s} (encoding={enc}, "
                  f"shape={df.shape})")
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"Could not decode {file_path.name} with any of {encodings}"
    )


# --------------------------------------------------------------------------
# 2. Regression Metrics (Module 1)
# --------------------------------------------------------------------------

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """Return RMSE, MAE, and R² as a dict."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    metrics = {
        f"{prefix}RMSE": round(rmse, 4),
        f"{prefix}MAE":  round(mae, 4),
        f"{prefix}R2":   round(r2, 4),
    }
    return metrics


def print_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """Compute, print, and return regression metrics."""
    m = regression_metrics(y_true, y_pred)
    print(f"\n{'---'*14}")
    print(f"  {model_name} - Regression Metrics")
    print(f"{'---'*14}")
    for k, v in m.items():
        print(f"  {k:>8s}: {v:.4f}")
    print(f"{'---'*14}\n")
    return m


# --------------------------------------------------------------------------
# 3. Classification Metrics (Module 2)
# --------------------------------------------------------------------------

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """Return accuracy, weighted-F1, and per-class report dict."""
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, average="weighted"))

    metrics = {
        f"{prefix}Accuracy":    round(acc, 4),
        f"{prefix}Weighted_F1": round(f1, 4),
    }
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    model_name: str = "Model",
) -> None:
    """Print a full sklearn classification report."""
    print(f"\n{'---'*17}")
    print(f"  {model_name} - Classification Report")
    print(f"{'---'*17}")
    print(classification_report(y_true, y_pred, target_names=labels))


# --------------------------------------------------------------------------
# 4. Plotting Helpers
# --------------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
    ax: Optional[Axes] = None,
) -> Axes:
    """Scatter plot of actual vs predicted with 1:1 line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="1:1 line")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    ax: Optional[Axes] = None,
) -> Axes:
    """Residual (error) vs predicted value."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.3, s=10, color="coral")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title(title)
    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    ax: Optional[Axes] = None,
) -> Axes:
    """Seaborn heatmap of the confusion matrix."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels if labels is not None else "auto",
        yticklabels=labels if labels is not None else "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    title: str = "Feature Importance",
    ax: Optional[Axes] = None,
) -> Axes:
    """Horizontal bar chart of the top-N important features."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    idx = np.argsort(importances)[-top_n:]
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="teal",
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    return ax


def plot_pca_variance(
    explained_variance_ratio: np.ndarray,
    threshold: float = 0.95,
    ax: Optional[Axes] = None,
) -> Axes:
    """Cumulative explained variance plot for PCA."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    cumvar = np.cumsum(explained_variance_ratio)
    ax.plot(range(1, len(cumvar) + 1), cumvar, "b-o", markersize=3)
    ax.axhline(threshold, color="red", ls="--",
               label=f"{threshold:.0%} threshold")

    # Mark where threshold is crossed
    n_components = int(np.searchsorted(cumvar, threshold) + 1)
    ax.axvline(n_components, color="green", ls=":",
               label=f"{n_components} components")

    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA — Explained Variance")
    ax.legend()
    return ax


# --------------------------------------------------------------------------
# 5. Model Serialisation
# --------------------------------------------------------------------------

def save_model(model: object, path: Union[str, Path]) -> None:
    """Pickle a model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved -> {path}")


def load_model(path: Union[str, Path]) -> object:
    """Load a pickled model from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded <- {path}")
    return model


def save_json(data: dict, path: Union[str, Path]) -> None:
    """Save a dict as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"JSON saved  -> {path}")


def load_json(path: Union[str, Path]) -> dict:
    """Load a JSON file as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)
