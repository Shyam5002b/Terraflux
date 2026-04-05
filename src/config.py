"""
TerraFlux - Central Configuration

All project-wide paths, constants, feature definitions, and hyperparameter
defaults live here. Import this module everywhere to avoid hardcoded values.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# 1. Project Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/Terraflux/

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DATA_DIR  = DATA_DIR / "raw"
PROC_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR   = PROJECT_ROOT / "models"
OUTPUT_DIR  = PROJECT_ROOT / "outputs"

# Ensure output directories exist
PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# 2. Raw File Names
# --------------------------------------------------------------------------
RAW_FILES = {
    "esa_train":      RAW_DATA_DIR / "esa_eo4_train_soc.csv",
    "esa_test":       RAW_DATA_DIR / "esa_eo4_test_soc.csv",
    "rural_carbon":   RAW_DATA_DIR / "rural_carbon_dataset1.csv",
    "srdb_data":      RAW_DATA_DIR / "srdb-data-V5.csv",
    "srdb_equations": RAW_DATA_DIR / "srdb-equations-V5.csv",
    "srdb_studies":   RAW_DATA_DIR / "srdb-studies-V5.csv",
}

PROCESSED_FILES = {
    "esa_train":      PROC_DATA_DIR / "esa_train_processed.csv",
    "esa_test":       PROC_DATA_DIR / "esa_test_processed.csv",
    "rural_carbon":   PROC_DATA_DIR / "rural_carbon_processed.csv",
    "srdb_temporal":  PROC_DATA_DIR / "srdb_temporal_processed.csv",
}

# --------------------------------------------------------------------------
# 3. Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# 4. ESA Dataset Settings (Module 1)
# --------------------------------------------------------------------------
ESA_TARGET_COL = "logoc_d.f"

# Columns to drop before modelling (IDs / non-features)
ESA_DROP_COLS = ["olc_id", "UUID", "sample_id"]

# PCA variance retention target (95%, adjust if retention is too low)
PCA_VARIANCE_THRESHOLD = 0.95

# Seasonal band prefixes — each has 6 bi-monthly columns
ESA_SEASONAL_BAND_PREFIXES = [
    "blue", "bsi", "evi", "fapar", "green", "msavi",
    "nbr2", "nbr", "ndmi", "ndsi", "ndsmi", "ndti",
    "ndvi", "ndwi", "nir", "nirv", "red", "savi",
    "swir1", "swir2", "thermal",
]

# Season suffixes used in the ESA column names
ESA_SEASON_SUFFIXES = [
    "0101_0228", "0301_0430", "0501_0630",
    "0701_0831", "0901_1031", "1101_1231",
]

# Module 1 hyperparameter defaults
MODULE1_PARAMS = {
    "rf": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    },
    "xgb": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
    },
}

# --------------------------------------------------------------------------
# 5. Rural Carbon Settings (Module 2)
# --------------------------------------------------------------------------
RURAL_TARGET_COL = "Carbon_Emission_tCO2"
RURAL_RISK_COL   = "Emission_Risk"      # created during processing
RURAL_N_CLASSES  = 3                     # Low / Medium / High
RURAL_CLASS_LABELS = ["Low", "Medium", "High"]

# Categorical columns to one-hot encode
RURAL_CAT_COLS = ["Region", "Crop_Type"]

# Module 2 hyperparameter defaults
MODULE2_PARAMS = {
    "rf": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    },
    "xgb": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": RANDOM_SEED,
    },
}

# --------------------------------------------------------------------------
# 6. SRDB Temporal Settings (Module 3)
# --------------------------------------------------------------------------
SRDB_KEY_COLS = [
    "Record_number", "Study_midyear", "Latitude", "Longitude",
    "Biome", "Ecosystem_type", "MAT", "MAP",
    "Rs_annual", "Rs_spring", "Rs_summer", "Rs_autumn", "Rs_winter",
    "Rs_growingseason",
]

# --------------------------------------------------------------------------
# 7. Cross-Validation
# --------------------------------------------------------------------------
CV_FOLDS = 5
