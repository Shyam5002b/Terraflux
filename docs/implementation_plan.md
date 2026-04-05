# TerraFlux — Modular Carbon Framework Implementation Plan

## Overview

Build a **3-module ML framework** where each module answers a different part of the soil carbon problem, using datasets that cannot be merged due to incompatible coordinate systems.

| Module | Dataset | Task | Output |
|--------|---------|------|--------|
| **Module 1** | ESA EO4 Train/Test | Regression | Predicted SOC (`logoc_d.f`) |
| **Module 2** | Rural Carbon | Classification | Emission Risk Level (Low/Medium/High) |
| **Module 3** | SRDB Data V5 | Temporal Analysis | Seasonal respiration benchmarks |

---

## EDA Verification Results

> [!NOTE]
> All 6 EDA notebooks have been verified against the raw CSV files. Key confirmations:
> - Rural Carbon: 3000×13, zero missing values, target = `Carbon_Emission_tCO2` ✅
> - ESA Train: 23898×227, zero missing, target = `logoc_d.f` ✅
> - ESA Test: 7138×226, same features minus `logoc_d.f` ✅
> - SRDB Data: 10366×85, seasonal columns confirmed (`Rs_spring/summer/autumn/winter`) ✅
> - SRDB Equations & Studies: metadata files — not used in modelling ✅

---

## Proposed Changes

### Project Structure

```
Terraflux/
├── data/
│   ├── raw/                          # (existing — untouched)
│   └── processed/                    # (currently empty — will be populated)
│       ├── esa_train_processed.csv
│       ├── esa_test_processed.csv
│       ├── rural_carbon_processed.csv
│       └── srdb_temporal_processed.csv
├── src/
│   ├── __init__.py                   # [NEW]
│   ├── config.py                     # [NEW] Central paths & constants
│   ├── data_processing/              # [NEW] Processing pipelines
│   │   ├── __init__.py
│   │   ├── esa_processor.py          # ESA train/test processing
│   │   ├── rural_processor.py        # Rural carbon processing
│   │   └── srdb_processor.py         # SRDB temporal processing
│   ├── model_training/               # [NEW] Model training & evaluation (code)
│   │   ├── __init__.py
│   │   ├── module1_soc_regressor.py  # ESA SOC regression
│   │   ├── module2_risk_classifier.py# Rural emission classifier
│   │   └── module3_temporal.py       # SRDB temporal analysis
│   ├── pipeline.py                   # [NEW] Unified inference pipeline
│   └── utils.py                      # [NEW] Shared utilities
├── notebooks/
│   ├── EDA_01–06                     # (existing EDA notebooks)
│   ├── 03_data_processing.ipynb      # [NEW] Run processing pipelines
│   ├── 04_module1_training.ipynb     # [NEW] Train SOC regressor
│   ├── 05_module2_training.ipynb     # [NEW] Train risk classifier
│   ├── 06_module3_temporal.ipynb     # [NEW] SRDB temporal analysis
│   └── 07_combined_inference.ipynb   # [NEW] Combined framework demo
├── models/                           # Saved model artifacts
│   ├── module1_soc_model.pkl
│   ├── module2_risk_model.pkl
│   └── module3_benchmarks.json
├── outputs/                          # Evaluation results & plots
├── requirements.txt                  # (updated)
└── README.md                         # [NEW]
```

---

### Component Details

---

### 1. Configuration & Utilities

#### [NEW] [config.py](file:///c:/Users/justi/Downloads/ML%20project%20data/Terraflux/src/config.py)

Central configuration: file paths, random seed, feature group definitions, hyperparameter defaults.

#### [NEW] [utils.py](file:///c:/Users/justi/Downloads/ML%20project%20data/Terraflux/src/utils.py)

Shared helpers: `load_csv_safe()` (encoding fallback), evaluation metric functions (RMSE, R², F1), plotting helpers, model serialization wrappers.

---

### 2. Data Processing Pipelines

Each processor reads from `data/raw/`, transforms, and saves to `data/processed/`.

#### [NEW] [esa_processor.py](file:///c:/Users/justi/Downloads/ML%20project%20data/Terraflux/src/data_processing/esa_processor.py)

**Input:** `esa_eo4_train_soc.csv`, `esa_eo4_test_soc.csv`

Processing steps:
1. Drop ID columns (`olc_id`, `UUID`, `sample_id`)
2. Create `soil_depth = hzn_bot - hzn_top` feature
3. Group seasonal features by band — compute mean & std across 6 seasons per band (e.g., `ndvi_mean`, `ndvi_std`) → reduces 220+ cols to ~80
4. Identify and handle zero-variance features
5. Standard-scale numeric features (fit on train, transform both)
6. Save scaler + processed CSVs

#### [NEW] [rural_processor.py](file:///c:/Users/justi/Downloads/ML%20project%20data/Terraflux/src/data_processing/rural_processor.py)

**Input:** `rural_carbon_dataset1.csv`

Processing steps:
1. Create target: bin `Carbon_Emission_tCO2` into 3 classes — **Low** (bottom 33%), **Medium** (middle 33%), **High** (top 33%) — using quantile-based cuts
2. One-hot encode `Region` and `Crop_Type`
3. Feature engineering: `livestock_total = Livestock_Cows + Livestock_Pigs`, `energy_renewable_kwh = Household_Energy_kWh * Renewable_Energy_Fraction`
4. Standard-scale numeric features
5. Save processed CSV with risk labels

#### [NEW] [srdb_processor.py](file:///c:/Users/justi/Downloads/ML%20project%20data/Terraflux/src/data_processing/srdb_processor.py)

**Input:** `srdb-data-V5.csv`

Processing steps:
1. Filter to rows with `Rs_annual > 0` AND valid `Study_midyear` AND valid `Latitude/Longitude`
2. Select key columns: `Study_midyear`, `Latitude`, `Longitude`, `Biome`, `Ecosystem_type`, `MAT`, `MAP`, `Rs_annual`, `Rs_spring`, `Rs_summer`, `Rs_autumn`, `Rs_winter`, `Rs_growingseason`
3. Create seasonal ratios: `Rs_summer_ratio = Rs_summer / Rs_annual` (where available)
4. Add decade feature: `decade = floor(Study_midyear / 10) * 10`
5. Impute `MAT`/`MAP` using group median by `Biome` (where missing)
6. Save processed CSV

---

### 3. Module 1 — Remote Sensing SOC Regressor

#### [NEW] [module1_soc_regressor.py](file:///c:/Users/justi/Downloads/ML%20project%20data/Terraflux/src/model_training/module1_soc_regressor.py)

**Task:** Predict `logoc_d.f` (log soil organic carbon) from satellite features.

**Approach:**
- **Models:** RandomForest, XGBoost (ensemble comparison)
- **Validation:** 5-fold cross-validation on training data (spatial CV if X/Y used)
- **Metrics:** RMSE, MAE, R²
- **Feature importance:** Top-20 SHAP or permutation importance
- **Pipeline:** Load processed data → train → evaluate → save model + metrics
- **Prediction:** Generate predictions on test set, save to `outputs/`

#### [NEW] [04_module1_training.ipynb](file:///c:/Users/justi/Downloads/ML%20project%20data/Terraflux/notebooks/04_module1_training.ipynb)

Interactive notebook: runs the module1 pipeline with visualizations — learning curves, residual plots, feature importance bar charts, actual vs predicted scatter.

---

### 4. Module 2 — Human Risk Classifier

#### [NEW] [module2_risk_classifier.py](file:///c:/Users/justi/Downloads/ML%20project%20data/Terraflux/src/model_training/module2_risk_classifier.py)

**Task:** Classify regions into Low/Medium/High emission risk from agricultural features.

**Approach:**
- **Models:** RandomForest Classifier, XGBoost Classifier
- **Validation:** Stratified 5-fold CV (preserves class balance)
- **Metrics:** Accuracy, Weighted F1, Confusion Matrix, Classification Report
- **Pipeline:** Load processed data → train → evaluate → save model

#### [NEW] [05_module2_training.ipynb](file:///c:/Users/justi/Downloads/ML%20project%20data/Terraflux/notebooks/05_module2_training.ipynb)

Interactive notebook: confusion matrix heatmap, ROC curves per class, feature importance, sample predictions.

---

### 5. Module 3 — SRDB Temporal Benchmark

#### [NEW] [module3_temporal.py](file:///c:/Users/justi/Downloads/ML%20project%20data/Terraflux/src/model_training/module3_temporal.py)

**Task:** Analyze seasonal & decadal trends in soil respiration globally.

**Approach:**
- **Seasonal analysis:** Box plots & statistical tests of Rs by season across biomes
- **Temporal trend:** Rs_annual trend over decades, grouped by biome
- **Climate sensitivity:** Rs vs MAT/MAP regression per biome
- **Benchmark generation:** Save summary statistics as JSON (mean Rs per biome, per decade, per season) — used by combined pipeline as "reality check"

#### [NEW] [06_module3_temporal.ipynb](file:///c:/Users/justi/Downloads/ML%20project%20data/Terraflux/notebooks/06_module3_temporal.ipynb)

Interactive notebook: time series plots, seasonal decomposition, biome-level heatmaps.

---

### 6. Combined Inference Pipeline

#### [NEW] [pipeline.py](file:///c:/Users/justi/Downloads/ML%20project%20data/Terraflux/src/pipeline.py)

Unified `TerraFluxPipeline` class that:
1. Loads all 3 trained modules
2. Accepts satellite features → runs Module 1 → outputs SOC prediction
3. Accepts agricultural inputs → runs Module 2 → outputs risk classification
4. Cross-references both results with Module 3 benchmarks
5. Returns a combined assessment dictionary

```python
class TerraFluxPipeline:
    def __init__(self, model_dir='models/'):
        self.soc_model = load_model('module1_soc_model.pkl')
        self.risk_model = load_model('module2_risk_model.pkl')
        self.benchmarks = load_json('module3_benchmarks.json')
    
    def predict_soc(self, satellite_features: dict) -> float: ...
    def classify_risk(self, agricultural_inputs: dict) -> str: ...
    def get_benchmark(self, biome: str, season: str) -> dict: ...
    def full_assessment(self, satellite_features, agricultural_inputs, biome, season) -> dict: ...
```

#### [NEW] [07_combined_inference.ipynb](file:///c:/Users/justi/Downloads/ML%20project%20data/Terraflux/notebooks/07_combined_inference.ipynb)

Demo notebook showing the full pipeline working end-to-end with sample inputs.

---

## Decisions (User-Approved)

- ✅ **Module 1 feature reduction**: Use **PCA** targeting **95% variance explained**. Adjust if retention is too low.
- ✅ **Module 2 risk binning**: Use **quantile-based 3-class** splits for initial modelling. Check emission distribution first for skew/outliers. Refine thresholds after model analysis.
- ⏸️ **Module 3 SRDB filtering**: **Deferred** — decide after Module 1 PCA analysis to check if seasonal detail is retained.
- ✅ **EDA notebooks**: Keep all 6 (including equations & studies) — decide at project end.

---

## Verification Plan

### Automated Tests
- Run `python -m pytest tests/` covering:
  - Data processing: verify output shapes, no missing targets, correct dtypes
  - Model training: verify models achieve minimum thresholds (R² > 0.5 for Module 1, F1 > 0.6 for Module 2)
  - Pipeline: verify end-to-end inference returns expected output format

### Notebook Execution
- Run all notebooks 03–07 sequentially to verify no errors
- Check that `data/processed/` and `models/` are populated correctly

### Manual Verification
- Inspect Module 1 residual plots for systematic bias
- Check Module 2 confusion matrix for class imbalance issues
- Compare Module 3 benchmarks against published soil respiration literature values
