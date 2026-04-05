"""
Unified TerraFlux Inference Pipeline

This module brings together the three disparate ML sub-models (ESA SOC Regressor,
Rural Risk Classifier, and SRDB Benchmarks). It accepts real-world queries      
containing multiple sources of data, processes them, runs the appropriate models,
and returns a combined JSON assessment.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
import numpy as np

from src.config import WEIGHTS_DIR, PREPROCESS_DIR
from src.utils import load_model


class TerraFluxPipeline:
    def __init__(self, weights_dir: str | Path = WEIGHTS_DIR, prep_dir: str | Path = PREPROCESS_DIR):
        """
        Initialize the unified pipeline, loading all three pre-trained modules. 
        If a module is not found, a warning is raised and self.{module} is None.
        """
        self.weights_dir = Path(weights_dir)
        self.prep_dir = Path(prep_dir)

        # Module 1
        soc_model_path = self.weights_dir / "esa_soc_model.pkl"
        soc_scaler_path = self.prep_dir / "esa_scaler.pkl"
        soc_pca_path = self.prep_dir / "esa_pca.pkl"

        self.soc_model = load_model(soc_model_path) if soc_model_path.exists() else None
        self.soc_scaler = load_model(soc_scaler_path) if soc_scaler_path.exists() else None
        self.soc_pca = load_model(soc_pca_path) if soc_pca_path.exists() else None

        # Module 2
        risk_model_path = self.weights_dir / "m2_rural_risk_model.pkl"
        risk_scaler_path = self.prep_dir / "rural_scaler.pkl"

        self.risk_model = load_model(risk_model_path) if risk_model_path.exists() else None
        self.risk_scaler = load_model(risk_scaler_path) if risk_scaler_path.exists() else None

        # Module 3
        benchmarks_path = self.weights_dir / "m3_srdb_regression_model.pkl"     
        features_path = self.prep_dir / "m3_srdb_features.pkl"
        
        self.srdb_model = load_model(benchmarks_path) if benchmarks_path.exists() else None
        self.srdb_features = load_model(features_path) if features_path.exists() else None
        self.benchmarks = None  # TODO: Load actual benchmarks if available     


    def predict_soc(self, features: dict | pd.DataFrame) -> list[float]:
        """
        Predict Soil Organic Carbon (logoc_d.f) using Module 1.
        Now directly translates raw input rows using the built preprocessors.
        """
        if self.soc_model is None or self.soc_scaler is None or self.soc_pca is None:
            raise ValueError("Module 1 SOC model or preprocessors are not loaded.")

        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        from src.config import ESA_DROP_COLS, ESA_SEASONAL_BAND_PREFIXES, ESA_SEASON_SUFFIXES
        
        # 1. Clean IDs
        drop_cols = [c for c in ESA_DROP_COLS if c in df.columns]
        if 'logoc_d.f' in df.columns:
            drop_cols.append('logoc_d.f')
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

        # 2. Engineer Soil Depth
        if 'hzn_bot' in df.columns and 'hzn_top' in df.columns:
            df['soil_depth'] = df['hzn_bot'] - df['hzn_top']

        # 3. Compress Time-Series
        cols_to_drop = []
        for prefix in ESA_SEASONAL_BAND_PREFIXES:
            band_cols = [c for c in df.columns if c.startswith(f"{prefix}_") and any(suffix in c for suffix in ESA_SEASON_SUFFIXES)]
            if band_cols:
                df[f"{prefix}_mean"] = df[band_cols].mean(axis=1)
                df[f"{prefix}_std"] = df[band_cols].std(axis=1)
                cols_to_drop.extend(band_cols)
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        # Align with scaler
        if hasattr(self.soc_scaler, 'feature_names_in_'):
            expected = self.soc_scaler.feature_names_in_
            missing_cols = [c for c in expected if c not in df.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0.0, index=df.index, columns=missing_cols)
                df = pd.concat([df, missing_df], axis=1)
            df = df[expected]

        # 4. Scale & PCA
        X_scaled = self.soc_scaler.transform(df)
        X_pca = self.soc_pca.transform(X_scaled)
        
        return self.soc_model.predict(X_pca).tolist()  # type: ignore


    def predict_risk(self, features: dict | pd.DataFrame) -> list[str]:
        """
        Classify region into Risk (Low/Medium/High) using Module 2.
        Now executes encoding and feature alignment dynamically.
        """
        if self.risk_model is None or self.risk_scaler is None:
            raise ValueError("Module 2 Risk model or scaler is not loaded.")

        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        from src.config import RURAL_CAT_COLS, RURAL_TARGET_COL, RURAL_RISK_COL
        
        # 1. Drop Targets
        cols_to_drop = [RURAL_TARGET_COL, RURAL_RISK_COL]
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors="ignore")
        
        # 2. Extract Relational Parameters
        if "Livestock_Cows" in df.columns and "Livestock_Pigs" in df.columns:
            df["Livestock_Total"] = df["Livestock_Cows"] + df["Livestock_Pigs"]
        if "Household_Energy_kWh" in df.columns and "Renewable_Energy_Fraction" in df.columns:
            df["Renewable_Energy_kWh"] = df["Household_Energy_kWh"] * df["Renewable_Energy_Fraction"]
            
        # 3. Categorical encoding
        cats_present = [c for c in RURAL_CAT_COLS if c in df.columns]
        if cats_present:
            df = pd.get_dummies(df, columns=cats_present, drop_first=True)

        # Align EXACTLY with XGBoost training cols
        if hasattr(self.risk_model, 'feature_names_in_'):
            expected_cols = self.risk_model.feature_names_in_
            missing_cols = [c for c in expected_cols if c not in df.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
                df = pd.concat([df, missing_df], axis=1)
            df = df[expected_cols]

        # 4. Scale ONLY the numeric cols (XGBoost expects standard scaled input on original numeric fields)
        if hasattr(self.risk_scaler, 'feature_names_in_'):
            scale_cols = self.risk_scaler.feature_names_in_
            df[scale_cols] = self.risk_scaler.transform(df[scale_cols])

        # Execute Prediction and decode categorical risk via inverse_transform
        risk_encoder_path = self.prep_dir / "m2_risk_encoder.pkl"
        if risk_encoder_path.exists():
            encoder = joblib.load(risk_encoder_path)
            raw_preds = self.risk_model.predict(df)
            return encoder.inverse_transform(raw_preds).tolist()
        else:
            # Fallback to integer predictions if the encoder artifact is lost
            return self.risk_model.predict(df).tolist()  # type: ignore

    def predict_srdb(self, rural_data: pd.DataFrame, biome: str) -> list[float]:
        """
        Predict Annual Soil Respiration (Rs_annual) using Module 3 (SRDB).
        Dynamically cross-references Rural climate parameters.
        """
        if self.srdb_model is None or self.srdb_features is None:
            raise ValueError("Module 3 SRDB model or features are not loaded.")

        if isinstance(rural_data, dict):
            rural_data = pd.DataFrame([rural_data])
            
        import pandas as pd
        import numpy as np

        # Extract overlapping climate inputs for the global model
        year = rural_data["Year"].iloc[0] if "Year" in rural_data.columns else 2023
        mat = rural_data["Temperature_C"].iloc[0] if "Temperature_C" in rural_data.columns else 15.0
        # Convert explicit monthly rainfall back to naive annual if we must
        map_mm = rural_data["Rainfall_mm"].iloc[0] * 12 if "Rainfall_mm" in rural_data.columns else 1000.0 

        df = pd.DataFrame([{
            "Study_midyear": year,
            "Latitude": 0.0,  # Fallback if unsupplied
            "Longitude": 0.0,
            "MAT": mat,
            "MAP": map_mm,
            f"Biome_{biome}": 1.0
        }])

        # Align with saved training features
        for c in self.srdb_features:
            if c not in df.columns:
                df[c] = 0.0
        df = df[self.srdb_features]

        # Model was trained on log1p(Rs_annual)
        log_preds = self.srdb_model.predict(df)
        return np.expm1(log_preds).tolist()

    def benchmark_respiration(self, biome: str) -> dict:
        """
        Get seasonal benchmarks for a biome from Module 3.
        """
        if self.benchmarks is None:
            return {"status": "Benchmarks unavailble"}

        return self.benchmarks.get("biomes", {}).get(biome, {"status": "Biome not found"})

    def evaluate_region(
        self,
        esa_data: dict | pd.DataFrame,
        rural_data: dict | pd.DataFrame,
        biome: str
    ) -> dict:
        """
        Run the complete unified pipeline to evaluate a region's soil health    
        and carbon emission risk, cross-referenced with global benchmarks.      
        """

        result = {}

        # 1. Module 1
        if self.soc_model:
            try:
                soc_preds = self.predict_soc(esa_data)
                result["predicted_soc_log"] = soc_preds
            except Exception as e:
                result["predicted_soc_log"] = f"Error: {e}"
        else:
            result["predicted_soc_log"] = "Module 1 required"

        # 2. Module 2
        if self.risk_model:
            try:
                risk_preds = self.predict_risk(rural_data)
                result["predicted_emission_risk"] = risk_preds
            except Exception as e:
                 result["predicted_emission_risk"] = f"Error: {e}"
        else:
             result["predicted_emission_risk"] = "Module 2 required"

        # 3. Module 3 (SRDB Temp Regression)
        if self.srdb_model:
            try:
                srdb_preds = self.predict_srdb(rural_data, biome)
                result["predicted_annual_soil_respiration_gC_m2"] = srdb_preds
            except Exception as e:
                result["predicted_annual_soil_respiration_gC_m2"] = f"Error: {e}"
        else:
            result["predicted_annual_soil_respiration_gC_m2"] = "Module 3 required"

        if self.benchmarks:
            result["biome_benchmarks"] = self.benchmark_respiration(biome)

        return result