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

from src.config import MODEL_DIR
from src.utils import load_model


class TerraFluxPipeline:
    def __init__(self, model_dir: str | Path = MODEL_DIR):
        """
        Initialize the unified pipeline, loading all three pre-trained modules.
        If a module is not found, a warning is raised and self.{module} is None.
        """
        self.model_dir = Path(model_dir)
        
        # Module 1
        soc_model_path = self.model_dir / "module1_soc_model.pkl"
        soc_scaler_path = self.model_dir / "esa_scaler.pkl"
        soc_pca_path = self.model_dir / "esa_pca.pkl"

        self.soc_model = load_model(soc_model_path) if soc_model_path.exists() else None
        self.soc_scaler = load_model(soc_scaler_path) if soc_scaler_path.exists() else None
        self.soc_pca = load_model(soc_pca_path) if soc_pca_path.exists() else None

        # Module 2
        risk_model_path = self.model_dir / "module2_risk_model.pkl"
        risk_scaler_path = self.model_dir / "rural_scaler.pkl"
        
        self.risk_model = load_model(risk_model_path) if risk_model_path.exists() else None
        self.risk_scaler = load_model(risk_scaler_path) if risk_scaler_path.exists() else None

        # Module 3
        benchmarks_path = self.model_dir / "module3_benchmarks.json"
        if benchmarks_path.exists():
            with open(benchmarks_path, "r", encoding="utf-8") as f:
                self.benchmarks = json.load(f)
        else:
            self.benchmarks = None


    def predict_soc(self, features: dict | pd.DataFrame) -> float | list[float]:
        """
        Predict Soil Organic Carbon (logoc_d.f) using Module 1.
        Expects raw (or minimally pre-processed) features.
        Normally would scale and PCA here. Currently assumes pre-processed 
        or handles it if raw passed.
        """
        if self.soc_model is None:
            raise ValueError("Module 1 SOC model is not loaded.")
        
        if isinstance(features, dict):
            features = pd.DataFrame([features])
            
        # Optional: insert scaling and PCA logic here based on trained pipeline
        # For now, just call predict
        return self.soc_model.predict(features).tolist()  # type: ignore

    def predict_risk(self, features: dict | pd.DataFrame) -> list[str]:
        """
        Classify region into Risk (Low/Medium/High) using Module 2.
        """
        if self.risk_model is None:
            raise ValueError("Module 2 Risk model is not loaded.")
        
        if isinstance(features, dict):
            features = pd.DataFrame([features])
            
        return self.risk_model.predict(features).tolist()  # type: ignore

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
             
        # 3. Module 3
        if self.benchmarks:
            result["biome_benchmarks"] = self.benchmark_respiration(biome)
            
        return result
