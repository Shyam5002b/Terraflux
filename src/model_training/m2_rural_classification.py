import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Paths
DATA_PATH = "data/processed/rural_carbon_processed.csv"
WEIGHTS_DIR = "models/weights"
PREPROCESS_DIR = "models/preprocessors"
MODEL_PATH = os.path.join(WEIGHTS_DIR, "m2_rural_risk_model.pkl")
ENCODER_PATH = os.path.join(PREPROCESS_DIR, "m2_risk_encoder.pkl")

def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Target and features
    # We drop Carbon_Emission_tCO2 to prevent data leakage since Emission_Risk is derived from it
    cols_to_drop = ['Emission_Risk']
    if 'Carbon_Emission_tCO2' in df.columns:
        cols_to_drop.append('Carbon_Emission_tCO2')
        
    X = df.drop(columns=cols_to_drop)
    y_raw = df['Emission_Risk']

    # Encode target labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    
    # Display class mapping
    class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print(f"Class mapping: {class_mapping}")

    # Train/Test Split
    print("Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Classifier
    print("Initializing and training Baseline XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric="mlogloss"
    )
    
    model.fit(X_train, y_train)

    # Evaluate
    print("Evaluating baseline model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Save model and encoder
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(PREPROCESS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Label Encoder saved to {ENCODER_PATH}")

if __name__ == '__main__':
    main()
