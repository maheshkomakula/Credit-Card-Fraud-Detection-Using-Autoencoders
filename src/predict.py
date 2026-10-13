"""
Prediction Module for Credit Card Fraud Detection using Autoencoders.
Computes reconstruction error for incoming transactions and flags anomalies
based on the learned threshold.
"""

import json
import os
import sys

# Ensure root folder is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.preprocess import scale_transaction, get_feature_names


class FraudDetector:
    """Class wrapper for fraud prediction using trained Autoencoder."""

    def __init__(
        self,
        model_path: str = "models/autoencoder.keras",
        scaler_path: str = "models/scaler.pkl",
        threshold_path: str = "models/threshold.json",
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.threshold_path = threshold_path

        self.model = None
        self.scaler = None
        self.threshold = None
        self.threshold_info = {}
        self.load_artifacts()

    def load_artifacts(self):
        """Load model, scaler, and threshold metadata."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Train the model first using src/train.py.")
        self.model = keras.models.load_model(self.model_path)

        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}.")
        self.scaler = joblib.load(self.scaler_path)

        if os.path.exists(self.threshold_path):
            with open(self.threshold_path, "r") as f:
                self.threshold_info = json.load(f)
                self.threshold = float(self.threshold_info.get("threshold", 0.5))
        else:
            self.threshold = 0.5

    def predict_single(self, transaction: dict, feature_cols: list = None) -> dict:
        """
        Predict for a single transaction.
        transaction: dict of feature_name -> float value
        """
        if feature_cols is None:
            if hasattr(self.scaler, "feature_names_in_"):
                feature_cols = list(self.scaler.feature_names_in_)
            else:
                feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

        # Scale transaction
        scaled_input = scale_transaction(transaction, self.scaler, feature_cols)

        # Reconstruct
        reconstruction = self.model.predict(scaled_input, verbose=0)
        
        # Per-feature squared errors
        feature_sq_err = np.square(scaled_input - reconstruction)[0]
        feature_errors = {feature_cols[i]: float(feature_sq_err[i]) for i in range(len(feature_cols))}

        # Overall MSE reconstruction error
        mse_error = float(np.mean(feature_sq_err))

        is_suspicious = mse_error > self.threshold
        prediction_label = "Suspicious" if is_suspicious else "Legitimate"

        # Normalized anomaly score (0.0 to 1.0 sigmoid or relative ratio)
        ratio = mse_error / self.threshold if self.threshold > 0 else 1.0
        confidence_score = float(1.0 / (1.0 + np.exp(-2.0 * (ratio - 1.0))))

        return {
            "reconstruction_error": mse_error,
            "threshold": self.threshold,
            "prediction": prediction_label,
            "is_suspicious": bool(is_suspicious),
            "anomaly_score": confidence_score,
            "feature_errors": feature_errors,
        }

    def predict_batch(self, df_or_array, feature_cols: list = None) -> pd.DataFrame:
        """
        Predict for a batch of transactions.
        df_or_array: DataFrame or 2D numpy array
        """
        if isinstance(df_or_array, pd.DataFrame):
            if feature_cols is None:
                feature_cols = [c for c in df_or_array.columns if c != "Class"]
            raw_features = df_or_array[feature_cols].copy()
            scaled_features = self.scaler.transform(raw_features)
        else:
            scaled_features = self.scaler.transform(df_or_array)

        reconstructions = self.model.predict(scaled_features, batch_size=1024, verbose=0)
        mse_errors = np.mean(np.square(scaled_features - reconstructions), axis=1)

        predictions = ["Suspicious" if err > self.threshold else "Legitimate" for err in mse_errors]
        is_suspicious = (mse_errors > self.threshold).astype(int)

        result_df = pd.DataFrame({
            "Reconstruction_Error": mse_errors,
            "Threshold": self.threshold,
            "Prediction": predictions,
            "Class_Predicted": is_suspicious,
        })

        if isinstance(df_or_array, pd.DataFrame) and "Class" in df_or_array.columns:
            result_df["True_Class"] = df_or_array["Class"].values

        return result_df


def demo_predictions():
    """Demonstrates prediction output format matching project requirements."""
    detector = FraudDetector()
    print("=" * 60)
    print("Credit Card Fraud Autoencoder - Live Inference Demo")
    print(f"Current Decision Threshold (T): {detector.threshold:.4f}")
    print("=" * 60)

    # Load dataset to extract sample legitimate and fraudulent transactions
    from src.preprocess import load_data
    df = load_data()
    feature_cols = [c for c in df.columns if c != "Class"]

    # Sample Legitimate Transaction (Class = 0)
    normal_sample = df[df["Class"] == 0].iloc[10][feature_cols].to_dict()
    res_a = detector.predict_single(normal_sample, feature_cols)

    # Find a detected Fraudulent Transaction (Class = 1) with higher reconstruction error
    fraud_df = df[df["Class"] == 1]
    detected_fraud_sample = None
    for idx in range(len(fraud_df)):
        sample_dict = fraud_df.iloc[idx][feature_cols].to_dict()
        pred = detector.predict_single(sample_dict, feature_cols)
        if pred["is_suspicious"]:
            detected_fraud_sample = sample_dict
            res_b = pred
            break

    if detected_fraud_sample is None:
        res_b = detector.predict_single(fraud_df.iloc[0][feature_cols].to_dict(), feature_cols)

    print("\nTransaction A")
    print(f"Reconstruction error: {res_a['reconstruction_error']:.4f}")
    print(f"Classification:       {res_a['prediction']}")

    print("\nTransaction B")
    print(f"Reconstruction error: {res_b['reconstruction_error']:.4f}")
    print(f"Classification:       {res_b['prediction']}")
    print("=" * 60)


if __name__ == "__main__":
    demo_predictions()
