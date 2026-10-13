"""
Data Preprocessing Module for Credit Card Fraud Detection.
Handles dataset loading, feature scaling, and train/val/test splits.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str = "dataset/creditcard.csv") -> pd.DataFrame:
    """Load credit card dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def get_feature_names(df: pd.DataFrame) -> list:
    """Return the ordered list of 30 feature names (excluding 'Class')."""
    feature_cols = [c for c in df.columns if c != "Class"]
    return feature_cols


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_ratio_of_test: float = 0.5,
    random_state: int = 42,
    scaler_path: str = "models/scaler.pkl",
):
    """
    Preprocess credit card transactions for Autoencoder training:
    - Normal transactions (Class 0) are used for training.
    - Anomaly detection threshold is tuned on a combined validation set.
    - Model performance is evaluated on a completely held-out test set.
    """
    feature_cols = get_feature_names(df)
    
    # Check and handle missing values if any
    if df[feature_cols].isnull().any().any():
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    # Separate legitimate and fraudulent transactions
    normal_df = df[df["Class"] == 0].reset_index(drop=True)
    fraud_df = df[df["Class"] == 1].reset_index(drop=True)

    # Split normal transactions: train (1 - test_size) and holdout (test_size)
    normal_train, normal_holdout = train_test_split(
        normal_df, test_size=test_size, random_state=random_state
    )

    # Split normal holdout into validation and test sets
    normal_val, normal_test = train_test_split(
        normal_holdout, test_size=(1.0 - val_ratio_of_test), random_state=random_state
    )

    # Split fraud transactions into validation and test sets
    fraud_val, fraud_test = train_test_split(
        fraud_df, test_size=(1.0 - val_ratio_of_test), random_state=random_state
    )

    # Fit scaler ONLY on normal training data to avoid data leakage
    scaler = StandardScaler()
    scaler.fit(normal_train[feature_cols])

    # Transform features
    X_train = scaler.transform(normal_train[feature_cols])

    # Combine validation set (normal + fraud)
    val_combined = pd.concat([normal_val, fraud_val]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    X_val = scaler.transform(val_combined[feature_cols])
    y_val = val_combined["Class"].values

    # Combine test set (normal + fraud)
    test_combined = pd.concat([normal_test, fraud_test]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    X_test = scaler.transform(test_combined[feature_cols])
    y_test = test_combined["Class"].values

    # Save scaler
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "normal_train_count": len(normal_train),
        "val_counts": {"total": len(val_combined), "fraud": int(y_val.sum())},
        "test_counts": {"total": len(test_combined), "fraud": int(y_test.sum())},
    }


def scale_transaction(features, scaler, feature_cols=None) -> np.ndarray:
    """Scale a single transaction or batch given a dict, Series, or 2D array."""
    if isinstance(features, dict):
        if feature_cols is None:
            feature_cols = list(features.keys())
        df = pd.DataFrame([{col: features.get(col, 0.0) for col in feature_cols}])
        return scaler.transform(df)
    elif isinstance(features, pd.Series):
        if feature_cols is None:
            feature_cols = features.index.tolist()
        df = pd.DataFrame([features.to_dict()])
        return scaler.transform(df)
    elif isinstance(features, pd.DataFrame):
        return scaler.transform(features)
    elif isinstance(features, np.ndarray):
        arr = features if features.ndim == 2 else features.reshape(1, -1)
        return scaler.transform(arr)
    else:
        raise TypeError(f"Unsupported features type: {type(features)}")


if __name__ == "__main__":
    print("Testing data preprocessing pipeline...")
    data = load_data()
    print(f"Loaded dataset with shape {data.shape}")
    processed = preprocess_data(data)
    print(f"X_train shape (Normal only): {processed['X_train'].shape}")
    print(f"X_val shape: {processed['X_val'].shape}, Fraud count: {processed['val_counts']['fraud']}")
    print(f"X_test shape: {processed['X_test'].shape}, Fraud count: {processed['test_counts']['fraud']}")
    print("Scaler saved successfully to models/scaler.pkl")
