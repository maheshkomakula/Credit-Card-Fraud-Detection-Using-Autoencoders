"""
Autoencoder Model Training Module for Credit Card Fraud Detection.
Builds, compiles, trains the autoencoder on legitimate transactions,
and selects the optimal anomaly threshold on the validation set.
"""

import json
import os
import sys

# Ensure root folder is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

from src.preprocess import load_data, preprocess_data


def build_autoencoder(input_dim: int = 30) -> keras.Model:
    """
    Builds Autoencoder architecture as specified:
    - Input: 30 features
    - Encoder 1: 16 units (ReLU)
    - Encoder 2: 8 units (ReLU)
    - Latent layer: 4 units (ReLU)
    - Decoder 1: 8 units (ReLU)
    - Decoder 2: 16 units (ReLU)
    - Output: 30 units (Linear)
    """
    inputs = layers.Input(shape=(input_dim,), name="transaction_input")

    # Encoder
    encoded = layers.Dense(16, activation="relu", name="encoder_1")(inputs)
    encoded = layers.Dense(8, activation="relu", name="encoder_2")(encoded)
    latent = layers.Dense(4, activation="relu", name="latent_space")(encoded)

    # Decoder
    decoded = layers.Dense(8, activation="relu", name="decoder_1")(latent)
    decoded = layers.Dense(16, activation="relu", name="decoder_2")(decoded)
    outputs = layers.Dense(input_dim, activation="linear", name="reconstructed_output")(decoded)

    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="CreditCard_Autoencoder")
    return autoencoder


def compute_reconstruction_errors(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Calculates Mean Squared Error (MSE) reconstruction error per sample."""
    reconstructions = model.predict(X, batch_size=1024, verbose=0)
    mse = np.mean(np.square(X - reconstructions), axis=1)
    return mse


def find_optimal_threshold(errors: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Finds optimal anomaly detection threshold on validation data
    by maximizing the F1-Score across precision-recall curve in vectorized numpy.
    """
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, errors)
    
    # Compute F1 for all thresholds instantly
    denom = precisions[:-1] + recalls[:-1]
    f1_scores = np.where(denom > 0, 2 * (precisions[:-1] * recalls[:-1]) / (denom + 1e-12), 0.0)

    best_idx = int(np.argmax(f1_scores))
    best_thresh = float(pr_thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    best_prec = float(precisions[best_idx])
    best_rec = float(recalls[best_idx])

    # Also compute 95th, 99th, and mean+3*std baseline thresholds for comparison
    normal_errors = errors[y_true == 0]
    mean_val = float(np.mean(normal_errors))
    std_val = float(np.std(normal_errors))

    return {
        "threshold": best_thresh,
        "f1_score": best_f1,
        "precision": best_prec,
        "recall": best_rec,
        "baseline_95th": float(np.percentile(normal_errors, 95)),
        "baseline_99th": float(np.percentile(normal_errors, 99)),
        "baseline_3sigma": float(mean_val + 3 * std_val),
    }


def train_model(
    epochs: int = 25,
    batch_size: int = 256,
    model_save_path: str = "models/autoencoder.keras",
    threshold_save_path: str = "models/threshold.json",
    history_save_path: str = "models/history.json",
    dataset_path: str = "dataset/creditcard.csv",
):
    """Full training pipeline for credit card fraud autoencoder."""
    print("=" * 60)
    print("Autoencoder Training for Credit Card Fraud Detection")
    print("=" * 60)

    # 1. Load & Preprocess
    print("\n[Step 1/4] Loading and preprocessing dataset...")
    df = load_data(dataset_path)
    data_dict = preprocess_data(df)
    X_train = data_dict["X_train"]
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"]

    # 2. Build Model
    print("\n[Step 2/4] Building Autoencoder architecture...")
    model = build_autoencoder(input_dim=X_train.shape[1])
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    # 3. Train Model
    print(f"\n[Step 3/4] Training model for up to {epochs} epochs (Batch size={batch_size})...")
    
    # We use a validation split from normal data for early stopping on MSE
    val_split_idx = int(len(X_train) * 0.9)
    X_tr = X_train[:val_split_idx]
    X_val_loss = X_train[val_split_idx:]

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_tr,
        X_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_loss, X_val_loss),
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    # Save trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"\nTrained Autoencoder saved to: {model_save_path}")

    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_save_path, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"Training history saved to: {history_save_path}")

    # 4. Tune Threshold on Validation Set
    print("\n[Step 4/4] Calculating validation errors and tuning threshold...")
    val_errors = compute_reconstruction_errors(model, X_val)
    threshold_info = find_optimal_threshold(val_errors, y_val)
    
    print("\nOptimal Threshold Results on Validation Set:")
    print(f"  Selected Threshold (T): {threshold_info['threshold']:.4f}")
    print(f"  Validation F1-Score:    {threshold_info['f1_score']:.4f}")
    print(f"  Validation Precision:   {threshold_info['precision']:.4f}")
    print(f"  Validation Recall:      {threshold_info['recall']:.4f}")

    with open(threshold_save_path, "w") as f:
        json.dump(threshold_info, f, indent=4)
    print(f"Threshold metadata saved to: {threshold_save_path}")

    return model, threshold_info, history_dict


if __name__ == "__main__":
    train_model(epochs=20, batch_size=256)
