"""
Model Evaluation and Visualization Module for Credit Card Fraud Detection.
Evaluates Autoencoder reconstruction performance on the test set,
generates metrics (Precision, Recall, F1, ROC-AUC), and produces publication-quality charts.
"""

import json
import os
import sys

# Ensure root folder is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.predict import FraudDetector
from src.preprocess import load_data, preprocess_data


def plot_training_history(history_path: str = "models/history.json", save_path: str = "reports/training_loss.png"):
    """Plot training and validation loss over epochs."""
    if not os.path.exists(history_path):
        print(f"History file not found at {history_path}. Skipping loss plot.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    plt.figure(figsize=(9, 5))
    epochs = range(1, len(history["loss"]) + 1)
    plt.plot(epochs, history["loss"], label="Training Loss (MSE)", color="#2563eb", lw=2)
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Validation Loss (MSE)", color="#ef4444", lw=2, linestyle="--")

    plt.title("Autoencoder Training & Validation Loss (MSE)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Reconstruction Error (MSE)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training loss plot to: {save_path}")


def plot_error_distribution(
    errors: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    save_path: str = "reports/reconstruction_error_distribution.png",
):
    """Plot reconstruction error distribution for Legitimate vs Fraudulent transactions."""
    normal_errors = errors[y_true == 0]
    fraud_errors = errors[y_true == 1]

    plt.figure(figsize=(10, 6))

    # Log-scale histogram/KDE for visual clarity given heavy tails
    bins = np.linspace(0, np.percentile(errors, 99.8) * 1.5, 60)
    plt.hist(
        normal_errors,
        bins=bins,
        alpha=0.65,
        color="#10b981",
        label=f"Legitimate Transactions (n={len(normal_errors):,})",
        density=True,
    )
    plt.hist(
        fraud_errors,
        bins=bins,
        alpha=0.65,
        color="#ef4444",
        label=f"Fraudulent Transactions (n={len(fraud_errors):,})",
        density=True,
    )

    plt.axvline(
        threshold,
        color="#1e1e24",
        linestyle="--",
        linewidth=2.2,
        label=f"Decision Threshold T = {threshold:.4f}",
    )

    plt.title("Reconstruction Error Distribution (Legitimate vs Fraud)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Reconstruction Error (MSE)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved reconstruction error distribution plot to: {save_path}")


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    save_path: str = "reports/confusion_matrix.png",
):
    """Plot Confusion Matrix Heatmap."""
    plt.figure(figsize=(7, 6))
    labels = ["Legitimate (0)", "Fraudulent (1)"]
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title("Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap to: {save_path}")


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    errors: np.ndarray,
    save_path: str = "reports/roc_pr_curves.png",
):
    """Plot ROC Curve and Precision-Recall Curve side-by-side."""
    fpr, tpr, _ = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, errors)
    pr_auc = average_precision_score(y_true, errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ROC Curve
    axes[0].plot(fpr, tpr, color="#2563eb", lw=2.2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", lw=1.5, label="Random Guess")
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # PR Curve
    axes[1].plot(recall, precision, color="#10b981", lw=2.2, label=f"PR Curve (Avg Prec = {pr_auc:.4f})")
    no_skill = sum(y_true) / len(y_true)
    axes[1].plot([0, 1], [no_skill, no_skill], color="#9ca3af", linestyle="--", lw=1.5, label=f"No Skill Baseline ({no_skill:.4f})")
    axes[1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC & PR curves to: {save_path}")


def evaluate_system(
    dataset_path: str = "dataset/creditcard.csv",
    save_reports: bool = True,
):
    """Full evaluation of the Autoencoder fraud detection system on held-out test data."""
    print("=" * 60)
    print("Evaluating Autoencoder Fraud Detection System on Test Set")
    print("=" * 60)

    # 1. Load Data
    df = load_data(dataset_path)
    processed = preprocess_data(df)
    X_test = processed["X_test"]
    y_test = processed["y_test"]

    # 2. Detector
    detector = FraudDetector()
    threshold = detector.threshold

    # 3. Compute test reconstruction errors
    reconstructions = detector.model.predict(X_test, batch_size=1024, verbose=0)
    test_errors = np.mean(np.square(X_test - reconstructions), axis=1)

    # Predictions
    y_pred = (test_errors > threshold).astype(int)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, test_errors))
    pr_auc = float(average_precision_score(y_test, test_errors))

    # Test set summary
    total_test = len(y_test)
    total_fraud = int(sum(y_test))
    total_normal = total_test - total_fraud

    metrics_summary = {
        "test_total_samples": total_test,
        "test_normal_samples": total_normal,
        "test_fraud_samples": total_fraud,
        "selected_threshold": threshold,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "fraud_detection_rate": float(tp / total_fraud) if total_fraud > 0 else 0.0,
        "false_alarm_rate": float(fp / total_normal) if total_normal > 0 else 0.0,
    }

    # Print Report
    print("\n[PERFORMANCE METRICS ON HELD-OUT TEST DATA]")
    print(f"Total Test Transactions:  {total_test:,}")
    print(f"Actual Legitimate:        {total_normal:,}")
    print(f"Actual Fraudulent:        {total_fraud:,}")
    print("-" * 50)
    print(f"Decision Threshold (T):   {threshold:.5f}")
    print(f"Precision:                {prec:.4f}  ({tp} detected frauds were true frauds out of {tp + fp} alerts)")
    print(f"Recall (Sensitivity):     {rec:.4f}  (Successfully caught {tp} out of {total_fraud} total frauds)")
    print(f"F1-Score:                 {f1:.4f}")
    print(f"ROC-AUC:                  {roc_auc:.4f}")
    print(f"PR-AUC (Avg Precision):   {pr_auc:.4f}")
    print("-" * 50)
    print("Confusion Matrix:")
    print(f"  TN (Correct Legitimate): {tn:,}  |  FP (False Alarms): {fp:,}")
    print(f"  FN (Missed Frauds):      {fn:,}  |  TP (Caught Frauds): {tp:,}")
    print("=" * 60)

    if save_reports:
        os.makedirs("reports", exist_ok=True)
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics_summary, f, indent=4)
        print("Metrics saved to: reports/metrics.json")

        plot_training_history()
        plot_error_distribution(test_errors, y_test, threshold)
        plot_confusion_matrix_heatmap(cm)
        plot_roc_and_pr_curves(y_test, test_errors)

    return metrics_summary


if __name__ == "__main__":
    evaluate_system()
