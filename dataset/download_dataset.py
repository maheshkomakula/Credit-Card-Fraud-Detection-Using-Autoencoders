"""
Utility script to download the Kaggle Credit Card Fraud dataset if not present.
"""

import os
import shutil

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_PATH = os.path.join(DATASET_DIR, "creditcard.csv")


def ensure_dataset():
    if os.path.exists(DEST_PATH) and os.path.getsize(DEST_PATH) > 1000000:
        print(f"[OK] Dataset already present at: {DEST_PATH}")
        return DEST_PATH

    print("Downloading creditcard.csv from public repository mirror...")
    try:
        from huggingface_hub import hf_hub_download
        cached_path = hf_hub_download(
            repo_id="David-Egea/Creditcard-fraud-detection",
            filename="creditcard.csv",
            repo_type="dataset",
        )
        shutil.copy(cached_path, DEST_PATH)
        print(f"[SUCCESS] Downloaded and copied to: {DEST_PATH} ({os.path.getsize(DEST_PATH):,} bytes)")
        return DEST_PATH
    except Exception as e:
        print(f"Automatic download failed: {e}")
        print("Please download creditcard.csv manually from Kaggle:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"and place it inside: {DATASET_DIR}")
        return None


if __name__ == "__main__":
    ensure_dataset()
