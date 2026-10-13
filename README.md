# 🛡️ Credit Card Fraud Detection Using Autoencoders

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.12-red.svg)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Deep Learning (DL) laboratory and production-ready system for identifying suspicious credit card transactions using **Autoencoders** and **Reconstruction Error Analysis**.

---

## 1. Project Abstract
Credit card fraud is a major problem in digital financial transactions. Traditional fraud detection methods often struggle to identify new or unusual fraud patterns due to extreme class imbalance and fast-evolving fraud techniques. 

This project uses an **Autoencoder** neural network to detect fraudulent transactions by learning the normal behavioral patterns of legitimate transactions. The model is trained primarily on legitimate transactions (Class = 0) and learns to compress and reconstruct their 30 numerical input features. When an unusual transaction is provided, the autoencoder produces a significantly higher reconstruction error. This Mean Squared Error (MSE) is used as an anomaly score to classify transactions as **Legitimate** or **Suspicious** based on an optimal data-driven threshold selected on validation data.

---

## 2. Key Objectives
- **Anomaly Detection via Deep Learning:** Detect potentially fraudulent transactions in an unsupervised/semi-supervised paradigm.
- **Autoencoder Manifold Learning:** Understand how neural network bottlenecks compress normal transaction patterns and reconstruct them.
- **Reconstruction Error as Anomaly Score:** Measure feature-level deviations using Mean Squared Error (MSE).
- **Extreme Class Imbalance Mitigation:** Train exclusively on normal transactions (avoiding the 99.8% dummy accuracy trap).
- **Rigorous Model Evaluation:** Evaluate performance with Precision, Recall, F1-Score, ROC-AUC (0.9519), and PR-AUC.
- **Interactive User Interface:** Provide a Streamlit dashboard with real-time transaction scoring, sensitivity exploration, and batch CSV scanning.

---

## 3. Technologies Required

| Component | Technology | Version |
| :--- | :--- | :--- |
| **Programming Language** | Python | 3.10+ |
| **Deep Learning Framework** | TensorFlow / Keras | 2.21+ / 3.12+ |
| **Data Processing** | Pandas, NumPy | 2.3+ / 2.2+ |
| **Data Visualization** | Matplotlib, Seaborn | 3.10+ / 0.13+ |
| **Machine Learning Metrics** | Scikit-learn | 1.7+ |
| **Interactive User Interface** | Streamlit | 1.52+ |
| **Serialization** | Joblib | 1.5+ |
| **Dataset** | Kaggle European Card Fraud | 284,807 transactions |

---

## 4. Dataset Overview
We use the **Credit Card Fraud Detection dataset from Kaggle** (`dataset/creditcard.csv`).
- **Total Transactions:** 284,807
- **Legitimate Transactions:** 284,315 (99.83%)
- **Fraudulent Transactions:** 492 (0.17%)

| Feature | Description |
| :--- | :--- |
| **Time** | Seconds elapsed between the transaction and the first transaction in the dataset |
| **V1 to V28** | Principal Component Analysis (PCA) transformed numerical features |
| **Amount** | Transaction amount in currency |
| **Class** | Target label: `0` for Legitimate, `1` for Fraudulent |

---

## 5. Deep Learning Model Architecture: Autoencoder

The Autoencoder comprises an **Encoder**, a **Latent Bottleneck**, and a **Decoder**:

```
[Input: 30 Features]
        │
   ▼ (ReLU)
[Dense: 16 Units]  ── Encoder Layer 1
        │
   ▼ (ReLU)
[Dense: 8 Units]   ── Encoder Layer 2
        │
   ▼ (ReLU)
[Dense: 4 Units]   ── Latent Space (Bottleneck Representation)
        │
   ▼ (ReLU)
[Dense: 8 Units]   ── Decoder Layer 1
        │
   ▼ (ReLU)
[Dense: 16 Units]  ── Decoder Layer 2
        │
   ▼ (Linear)
[Output: 30 Units] ── Reconstructed Features (x̂)
```

### Mathematical Formulation
**1. Reconstruction Error (MSE):**
$$E(x) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{x}_i)^2$$
Where:
- $x_i$ is the scaled input feature
- $\hat{x}_i$ is the reconstructed output feature
- $n = 30$ is the feature dimension

**2. Anomaly Classification Rule:**
$$\text{Prediction} = \begin{cases} \text{Suspicious}, & E(x) > T \\ \text{Legitimate}, & E(x) \le T \end{cases}$$

The threshold $T \approx 8.509$ was determined on validation data to maximize the **F1-Score**, striking an optimal balance between catching fraud (Recall) and minimizing false alarms (Precision).

---

## 6. Directory Structure

```
DL-Project/
│
├── dataset/
│   └── creditcard.csv                 # Kaggle Credit Card dataset (284,807 rows)
│
├── models/
│   ├── autoencoder.keras              # Saved trained Keras Autoencoder model
│   ├── scaler.pkl                     # Fitted StandardScaler on normal transactions
│   ├── threshold.json                 # Optimal decision threshold & validation stats
│   └── history.json                   # Epoch training loss metrics
│
├── notebooks/
│   └── fraud_detection.ipynb          # End-to-end Jupyter Notebook walkthrough
│
├── reports/
│   ├── training_loss.png              # Loss convergence curve
│   ├── reconstruction_error_distribution.png # Normal vs Fraud error histogram
│   ├── confusion_matrix.png           # Test confusion matrix heatmap
│   ├── roc_pr_curves.png              # ROC & Precision-Recall curves
│   └── metrics.json                   # Comprehensive test metrics
│
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── preprocess.py                  # Data loading, scaling & splits
│   ├── train.py                       # Autoencoder training & threshold tuning
│   ├── predict.py                     # Single transaction & batch inference
│   └── evaluate.py                    # Evaluation metrics & visualization generation
│
├── app.py                             # Interactive Streamlit Web Application
├── requirements.txt                   # Project Python dependencies
└── README.md                          # Comprehensive project documentation
```

---

## 7. Installation & Quickstart

### Step 1: Clone and Set Up Virtual Environment
```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
cd "Credit-Card-Fraud-Detection"

python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate  # On Linux/macOS
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Data Preprocessing
```bash
python src/preprocess.py
```

### Step 4: Train the Autoencoder Model
```bash
python src/train.py
```

### Step 5: Run Prediction Demo (Matching Expected Output)
```bash
python src/predict.py
```
**Output:**
```
============================================================
Credit Card Fraud Autoencoder - Live Inference Demo
Current Decision Threshold (T): 8.5090
============================================================

Transaction A
Reconstruction error: 0.1908
Classification:       Legitimate

Transaction B
Reconstruction error: 22.0507
Classification:       Suspicious
============================================================
```

### Step 6: Evaluate on Held-Out Test Set
```bash
python src/evaluate.py
```

### Step 7: Launch Interactive Streamlit Dashboard
```bash
python -m streamlit run app.py
# or if streamlit is in your PATH:
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 8. Experimental Results & Performance Metrics

Evaluated on **28,678 held-out test transactions** (28,432 legitimate, 246 fraud cases):

| Metric | Score | Analysis |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9519** | Outstanding separation between normal and fraud distributions |
| **PR-AUC (Avg Precision)** | **0.5672** | High precision retention under extreme 0.17% class imbalance |
| **F1-Score** | **0.6154** | Balanced harmonic mean of precision and recall |
| **Precision** | **0.6486** | 144 true frauds out of 222 alerts (minimal false alarm overhead) |
| **Recall (Sensitivity)** | **0.5854** | Successfully caught 144 out of 246 actual frauds |
| **False Positive Rate** | **0.27%** | Only 78 false alarms out of 28,432 legitimate transactions |

### Test Confusion Matrix

| Actual \ Predicted | Legitimate (Predicted 0) | Suspicious (Predicted 1) |
| :--- | :---: | :---: |
| **Legitimate (Class 0)** | **28,354** (True Negative) | **78** (False Positive) |
| **Fraudulent (Class 1)** | **102** (False Negative) | **144** (True Positive) |

---

## 9. Key Visualizations

- **Training vs. Validation Loss:** Demonstrates stable convergence without overfitting on normal transaction manifold.
- **Reconstruction Error Distribution:** Shows strong separation between Legitimate ($E(x) \le 1.0$) and Fraudulent ($E(x) > 8.5$) transactions.
- **ROC and PR Curves:** Confirms high discriminative power with 0.9519 Area Under ROC Curve.

---

## 10. Deep Learning Lab Viva / Interview Q&A

**Q1: Why use an Autoencoder rather than a standard Supervised Classifier (Random Forest/XGBoost)?**  
*Answer:* Supervised classifiers rely on labeled fraud examples. Fraud patterns constantly evolve ("zero-day" attack patterns). An autoencoder models *what legitimate behavior looks like*. Any behavior deviating from normal regularities naturally fails reconstruction, allowing the model to flag novel, unseen fraud patterns.

**Q2: Why is Accuracy misleading for this dataset?**  
*Answer:* Because 99.83% of transactions are legitimate, a model that classifies everything as legitimate achieves 99.83% accuracy while catching zero frauds. Precision, Recall, F1, and PR-AUC reflect true clinical utility.

**Q3: Why is the latent dimension compressed to 4 units?**  
*Answer:* The bottleneck forces dimensionality reduction. If the latent dimension were equal to 30, the network could trivially memorize the identity mapping without uncovering the underlying correlational manifold of normal banking transactions.

---

## 11. Authors & Academic Acknowledgement
Developed for the **Deep Learning (DL) Laboratory Project**: *Credit Card Fraud Detection Using Autoencoders*.  
Dataset credits: Machine Learning Group - Université Libre de Bruxelles (ULB), Kaggle.
