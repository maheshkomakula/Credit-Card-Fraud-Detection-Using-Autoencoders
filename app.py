"""
Streamlit Web Application: Credit Card Fraud Detection Using Autoencoders.
Interactive dashboard for real-time transaction scoring, batch fraud scanning,
and deep learning model evaluation.
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

# Ensure root directory is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import FraudDetector
from src.preprocess import load_data, preprocess_data

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detector | Autoencoder DL Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern dark-mode aesthetic
st.markdown(
    """
    <style>
    /* Global App Aesthetics */
    .main {
        background-color: #0b0f19;
    }
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%);
        color: #f3f4f6;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Header Card */
    .hero-banner {
        background: linear-gradient(90deg, rgba(30, 58, 138, 0.5) 0%, rgba(88, 28, 135, 0.5) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 25px;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-card-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
    }
    .metric-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }

    /* Verdict Boxes */
    .verdict-legit {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border: 1.5px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        animation: fadeIn 0.4s ease-in;
    }
    .verdict-fraud {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%);
        border: 1.5px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        animation: fadeIn 0.4s ease-in;
    }
    .verdict-header {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .verdict-desc {
        font-size: 0.95rem;
        color: #cbd5e1;
    }

    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .badge-legit {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-fraud {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1322;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_detector():
    """Load detector model, scaler, and threshold metadata."""
    detector = FraudDetector()
    return detector


@st.cache_data
def get_sample_dataset():
    """Load dataset cache."""
    return load_data("dataset/creditcard.csv")


@st.cache_data
def get_preprocessed_test():
    """Load preprocessed test set for evaluation explorer."""
    df = get_sample_dataset()
    res = preprocess_data(df)
    return res["X_test"], res["y_test"], res["feature_cols"]


# Load artifacts
try:
    detector = get_detector()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure src/train.py has been run.")
    model_loaded = False

# Hero Banner
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🛡️ Credit Card Fraud Detection</div>
        <div class="hero-subtitle">Deep Learning Anomaly Detection System using Keras Autoencoders & Reconstruction Error Analysis</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.image("https://img.icons8.com/isometric/100/shield.png", width=70)
st.sidebar.title("Configuration")

# Threshold controller
current_thresh = detector.threshold if model_loaded else 8.5
threshold_input = st.sidebar.slider(
    "Decision Threshold (T)",
    min_value=1.0,
    max_value=30.0,
    value=float(current_thresh),
    step=0.25,
    help="Transactions with Reconstruction Error > T are flagged as Suspicious.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Specifications")
st.sidebar.info(
    """
    - **Architecture**: 30 → 16 → 8 → 4 → 8 → 16 → 30
    - **Optimizer**: Adam (lr=0.001)
    - **Loss Function**: MSE
    - **Training Data**: Legitimate (Class=0)
    - **Anomaly Metric**: $E(x) = \\frac{1}{n}\\sum (x_i - \\hat{x}_i)^2$
    """
)

# Navigation Tabs
tabs = st.tabs([
    "🔍 Live Transaction Tester",
    "📂 Batch File Scanner",
])

# -------------------------------------------------------------
# TAB 1: LIVE TRANSACTION TESTER
# -------------------------------------------------------------
with tabs[0]:
    st.markdown("### Test Individual Credit Card Transactions")
    st.caption("Inspect how the Autoencoder reconstructs incoming feature vectors and flags anomalous patterns.")

    df_sample = get_sample_dataset()
    feature_cols = [c for c in df_sample.columns if c != "Class"]

    # Preset transaction buttons
    col_preset1, col_preset2, col_preset3 = st.columns(3)
    preset_choice = None

    with col_preset1:
        if st.button("🟢 Load Verified Legitimate Transaction", use_container_width=True):
            st.session_state["loaded_txn"] = df_sample[df_sample["Class"] == 0].iloc[10][feature_cols].to_dict()
            st.session_state["txn_type"] = "Legitimate (Class 0)"

    with col_preset2:
        if st.button("🔴 Load Verified Fraudulent Transaction", use_container_width=True):
            # Select detected fraud
            fraud_rows = df_sample[df_sample["Class"] == 1]
            st.session_state["loaded_txn"] = fraud_rows.iloc[1][feature_cols].to_dict()
            st.session_state["txn_type"] = "Fraudulent (Class 1)"

    with col_preset3:
        if st.button("🎲 Generate Random Test Transaction", use_container_width=True):
            rand_row = df_sample.sample(1, random_state=int(np.random.randint(0, 10000)))
            is_fraud = rand_row["Class"].values[0]
            st.session_state["loaded_txn"] = rand_row[feature_cols].iloc[0].to_dict()
            st.session_state["txn_type"] = f"Random Sample ({'Fraud' if is_fraud == 1 else 'Legitimate'})"

    # Initialize session state txn if not present
    if "loaded_txn" not in st.session_state:
        st.session_state["loaded_txn"] = df_sample[df_sample["Class"] == 0].iloc[10][feature_cols].to_dict()
        st.session_state["txn_type"] = "Legitimate (Default)"

    st.markdown(f"**Current Input Source:** `{st.session_state.get('txn_type', 'Default')}`")

    # Inputs layout
    with st.expander("📝 View / Edit Transaction Feature Values (30 Features)", expanded=False):
        st.write("Adjust numerical values below or test directly with the loaded preset.")
        user_txn = {}
        c1, c2 = st.columns(2)
        with c1:
            user_txn["Time"] = st.number_input("Time (seconds)", value=float(st.session_state["loaded_txn"]["Time"]))
            user_txn["Amount"] = st.number_input("Amount ($)", value=float(st.session_state["loaded_txn"]["Amount"]))
            for i in range(1, 15):
                col_name = f"V{i}"
                user_txn[col_name] = st.number_input(
                    col_name, value=float(st.session_state["loaded_txn"][col_name]), format="%.5f"
                )
        with c2:
            for i in range(15, 29):
                col_name = f"V{i}"
                user_txn[col_name] = st.number_input(
                    col_name, value=float(st.session_state["loaded_txn"][col_name]), format="%.5f"
                )

    # Score prediction
    if st.button("🚀 Analyze Transaction with Autoencoder", type="primary", use_container_width=True):
        active_txn = user_txn if "Amount" in user_txn else st.session_state["loaded_txn"]
        
        # Override detector threshold with user slider threshold
        detector.threshold = threshold_input
        res = detector.predict_single(active_txn, feature_cols)

        error = res["reconstruction_error"]
        is_suspicious = res["is_suspicious"]
        thresh = res["threshold"]

        st.markdown("---")
        st.markdown("### Detection Result")

        # Result Banner
        if is_suspicious:
            st.markdown(
                f"""
                <div class="verdict-fraud">
                    <div class="verdict-header" style="color: #ef4444;">🚨 SUSPICIOUS TRANSACTION FLAGGED</div>
                    <div class="verdict-desc">
                        Reconstruction error (<b>{error:.4f}</b>) exceeds anomaly threshold (<b>{thresh:.4f}</b>).<br>
                        This transaction deviates significantly from learned normal transaction patterns.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="verdict-legit">
                    <div class="verdict-header" style="color: #10b981;">✅ TRANSACTION CLASSIFIED AS LEGITIMATE</div>
                    <div class="verdict-desc">
                        Reconstruction error (<b>{error:.4f}</b>) is well within normal tolerance (Threshold: <b>{thresh:.4f}</b>).<br>
                        The Autoencoder successfully reconstructed all features with minimal deviation.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        # Metric Columns
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Reconstruction Error E(x)", f"{error:.4f}", delta=f"{error - thresh:+.4f} vs T", delta_color="inverse")
        with m2:
            st.metric("Decision Threshold T", f"{thresh:.4f}")
        with m3:
            ratio = error / thresh if thresh > 0 else 0
            st.metric("Error / Threshold Ratio", f"{ratio:.2f}x")
        with m4:
            st.metric("Classification", res["prediction"])

        # Feature error contribution breakdown chart
        st.markdown("#### Feature Reconstruction Error Breakdown")
        st.caption("Identifies which specific transaction attributes contributed most heavily to the anomaly score.")
        
        feat_errs = res["feature_errors"]
        top_errs = sorted(feat_errs.items(), key=lambda x: x[1], reverse=True)[:10]
        top_df = pd.DataFrame(top_errs, columns=["Feature", "Squared Error"])

        fig, ax = plt.subplots(figsize=(10, 3.8))
        colors = ["#ef4444" if is_suspicious else "#38bdf8"] * len(top_df)
        ax.bar(top_df["Feature"], top_df["Squared Error"], color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Squared Error (x - x̂)²", fontsize=10)
        ax.set_title("Top 10 Feature Reconstruction Discrepancies", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# -------------------------------------------------------------
# TAB 2: BATCH SCANNER
# -------------------------------------------------------------
with tabs[1]:
    st.markdown("### Batch Transaction Fraud Scanner")
    st.caption("Process a batch of credit card transactions to instantly identify all suspicious anomalies.")

    batch_source = st.radio(
        "Choose Data Source:",
        ["Scan 500 Samples from Test Dataset", "Upload Custom CSV File"],
        horizontal=True,
    )

    batch_df = None
    if batch_source == "Upload Custom CSV File":
        uploaded_file = st.file_uploader("Upload CSV file containing transaction features", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df):,} transactions from uploaded file.")
    else:
        df_full = get_sample_dataset()
        # Take a balanced slice containing normal + fraud for an engaging demo
        norm_slice = df_full[df_full["Class"] == 0].sample(480, random_state=42)
        fraud_slice = df_full[df_full["Class"] == 1].sample(20, random_state=42)
        batch_df = pd.concat([norm_slice, fraud_slice]).sample(frac=1, random_state=42).reset_index(drop=True)
        st.info("Loaded 500 test transactions (including 20 real fraudulent test transactions for demonstration).")

    if batch_df is not None:
        if st.button("⚡ Scan Transactions Now", type="primary"):
            with st.spinner("Reconstructing features through Autoencoder..."):
                detector.threshold = threshold_input
                pred_df = detector.predict_batch(batch_df)

                total_scanned = len(pred_df)
                total_flagged = int((pred_df["Prediction"] == "Suspicious").sum())
                total_legit = total_scanned - total_flagged
                anomaly_pct = (total_flagged / total_scanned) * 100

                # Summary cards
                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    st.metric("Total Scanned", f"{total_scanned:,}")
                with b2:
                    st.metric("Legitimate", f"{total_legit:,}")
                with b3:
                    st.metric("Suspicious Flagged", f"{total_flagged:,}", delta=f"{anomaly_pct:.1f}% rate", delta_color="inverse")
                with b4:
                    if "True_Class" in pred_df.columns:
                        caught_frauds = int(((pred_df["Prediction"] == "Suspicious") & (pred_df["True_Class"] == 1)).sum())
                        total_frauds = int(pred_df["True_Class"].sum())
                        st.metric("Actual Frauds Caught", f"{caught_frauds}/{total_frauds}")

                st.markdown("#### Scanned Results Table")
                filter_choice = st.selectbox("Filter Display:", ["Show All", "Only Suspicious Alerts", "Only Legitimate"])

                display_df = pred_df.copy()
                if filter_choice == "Only Suspicious Alerts":
                    display_df = display_df[display_df["Prediction"] == "Suspicious"]
                elif filter_choice == "Only Legitimate":
                    display_df = display_df[display_df["Prediction"] == "Legitimate"]

                st.dataframe(display_df.head(200), use_container_width=True)

                # Export flagged CSV
                csv_data = display_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Flagged Report as CSV",
                    data=csv_data,
                    file_name="flagged_fraud_report.csv",
                    mime="text/csv",
                )




