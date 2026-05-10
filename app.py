"""
============================================================
  Credit Card Fraud Detection – Streamlit Web Application
  Run with: streamlit run app.py
============================================================
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config (must be the very first Streamlit call) ─────
st.set_page_config(
    page_title = "Credit Card Fraud Detector",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ──────────────────────────────────────────────────────────
# PATHS  –  always relative to THIS script's folder,
#           no matter where you launch streamlit from.
# ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
DATA_PATH   = os.path.join(BASE_DIR, "data", "creditcard.csv")

# ──────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ─────────────────── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
}
/* ── Headings ───────────────── */
h1, h2, h3 {color: #ffffff !important;}
/* ── Metric cards ───────────── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 12px 18px;
}
/* ── Input widgets ──────────── */
.stNumberInput input, .stSlider {
    background: rgba(255,255,255,0.08) !important;
}
/* ── Result banners ─────────── */
.fraud-banner {
    background: linear-gradient(90deg,#c0392b,#e74c3c);
    border-radius: 14px;
    padding: 28px 36px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 1px;
    box-shadow: 0 8px 30px rgba(231,76,60,0.45);
    animation: pulse 1.5s infinite;
}
.safe-banner {
    background: linear-gradient(90deg,#27ae60,#2ecc71);
    border-radius: 14px;
    padding: 28px 36px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 1px;
    box-shadow: 0 8px 30px rgba(46,204,113,0.45);
}
@keyframes pulse {
    0%   {box-shadow: 0 8px 30px rgba(231,76,60,0.45);}
    50%  {box-shadow: 0 8px 50px rgba(231,76,60,0.85);}
    100% {box-shadow: 0 8px 30px rgba(231,76,60,0.45);}
}
/* ── Button ─────────────────── */
.stButton>button {
    background: linear-gradient(90deg,#6a11cb,#2575fc) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    padding: 14px 40px !important;
    width: 100%;
    transition: transform 0.15s;
}
.stButton>button:hover {transform: scale(1.03);}
/* ── Progress bar ───────────── */
.stProgress > div > div {background-color: #e74c3c !important;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model and scaler (cached so they load only once)."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data(show_spinner=False)
def load_sample_data():
    """Load a small sample of the dataset for the dashboard."""
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH, nrows=10_000)


def predict(input_values: list, model, scaler) -> dict:
    """
    Run inference on a single transaction.

    input_values : [Time, V1…V28, Amount]  –  30 raw floats
    """
    cols   = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df_row = pd.DataFrame([input_values], columns=cols)

    # Scale Time & Amount (same transform as training)
    df_row[["Time", "Amount"]] = scaler.transform(df_row[["Time", "Amount"]])

    pred  = model.predict(df_row)[0]
    proba = model.predict_proba(df_row)[0]   # [P(0), P(1)]

    return {
        "label"      : "Fraud" if pred == 1 else "Not Fraud",
        "fraud_prob" : float(proba[1]),
        "safe_prob"  : float(proba[0]),
    }


def plot_gauge(fraud_prob: float):
    """Render a simple colour-coded probability gauge."""
    fig, ax = plt.subplots(figsize=(5, 0.55))
    color = "#e74c3c" if fraud_prob > 0.5 else "#2ecc71"
    ax.barh([0], [fraud_prob],    color=color,   height=0.5, zorder=3)
    ax.barh([0], [1-fraud_prob],  color="#2c2c3e", height=0.5,
            left=fraud_prob, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%","25%","50%","75%","100%"], color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    return fig


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Navigation")
    page = st.radio(
        "",
        ["🏠 Dashboard", "🔮 Predict Transaction", "📁 Bulk CSV Analysis", "📊 Model Reports", "📖 About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    st.markdown("""
    - **Algorithm** : Random Forest  
    - **Balancing** : SMOTE  
    - **Scaler**    : StandardScaler (Time & Amount)  
    - **Features**  : Time, V1–V28, Amount  
    """)
    st.markdown("---")
    st.caption("Built with ❤️ using scikit-learn + Streamlit")


# ──────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ──────────────────────────────────────────────────────────
model, scaler = load_artifacts()
model_ready   = model is not None

if not model_ready:
    st.warning(
        "⚠️ Trained model not found. Please run **`python train.py`** first.",
        icon="⚠️",
    )


# ══════════════════════════════════════════════════════════
# PAGE 1 – DASHBOARD
# ══════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown("##### Real-time ML-powered fraud analysis with Random Forest")
    st.markdown("---")

    # ── High-level KPI cards ──────────────────────────────
    df_sample = load_sample_data()
    if df_sample is not None:
        total   = len(df_sample)
        fraud   = df_sample["Class"].sum()
        n_fraud = total - fraud
        rate    = fraud / total * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions",  f"{total:,}")
        c2.metric("Fraud Cases",         f"{int(fraud):,}", delta=f"{rate:.2f}% of total", delta_color="inverse")
        c3.metric("Legitimate Cases",    f"{int(n_fraud):,}")
        c4.metric("Fraud Rate",          f"{rate:.3f}%")

        st.markdown("---")

        # ── Plots row ─────────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📊 Class Distribution")
            plot_path = os.path.join(PLOTS_DIR, "class_distribution.png")
            if os.path.exists(plot_path):
                st.image(plot_path, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(5,3))
                ax.bar(["Not Fraud","Fraud"],[n_fraud,fraud],color=["#2ecc71","#e74c3c"])
                ax.set_facecolor("#0f0c29"); fig.patch.set_facecolor("#0f0c29")
                ax.tick_params(colors="white"); ax.spines[:].set_color("gray")
                for lbl in ax.get_xticklabels()+ax.get_yticklabels():
                    lbl.set_color("white")
                st.pyplot(fig, use_container_width=True)

        with col_right:
            st.subheader("🔥 Amount Distribution")
            plot_path = os.path.join(PLOTS_DIR, "amount_distribution.png")
            if os.path.exists(plot_path):
                st.image(plot_path, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(5,3))
                ax.hist(df_sample[df_sample.Class==0].Amount, bins=50,
                        color="#2ecc71", alpha=0.7, label="Not Fraud", log=True)
                ax.hist(df_sample[df_sample.Class==1].Amount, bins=50,
                        color="#e74c3c", alpha=0.9, label="Fraud", log=True)
                ax.legend(); ax.set_xlabel("Amount"); ax.set_ylabel("Count (log)")
                ax.set_facecolor("#0f0c29"); fig.patch.set_facecolor("#0f0c29")
                st.pyplot(fig, use_container_width=True)

    else:
        st.info("📁 Place **creditcard.csv** in the `./data/` folder to see dataset statistics.")


# ══════════════════════════════════════════════════════════
# PAGE 2 – PREDICT TRANSACTION
# ══════════════════════════════════════════════════════════
elif "Predict" in page:
    st.title("🔮 Real-Time Transaction Prediction")
    st.markdown("Enter the transaction features below and click **Check Transaction**.")
    st.markdown("---")

    # ── Example presets ───────────────────────────────────
    st.markdown("#### 🧪 Quick-load example transactions")
    eg_col1, eg_col2, eg_col3 = st.columns(3)

    # Examples verified against the trained model – guaranteed correct predictions
    EXAMPLES = {
        "Normal ✅": dict(
            time=47300, amount=9.21,
            v=[ 1.0528, -1.7774, -0.6547,  2.0709, -0.0151, -0.2187,
                1.1824, -1.0322, -0.9546, -1.4096,  0.6632, -0.3976,
                0.1324, -1.6407,  0.7592, -0.4432, -1.8859,  0.8475,
                0.5869,  1.0287,  1.5856, -0.6557, -0.5050,  0.7724,
                0.8304, -0.0701, -0.2691, -0.7938],
        ),
        "Fraud ⚠️": dict(
            time=472, amount=529.0,
            v=[-3.0435, -3.1573,  1.0885,  2.2886,  1.3598, -1.0648,
                0.3256, -0.0678, -0.2710, -0.8386, -0.4146, -0.5031,
                0.6765, -1.6920,  2.0006,  0.6668,  0.5997,  1.7253,
                0.2833,  2.1023,  0.6617,  0.4355,  1.3760, -0.2938,
                0.2798, -0.1454, -0.2528,  0.0358],
        ),
        "Suspicious 🟡": dict(
            time=129186, amount=37.26,
            v=[-2.5346,  6.9470, -1.1006, -0.5301, -0.1365, -1.0065,
               -5.9766, -0.5857, -4.0388, -5.7026,  2.3425, -5.8099,
                0.7207, -7.6307, -2.3047, -1.9131, -6.6564, -0.1923,
               -0.5569,  0.7673,  2.4819,  0.8119, -2.1257, -1.9378,
               -0.5586, -0.3755, -2.4107,  1.1471],
        ),
    }

    # ── When a load button is clicked: push values into session_state ──
    # Call st.rerun() after setting values to force a full re-render
    # so every number_input actually shows the new values.
    def load_example(name):
        ex = EXAMPLES[name]
        st.session_state["inp_time"]   = float(ex["time"])
        st.session_state["inp_amount"] = float(ex["amount"])
        for i in range(1, 29):
            st.session_state[f"inp_v{i}"] = float(ex["v"][i - 1])
        st.session_state["loaded_example"] = name
        st.rerun()   # ← forces full re-render with new values

    if eg_col1.button("Load Normal ✅",     use_container_width=True): load_example("Normal ✅")
    if eg_col2.button("Load Fraud ⚠️",      use_container_width=True): load_example("Fraud ⚠️")
    if eg_col3.button("Load Suspicious 🟡", use_container_width=True): load_example("Suspicious 🟡")

    # Show which example is loaded
    loaded = st.session_state.get("loaded_example", "")
    if loaded:
        color = "#e74c3c" if "Fraud" in loaded else ("#f39c12" if "Suspicious" in loaded else "#2ecc71")
        st.markdown(
            f"<div style='background:{color}22;border-left:4px solid {color};"
            f"padding:8px 14px;border-radius:6px;margin:6px 0;font-size:14px;'>"
            f"📌 Loaded: <b>{loaded}</b></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### ✏️ Transaction Details")

    # ── Inputs – use session_state keys so load buttons update them ──
    col_t, col_a = st.columns(2)
    time_val   = col_t.number_input(
        "⏱️ Time (seconds from first transaction)",
        min_value=0.0, max_value=200000.0, step=1.0,
        key="inp_time",
    )
    amount_val = col_a.number_input(
        "💰 Amount (USD)",
        min_value=0.0, max_value=30000.0, step=0.01, format="%.2f",
        key="inp_amount",
    )

    st.markdown("##### 🧬 PCA-Transformed Features (V1 – V28)")
    st.caption("These are the anonymised PCA features from the Kaggle dataset.")

    v_cols = st.columns(4)
    v_vals = []
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        val = v_cols[col_idx].number_input(
            f"V{i}", format="%.4f", key=f"inp_v{i}",
        )
        v_vals.append(val)

    st.markdown("---")
    if st.button("🔍 Check Transaction", use_container_width=True, type="primary"):
        if not model_ready:
            st.error("Model not loaded. Run `python train.py` first.")
        else:
            # Assemble: [Time, V1…V28, Amount]
            input_list = [time_val] + v_vals + [amount_val]

            with st.spinner("Analysing transaction …"):
                time.sleep(0.4)
                result = predict(input_list, model, scaler)

            st.markdown("---")
            st.markdown("### 📋 Result")

            is_fraud = result["label"] == "Fraud"

            if is_fraud:
                st.markdown(
                    '<div class="fraud-banner">⚠️ FRAUD DETECTED</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="safe-banner">✅ SAFE TRANSACTION</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability gauge
            r1, r2, r3 = st.columns([1, 3, 1])
            with r2:
                st.markdown("**Fraud Probability**")
                st.pyplot(plot_gauge(result["fraud_prob"]), use_container_width=True)

            # Metric cards
            m1, m2, m3 = st.columns(3)
            m1.metric("Fraud Probability",    f"{result['fraud_prob']*100:.2f}%")
            m2.metric("Safe Probability",     f"{result['safe_prob']*100:.2f}%")
            m3.metric("Decision",             result["label"],
                      delta="⚠️ Alert" if is_fraud else "✅ Clear",
                      delta_color="inverse" if is_fraud else "normal")

            # Risk meter
            st.progress(result["fraud_prob"])

            if is_fraud:
                st.error("🚨 **Action Required**: This transaction has been flagged for review.")
            else:
                st.success("🎉 **All Clear**: This transaction appears legitimate.")


# ══════════════════════════════════════════════════════════
# PAGE 3 – MODEL REPORTS
# ══════════════════════════════════════════════════════════
elif "Reports" in page:
    st.title("📊 Model Evaluation Reports")
    st.markdown("---")

    plot_map = {
        "confusion_matrix.png"   : "🔲 Confusion Matrix",
        "roc_auc_curve.png"      : "📈 ROC-AUC Curve",
        "feature_importance.png" : "🏆 Feature Importance",
        "class_distribution.png" : "📊 Class Distribution",
    }

    found = [f for f in plot_map if os.path.exists(os.path.join(PLOTS_DIR, f))]

    if not found:
        st.info("Run **`python train.py`** to generate evaluation plots.")
    else:
        for i in range(0, len(found), 2):
            cols = st.columns(2)
            for j, fname in enumerate(found[i:i+2]):
                with cols[j]:
                    st.subheader(plot_map[fname])
                    st.image(os.path.join(PLOTS_DIR, fname), use_container_width=True)
            st.markdown("---")


# ══════════════════════════════════════════════════════════
# PAGE 4 – ABOUT
# ══════════════════════════════════════════════════════════
elif "About" in page:
    st.title("📖 About This Project")
    st.markdown("---")
    st.markdown("""
## Credit Card Fraud Detection System

This application demonstrates an **end-to-end Machine Learning pipeline**  
for detecting credit card fraud in real time.

---

### 🗂️ Project Structure
```
fraud_detection/
├── data/
│   └── creditcard.csv          ← Download from Kaggle
├── model/
│   ├── model.pkl               ← Trained Random Forest
│   └── scaler.pkl              ← Fitted StandardScaler
├── plots/                      ← EDA & evaluation plots
├── train.py                    ← Training pipeline
├── app.py                      ← Streamlit application
└── requirements.txt
```

---

### 🔧 Pipeline Steps
| Step | Details |
|------|---------|
| **Data Loading** | 284,807 transactions (Kaggle dataset) |
| **EDA** | Class distribution, correlation heatmap, feature comparison |
| **Preprocessing** | StandardScaler on Time & Amount; V1–V28 already PCA-scaled |
| **Imbalance Handling** | SMOTE (Synthetic Minority Oversampling) |
| **Model** | Random Forest (100 trees, depth 20, sqrt features) |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| **Persistence** | `joblib` → model.pkl + scaler.pkl |
| **UI** | Streamlit with real-time single transaction prediction |

---

### 📦 Dataset
- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows**: 284,807 transactions
- **Fraud Rate**: 0.172%
- **Features**: Time, V1–V28 (PCA), Amount, Class

---

### 🚀 Getting Started
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset and place in ./data/creditcard.csv

# 3. Train the model
python train.py

# 4. Launch the app
streamlit run app.py
```
    """)



# ══════════════════════════════════════════════════════════
# PAGE 3 – BULK CSV ANALYSIS  (new feature)
# ══════════════════════════════════════════════════════════
elif "Bulk CSV" in page:
    st.title("📁 Bulk CSV Transaction Analysis")
    st.markdown("Upload a CSV file containing multiple transactions and the model will predict **Fraud / Not Fraud** for every row.")
    st.markdown("---")

    # ── Format guide ──────────────────────────────────────
    with st.expander("📋 Required CSV Format  (click to expand)", expanded=False):
        st.markdown("""
Your CSV must have these **30 columns** in any order:

| Column | Description |
|--------|-------------|
| `Time` | Seconds elapsed from the first transaction |
| `V1` … `V28` | 28 anonymised PCA features |
| `Amount` | Transaction amount in USD |

**Optional:** Include a `Class` column (0/1) and the app will compare predictions against ground-truth labels.

A sample row:
```
Time,V1,V2,V3,...,V28,Amount
0,-1.3598,-0.0728,2.5363,...,-0.0211,149.62
```
        """)
        # Download a sample template
        sample_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        sample_df   = pd.DataFrame([[0.0]*30], columns=sample_cols)
        st.download_button(
            "⬇️ Download blank template",
            data     = sample_df.to_csv(index=False),
            file_name= "transaction_template.csv",
            mime     = "text/csv",
        )

    st.markdown("---")

    # ── File uploader ─────────────────────────────────────
    uploaded = st.file_uploader(
        "📂 Upload your transactions CSV",
        type   = ["csv"],
        help   = "Max 200 MB. Must contain Time, V1–V28, Amount columns.",
    )

    if uploaded is not None:
        # ── Force-reload model fresh (bypasses Streamlit cache bug) ──
        if not os.path.exists(MODEL_PATH):
            st.error("❌ Model not found. Run `python train.py` first.")
            st.stop()

        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)

        try:
            df_up = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")
            st.stop()

        # ── Validate columns ──────────────────────────────
        required_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        missing_cols  = [c for c in required_cols if c not in df_up.columns]

        if missing_cols:
            st.error(f"❌ Missing columns: `{', '.join(missing_cols)}`")
            st.info("Your CSV must include: Time, V1–V28, Amount")
            st.stop()

        has_labels = "Class" in df_up.columns

        # ── Threshold slider ──────────────────────────────
        st.markdown("#### ⚙️ Detection Sensitivity")
        threshold = st.slider(
            "Fraud Probability Threshold  "
            "(lower = catch more fraud, higher = fewer false alarms)",
            min_value = 0.10,
            max_value = 0.90,
            value     = 0.30,          # 0.30 is more sensitive than default 0.50
            step      = 0.05,
            format    = "%.2f",
        )
        st.caption(
            f"Any transaction with fraud probability **≥ {threshold*100:.0f}%** "
            "will be flagged as 🚨 Fraud."
        )

        # ── Run predictions ───────────────────────────────
        with st.spinner(f"🔍 Analysing {len(df_up):,} transactions …"):
            X_bulk = df_up[required_cols].copy()
            X_bulk[["Time", "Amount"]] = _scaler.transform(X_bulk[["Time", "Amount"]])

            probas = _model.predict_proba(X_bulk)[:, 1]   # raw probabilities
            preds  = (probas >= threshold).astype(int)     # apply custom threshold

        # ── Debug info box (helps confirm model is working) ──
        raw_preds_default = _model.predict(X_bulk)
        with st.expander("🔬 Model Debug Info", expanded=False):
            st.write(f"**Model file:** `{MODEL_PATH}`")
            st.write(f"**Scaler file:** `{SCALER_PATH}`")
            st.write(f"**Threshold used:** `{threshold}`")
            st.write(f"**Default threshold (0.50) would flag:** `{int(raw_preds_default.sum())}` fraud")
            st.write(f"**Your threshold ({threshold}) flags:** `{int(preds.sum())}` fraud")
            st.write(f"**Max fraud probability in file:** `{probas.max()*100:.2f}%`")
            st.write(f"**Min fraud probability in file:** `{probas.min()*100:.2f}%`")
            st.write(f"**Mean fraud probability:** `{probas.mean()*100:.2f}%`")
            prob_series = pd.Series(probas * 100).round(1)
            st.write("**Probability distribution (all rows):**")
            st.dataframe(prob_series.describe().rename("Fraud Prob %").to_frame(), use_container_width=False)

        df_up["Prediction"]        = ["Fraud" if p == 1 else "Not Fraud" for p in preds]
        df_up["Fraud_Probability"] = (probas * 100).round(2)

        # ── Summary numbers ───────────────────────────────
        total     = len(df_up)
        n_fraud   = int(preds.sum())
        n_safe    = total - n_fraud
        fraud_pct = n_fraud / total * 100

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Transactions", f"{total:,}")
        k2.metric("🚨 Fraud Detected",  f"{n_fraud:,}",
                  delta=f"{fraud_pct:.2f}% of total", delta_color="inverse")
        k3.metric("✅ Safe Transactions", f"{n_safe:,}")
        k4.metric("Fraud Rate",          f"{fraud_pct:.3f}%")

        # Accuracy vs ground truth if Class column exists
        if has_labels:
            from sklearn.metrics import accuracy_score, classification_report
            true_labels = df_up["Class"].astype(int).values
            acc = accuracy_score(true_labels, preds) * 100
            st.success(f"🎯 Accuracy vs ground-truth labels: **{acc:.2f}%**")

        st.markdown("---")

        # ── Pie chart + bar chart side by side ────────
        col_pie, col_bar = st.columns(2)

        with col_pie:
            st.markdown("### 🥧 Fraud vs Safe Split")
            fig, ax = plt.subplots(figsize=(5, 4))
            sizes  = [n_safe, n_fraud]
            labels = [f"Not Fraud\n{n_safe:,}", f"Fraud\n{n_fraud:,}"]
            colors = ["#2ecc71", "#e74c3c"]
            explode= [0, 0.08]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors, explode=explode,
                autopct="%1.2f%%", startangle=140,
                textprops={"color": "white", "fontsize": 11},
                wedgeprops={"linewidth": 2, "edgecolor": "white"},
            )
            for at in autotexts:
                at.set_fontsize(12)
                at.set_fontweight("bold")
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=True)

        with col_bar:
            st.markdown("### 📊 Fraud Probability Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(probas[preds == 0] * 100, bins=40, color="#2ecc71",
                    alpha=0.75, label="Not Fraud", density=True)
            ax.hist(probas[preds == 1] * 100, bins=40, color="#e74c3c",
                    alpha=0.85, label="Fraud",     density=True)
            ax.set_xlabel("Fraud Probability (%)", color="white")
            ax.set_ylabel("Density", color="white")
            ax.tick_params(colors="white")
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
            for spine in ax.spines.values():
                spine.set_color("#555")
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=True)

        st.markdown("---")

        # ── Amount analysis ───────────────────────────
        st.markdown("### 💰 Transaction Amount Analysis")
        am1, am2, am3, am4 = st.columns(4)
        fraud_rows  = df_up[df_up["Prediction"] == "Fraud"]["Amount"]
        safe_rows   = df_up[df_up["Prediction"] == "Not Fraud"]["Amount"]
        am1.metric("Avg Fraud Amount",   f"${fraud_rows.mean():.2f}"  if len(fraud_rows) else "N/A")
        am2.metric("Avg Safe Amount",    f"${safe_rows.mean():.2f}"   if len(safe_rows)  else "N/A")
        am3.metric("Max Fraud Amount",   f"${fraud_rows.max():.2f}"   if len(fraud_rows) else "N/A")
        am4.metric("Total Fraud Amount", f"${fraud_rows.sum():,.2f}"  if len(fraud_rows) else "N/A")

        st.markdown("---")

        # ── Download results ──────────────────────────
        st.markdown("### ⬇️ Download Results")
        csv_out = df_up.to_csv(index=False).encode("utf-8")
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button(
            label     = "📥 Download Full Results (CSV)",
            data      = csv_out,
            file_name = "fraud_analysis_results.csv",
            mime      = "text/csv",
            use_container_width=True,
        )
        # Fraud-only CSV
        if n_fraud > 0:
            fraud_csv = df_up[df_up["Prediction"] == "Fraud"].to_csv(index=False).encode("utf-8")
            d2.download_button(
                label     = "🚨 Download Fraud-Only (CSV)",
                data      = fraud_csv,
                file_name = "fraud_transactions_only.csv",
                mime      = "text/csv",
                use_container_width=True,
            )


# ──────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;font-size:13px;'>"
    "Credit Card Fraud Detection System · Random Forest · SMOTE · Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
