"""
============================================================
  Credit Card Fraud Detection - Model Training Pipeline
  Author  : ML Engineer
  Dataset : Kaggle Credit Card Fraud Detection
            https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
============================================================

HOW TO USE
----------
1. Download creditcard.csv from Kaggle and place it in ./data/
2. pip install -r requirements.txt
3. python train.py
4. streamlit run app.py
"""

# ── Standard Library ────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

# ── Data Handling ────────────────────────────────────────────
import numpy  as np
import pandas as pd

# ── Visualisation ────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Preprocessing & Sampling ─────────────────────────────────
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection  import train_test_split, StratifiedKFold
from sklearn.utils            import resample

# ── Optional SMOTE (requires imbalanced-learn compatible with your Python) ──
try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False

# ── Model ────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier

# ── Evaluation ───────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
    classification_report,
)

# ── Persistence ──────────────────────────────────────────────
import joblib

# ────────────────────────────────────────────────────────────
# 0.  PATHS
# ────────────────────────────────────────────────────────────
# Always resolve paths relative to THIS script's folder.
# This means python train.py works correctly regardless of
# which directory you run the command from.
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data",  "creditcard.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
MODEL_PATH  = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ────────────────────────────────────────────────────────────
def generate_synthetic_data(path: str, n_samples: int = 50_000) -> pd.DataFrame:
    """
    Generate a realistic synthetic credit-card dataset and save it as CSV.

    The synthetic data mimics the real Kaggle dataset structure:
      • 28 PCA-like features  (V1–V28)  drawn from class-specific Gaussians
      • Time  – seconds elapsed, uniformly spread over 48 h
      • Amount – log-normal, higher median for fraud
      • Class  – 0.2 % fraud  (mirrors real dataset imbalance)

    This lets the full pipeline run without downloading anything.
    """
    rng        = np.random.default_rng(42)
    n_fraud    = int(n_samples * 0.002)        # ≈ 0.2 % fraud
    n_normal   = n_samples - n_fraud

    def make_rows(n, label):
        # Each PCA feature has a slightly different mean for fraud vs normal
        shift = rng.uniform(-2, 2, size=28) if label == 1 else np.zeros(28)
        V     = rng.standard_normal((n, 28)) + shift
        Time  = rng.uniform(0, 172_800, n)     # 0 – 48 h in seconds
        if label == 1:
            Amount = rng.lognormal(mean=4.0, sigma=1.5, size=n)   # fraud: higher amounts
        else:
            Amount = rng.lognormal(mean=3.5, sigma=1.2, size=n)   # normal
        Amount = np.clip(Amount, 0.5, 25_000)
        df_part = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
        df_part.insert(0,  "Time",   Time)
        df_part["Amount"] = Amount
        df_part["Class"]  = label
        return df_part

    df = pd.concat([make_rows(n_normal, 0), make_rows(n_fraud, 1)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_data(path: str) -> pd.DataFrame:
    """
    Load the CSV dataset and print basic statistics.

    If the real Kaggle CSV is not present the function automatically
    generates a synthetic stand-in so the pipeline can run immediately.
    Download the real dataset later from:
      https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """
    if not os.path.exists(path):
        print("\n" + "="*60)
        print("  DATASET NOT FOUND – generating synthetic data …")
        print("="*60)
        print(f"  Expected path : {path}")
        print("  Generating 50,000-row synthetic dataset (≈ 0.2% fraud) …")
        df = generate_synthetic_data(path)
        print(f"  ✓ Saved to {path}")
        print("  NOTE: For production results, replace with the real Kaggle CSV.")
    else:
        df = pd.read_csv(path)

    print("\n" + "="*60)
    print("  DATASET LOADED")
    print("="*60)
    print(f"  Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory  : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"\n  Class distribution:")
    vc = df["Class"].value_counts()
    print(f"    Non-Fraud (0) : {vc[0]:>7,}  ({vc[0]/len(df)*100:.2f}%)")
    print(f"    Fraud     (1) : {vc[1]:>7,}  ({vc[1]/len(df)*100:.2f}%)")
    return df


# ────────────────────────────────────────────────────────────
# 2.  EXPLORATORY DATA ANALYSIS
# ────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame):
    """Generate and save EDA plots."""
    print("\n[EDA] Generating plots …")
    sns.set_style("whitegrid")
    palette = {"Not Fraud": "#2ecc71", "Fraud": "#e74c3c"}

    # ── 2a. Class Distribution ──────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Class"].value_counts()
    bars   = ax.bar(
        ["Not Fraud (0)", "Fraud (1)"],
        counts.values,
        color=["#2ecc71", "#e74c3c"],
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1500,
            f"{val:,}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_title("Class Distribution (Fraud vs Non-Fraud)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Transactions")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    # ── 2b. Transaction Amount Distribution ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, cls, label, color in zip(
        axes, [0, 1],
        ["Not Fraud", "Fraud"],
        ["#2ecc71", "#e74c3c"],
    ):
        subset = df[df["Class"] == cls]["Amount"]
        ax.hist(subset, bins=60, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"Amount Distribution – {label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Transaction Amount (USD)")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "amount_distribution.png"), dpi=150)
    plt.close()

    # ── 2c. Correlation Heatmap (top features by variance) ──
    # Use a sample to keep the heatmap readable
    sample_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 15)] + ["Class"]
    corr = df[sample_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=False, cmap="RdYlGn",
        vmin=-1, vmax=1, linewidths=0.3, ax=ax,
    )
    ax.set_title("Correlation Heatmap (Time, Amount, V1–V14, Class)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # ── 2d. Fraud vs Non-Fraud – mean feature values ────────
    mean_vals = df.groupby("Class")[[f"V{i}" for i in range(1, 10)]].mean().T
    mean_vals.columns = ["Not Fraud", "Fraud"]
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_vals.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="white")
    ax.set_title("Mean Feature Values (V1–V9): Fraud vs Non-Fraud", fontsize=13, fontweight="bold")
    ax.set_xlabel("PCA Feature")
    ax.set_ylabel("Mean Value")
    ax.legend(title="Class")
    ax.axhline(0, color="black", linewidth=0.5)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_comparison.png"), dpi=150)
    plt.close()

    print("  ✓ Plots saved to ./plots/")


# ────────────────────────────────────────────────────────────
# 3.  PREPROCESSING
# ────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    Steps:
      1. Drop duplicates & handle missing values
      2. Scale 'Time' and 'Amount' with StandardScaler
      3. Return feature matrix X, target y, and the fitted scaler
    """
    print("\n[PREPROCESS] Cleaning and scaling …")

    # -- Missing values
    missing = df.isnull().sum().sum()
    print(f"  Missing values : {missing}")
    df = df.dropna()

    # -- Duplicates
    dupes = df.duplicated().sum()
    print(f"  Duplicate rows : {dupes}")
    df = df.drop_duplicates()

    # -- Feature / target split
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # -- Scale Time & Amount (V1–V28 are already PCA-transformed)
    scaler = StandardScaler()
    X = X.copy()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    print(f"  Feature matrix : {X.shape}")
    return X, y, scaler


# ────────────────────────────────────────────────────────────
# 4.  HANDLE CLASS IMBALANCE  (SMOTE → undersampling fallback)
# ────────────────────────────────────────────────────────────
def apply_smote(X_train, y_train):
    """
    Balance classes.

    Strategy:
      • If imbalanced-learn is installed and compatible → use SMOTE.
      • Otherwise → undersample the majority class with sklearn resample()
        (no extra dependencies; keeps 10× fraud rows from majority).
    """
    n_fraud  = int((y_train == 1).sum())
    n_normal = int((y_train == 0).sum())
    print(f"\n[BALANCE] Before – Non-Fraud: {n_normal:,}  |  Fraud: {n_fraud:,}")

    if _SMOTE_AVAILABLE:
        print("  Method : SMOTE (oversampling)")
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
    else:
        print("  Method : Random undersampling (imblearn not compatible with Python 3.13)")
        print("           Tip: pip install imbalanced-learn==0.11.0  to enable SMOTE")

        target_majority = min(n_normal, n_fraud * 10)
        X_df = X_train.copy()
        X_df["__label__"] = y_train.values

        majority      = X_df[X_df["__label__"] == 0]
        minority      = X_df[X_df["__label__"] == 1]
        majority_down = resample(majority, replace=False,
                                  n_samples=target_majority, random_state=42)
        balanced = pd.concat([majority_down, minority]).sample(frac=1, random_state=42)
        y_res = balanced["__label__"].astype(int)
        X_res = balanced.drop("__label__", axis=1)

    print(f"  After  – Non-Fraud: {int((y_res==0).sum()):,}  |  Fraud: {int((y_res==1).sum()):,}")
    return X_res, y_res


# ────────────────────────────────────────────────────────────
# 5.  TRAIN RANDOM FOREST
# ────────────────────────────────────────────────────────────
def train_model(X_train, y_train) -> RandomForestClassifier:
    """
    Train a Random Forest with tuned hyperparameters.

    Key choices:
      n_estimators  = 100   – enough trees for stability
      max_depth     = 20    – limit depth to avoid over-fitting
      class_weight  = None  – SMOTE already balanced the data
      n_jobs        = -1    – use all CPU cores
    """
    print("\n[MODEL] Training Random Forest Classifier …")
    rf = RandomForestClassifier(
        n_estimators  = 100,
        max_depth     = 20,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features  = "sqrt",
        random_state  = 42,
        n_jobs        = -1,
    )
    rf.fit(X_train, y_train)
    print("  ✓ Model trained successfully")
    return rf


# ────────────────────────────────────────────────────────────
# 6.  EVALUATE MODEL
# ────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, feature_names):
    """Print metrics and save evaluation plots."""
    print("\n" + "="*60)
    print("  MODEL EVALUATION")
    print("="*60)

    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score (y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score   (y_test, y_pred)
    f1   = f1_score       (y_test, y_pred)
    auc  = roc_auc_score  (y_test, y_prob)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud'])}")

    # ── Confusion Matrix ────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
        linewidths=1, linecolor="white",
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ── ROC-AUC Curve ───────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#3498db", lw=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#bdc3c7", lw=1.5, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#3498db")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC-AUC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_auc_curve.png"), dpi=150)
    plt.close()

    # ── Feature Importance ──────────────────────────────────
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top15 = importances.nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top15)))
    top15.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    print("\n  ✓ Evaluation plots saved to ./plots/")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}


# ────────────────────────────────────────────────────────────
# 7.  SINGLE TRANSACTION PREDICTION  ← 🔥 KEY FEATURE
# ────────────────────────────────────────────────────────────
def predict_transaction(
    input_data: list,
    model_path : str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
) -> dict:
    """
    Predict whether a single transaction is Fraud or Not Fraud.

    Parameters
    ----------
    input_data : list of 30 floats [Time, V1…V28, Amount]

    Returns
    -------
    dict with keys:
        label       – "Fraud" | "Not Fraud"
        probability – float (0–1), probability of being Fraud
        confidence  – float (0–1), model's confidence in the label
    """
    # -- Load persisted artefacts
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # -- Build a single-row DataFrame (preserves column names)
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df_row = pd.DataFrame([input_data], columns=columns)

    # -- Apply the SAME scaling used during training
    df_row[["Time", "Amount"]] = scaler.transform(df_row[["Time", "Amount"]])

    # -- Predict
    pred  = model.predict(df_row)[0]
    proba = model.predict_proba(df_row)[0]   # [P(Not Fraud), P(Fraud)]

    label      = "Fraud" if pred == 1 else "Not Fraud"
    fraud_prob = proba[1]
    confidence = max(proba)

    return {
        "label"      : label,
        "probability": round(float(fraud_prob), 4),
        "confidence" : round(float(confidence), 4),
    }


# ────────────────────────────────────────────────────────────
# 8.  MAIN PIPELINE
# ────────────────────────────────────────────────────────────
def main():
    # 1. Load
    df = load_data(DATA_PATH)

    # 2. EDA
    run_eda(df)

    # 3. Preprocess
    X, y, scaler = preprocess(df)
    feature_names = X.columns.tolist()

    # 4. Train / Test split  (stratified → preserves fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[SPLIT] Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # 5. Handle imbalance with SMOTE (only on train set!)
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # 6. Train
    model = train_model(X_train_res, y_train_res)

    # 7. Evaluate
    metrics = evaluate_model(model, X_test, y_test, feature_names)

    # 8. Save artefacts
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  ✓ Model  saved → {MODEL_PATH}")
    print(f"  ✓ Scaler saved → {SCALER_PATH}")

    # 9. Quick demo – predict one test transaction
    print("\n" + "="*60)
    print("  DEMO: Single Transaction Prediction")
    print("="*60)
    sample = X_test.iloc[0].tolist()       # grab a test row (already scaled internally)
    # We need the ORIGINAL row for predict_transaction (it re-scales internally)
    # So let's grab the original indices
    original_idx = X_test.index[0]
    original_row = df.loc[original_idx, feature_names].tolist()

    result = predict_transaction(original_row)
    actual = "Fraud" if y_test.iloc[0] == 1 else "Not Fraud"
    print(f"  Actual label  : {actual}")
    print(f"  Predicted     : {result['label']}")
    print(f"  Fraud prob    : {result['probability']:.4f}")
    print(f"  Confidence    : {result['confidence']:.4f}")

    print("\n" + "="*60)
    print("  ✅  Training complete!  Run: streamlit run app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
