# 🔍 Credit Card Fraud Detection System

A complete, end-to-end Machine Learning system for real-time credit card fraud detection using Random Forest.

---

## 📁 Project Structure

```
fraud_detection/
├── data/
│   └── creditcard.csv          ← Download from Kaggle (link below)
├── model/
│   ├── model.pkl               ← Trained Random Forest (auto-generated)
│   └── scaler.pkl              ← Fitted StandardScaler (auto-generated)
├── plots/                      ← EDA & evaluation charts (auto-generated)
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_auc_curve.png
│   └── feature_importance.png
├── train.py                    ← Full training pipeline
├── app.py                      ← Streamlit web application
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the Dataset
- Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Place it at `./data/creditcard.csv`

### 3. Train the Model
```bash
python train.py
```

### 4. Launch the App
```bash
streamlit run app.py
```

---

## 🧠 ML Pipeline

| Step | Implementation |
|------|---------------|
| **Data Loading** | `pandas.read_csv` |
| **Missing Values** | `dropna()` |
| **Duplicates** | `drop_duplicates()` |
| **Feature Scaling** | `StandardScaler` on Time & Amount |
| **Class Imbalance** | `SMOTE` (oversamples fraud minority class) |
| **Model** | `RandomForestClassifier(n_estimators=100, max_depth=20)` |
| **Train/Test Split** | 80/20 stratified |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Persistence** | `joblib` → model.pkl + scaler.pkl |

---

## 🔮 Single Transaction Prediction

```python
from train import predict_transaction

# Feature order: [Time, V1, V2, ..., V28, Amount]
sample = [
    406,           # Time
    -1.1583,       # V1
     1.0614,       # V2
     1.9329,       # V3
    -0.3086,       # V4
     1.1972,       # V5
     0.8029,       # V6
     0.4929,       # V7
    -0.1431,       # V8
     0.6356,       # V9
     0.4638,       # V10
    -0.1142,       # V11
    -0.1833,       # V12
    -0.1951,       # V13
    -1.0339,       # V14
     0.7423,       # V15
     0.9527,       # V16
    -0.7943,       # V17
     0.0003,       # V18
    -1.7438,       # V19
     0.5553,       # V20
     0.6082,       # V21
    -0.0695,       # V22
    -0.8529,       # V23
    -1.1024,       # V24
    -0.2945,       # V25
    -0.3166,       # V26
     0.0194,       # V27
     0.0693,       # V28
    239.93,        # Amount
]

result = predict_transaction(sample)
print(result)
# → {'label': 'Fraud', 'probability': 0.97, 'confidence': 0.97}
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle – ULB Machine Learning Group |
| Total transactions | 284,807 |
| Fraud transactions | 492 (0.172%) |
| Features | Time, V1–V28 (PCA-anonymised), Amount |
| Target | Class (0 = Not Fraud, 1 = Fraud) |

---

## 🖥️ Streamlit UI Features

- **Dashboard** – KPI cards, class distribution, amount distribution, correlation heatmap
- **Predict Transaction** – Live form with all 30 features + one-click example loaders
- **Model Reports** – All evaluation plots (confusion matrix, ROC curve, feature importance)
- **About** – Pipeline explanation, dataset info, usage instructions

---

## 📈 Expected Model Performance

| Metric | Typical Range |
|--------|--------------|
| Accuracy | 99.9%+ |
| Precision (Fraud) | 90–97% |
| Recall (Fraud) | 80–90% |
| F1-Score (Fraud) | 85–93% |
| ROC-AUC | 0.97–0.99 |

> High accuracy is expected due to class imbalance; focus on Recall and ROC-AUC for fraud detection quality.
