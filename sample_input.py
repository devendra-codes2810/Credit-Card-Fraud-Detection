"""
============================================================
  sample_input.py
  ─────────────────────────────────────────────────────────
  Demonstrates how to call predict_transaction() after
  running train.py.

  Usage:
      python sample_input.py
============================================================
"""

from train import predict_transaction

# ────────────────────────────────────────────────────────
# EXAMPLE 1 – Typical LEGITIMATE transaction
# ────────────────────────────────────────────────────────
# Feature order: [Time, V1, V2, V3, ..., V28, Amount]

normal_transaction = [
       0,       # Time  (seconds from first transaction)
  -1.3598,      # V1
  -0.0728,      # V2
   2.5363,      # V3
   1.3782,      # V4
  -0.3383,      # V5
   0.4624,      # V6
   0.2396,      # V7
   0.0987,      # V8
   0.3638,      # V9
   0.0908,      # V10
  -0.5516,      # V11
  -0.6178,      # V12
  -0.9914,      # V13
  -0.3112,      # V14
   1.4681,      # V15
  -0.4704,      # V16
   0.2080,      # V17
   0.0258,      # V18
   0.4039,      # V19
   0.2514,      # V20
  -0.0183,      # V21
   0.2778,      # V22
  -0.1105,      # V23
   0.0669,      # V24
   0.1285,      # V25
  -0.1891,      # V26
   0.1336,      # V27
  -0.0211,      # V28
  149.62,       # Amount
]

# ────────────────────────────────────────────────────────
# EXAMPLE 2 – Known FRAUD transaction
# ────────────────────────────────────────────────────────
fraud_transaction = [
     406,        # Time
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


# ────────────────────────────────────────────────────────
# RUN PREDICTIONS
# ────────────────────────────────────────────────────────
def print_result(name: str, features: list):
    print(f"\n{'─'*50}")
    print(f"  Transaction : {name}")
    result = predict_transaction(features)
    icon   = "⚠️  FRAUD" if result["label"] == "Fraud" else "✅ SAFE"
    print(f"  Decision    : {icon}")
    print(f"  Fraud Prob  : {result['probability']*100:.2f}%")
    print(f"  Confidence  : {result['confidence']*100:.2f}%")
    print(f"{'─'*50}")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  SAMPLE TRANSACTION PREDICTIONS")
    print("="*50)

    print_result("Normal / Legitimate",  normal_transaction)
    print_result("Known Fraud",          fraud_transaction)

    print("\n✅ Done. Launch the UI with: streamlit run app.py\n")
