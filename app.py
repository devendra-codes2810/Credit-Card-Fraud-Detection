"""
Credit Card Fraud Detection — Gradio App
FIX: EXAMPLES dynamically pulled from data/creditcard.csv at startup
     using the actual trained model — guarantees correct predictions.
Run: python app_gradio.py → http://localhost:7860
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
DATA_PATH   = os.path.join(BASE_DIR, "data", "creditcard.csv")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Run python train.py first!")

COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ─────────────────────────────────────────────────────────────
# FALLBACK EXAMPLES (used only if dataset missing)
# ─────────────────────────────────────────────────────────────
FALLBACK_EXAMPLES = {
    "normal": [
        47300,
        1.0528,-1.7774,-0.6547, 2.0709,-0.0151,-0.2187,
        1.1824,-1.0322,-0.9546,-1.4096, 0.6632,-0.3976,
        0.1324,-1.6407, 0.7592,-0.4432,-1.8859, 0.8475,
        0.5869, 1.0287, 1.5856,-0.6557,-0.5050, 0.7724,
        0.8304,-0.0701,-0.2691,-0.7938,
        9.21
    ],
    "fraud": [
        472,
        -3.0435,-3.1573, 1.0885, 2.2886, 1.3598,-1.0648,
         0.3256,-0.0678,-0.2710,-0.8386,-0.4146,-0.5031,
         0.6765,-1.6920, 2.0006, 0.6668, 0.5997, 1.7253,
         0.2833, 2.1023, 0.6617, 0.4355, 1.3760,-0.2938,
         0.2798,-0.1454,-0.2528, 0.0358,
         529.0
    ],
    "suspicious": [
        129186,
        -2.5346, 6.9470,-1.1006,-0.5301,-0.1365,-1.0065,
        -5.9766,-0.5857,-4.0388,-5.7026, 2.3425,-5.8099,
         0.7207,-7.6307,-2.3047,-1.9131,-6.6564,-0.1923,
        -0.5569, 0.7673, 2.4819, 0.8119,-2.1257,-1.9378,
        -0.5586,-0.3755,-2.4107, 1.1471,
         37.26
    ],
}


# ─────────────────────────────────────────────────────────────
# DYNAMIC EXAMPLES BUILDER — picks REAL rows from dataset
# that the trained model classifies with right confidence
# ─────────────────────────────────────────────────────────────
def build_examples():
    if not os.path.exists(DATA_PATH):
        print("[WARN] data/creditcard.csv not found — using fallback EXAMPLES")
        return FALLBACK_EXAMPLES

    try:
        mdl = joblib.load(MODEL_PATH)
        scl = joblib.load(SCALER_PATH)
        df  = pd.read_csv(DATA_PATH)

        if "Class" not in df.columns:
            print("[WARN] Class column missing — using fallback")
            return FALLBACK_EXAMPLES

        X = df[COLS].copy()
        X[["Time", "Amount"]] = scl.transform(X[["Time", "Amount"]])
        probs = mdl.predict_proba(X)[:, 1]

        examples = {}

        # FRAUD: real fraud row with highest fraud probability
        fraud_mask = df["Class"] == 1
        if fraud_mask.any():
            fraud_probs = probs.copy()
            fraud_probs[~fraud_mask] = -1
            idx_fraud = int(np.argmax(fraud_probs))
            examples["fraud"] = df.iloc[idx_fraud][COLS].tolist()
            print(f"[OK] Fraud example  → row {idx_fraud}, model fraud prob = {probs[idx_fraud]*100:.2f}%")
        else:
            examples["fraud"] = FALLBACK_EXAMPLES["fraud"]

        # NORMAL: real legit row with lowest fraud probability
        legit_mask = df["Class"] == 0
        if legit_mask.any():
            legit_probs = probs.copy()
            legit_probs[~legit_mask] = 2.0
            idx_legit = int(np.argmin(legit_probs))
            examples["normal"] = df.iloc[idx_legit][COLS].tolist()
            print(f"[OK] Normal example → row {idx_legit}, model fraud prob = {probs[idx_legit]*100:.2f}%")
        else:
            examples["normal"] = FALLBACK_EXAMPLES["normal"]

        # SUSPICIOUS: borderline row (fraud prob 0.40–0.75)
        susp_mask = (probs >= 0.40) & (probs <= 0.75)
        if susp_mask.any():
            candidate_idx = np.where(susp_mask)[0]
            best = candidate_idx[np.argmin(np.abs(probs[candidate_idx] - 0.55))]
            examples["suspicious"] = df.iloc[int(best)][COLS].tolist()
            print(f"[OK] Suspicious     → row {int(best)}, model fraud prob = {probs[int(best)]*100:.2f}%")
        else:
            mid_mask = (probs >= 0.30) & (probs <= 0.85)
            if mid_mask.any():
                candidate_idx = np.where(mid_mask)[0]
                best = candidate_idx[np.argmin(np.abs(probs[candidate_idx] - 0.55))]
                examples["suspicious"] = df.iloc[int(best)][COLS].tolist()
                print(f"[OK] Suspicious(mid)→ row {int(best)}, model fraud prob = {probs[int(best)]*100:.2f}%")
            else:
                examples["suspicious"] = FALLBACK_EXAMPLES["suspicious"]

        return examples
    except Exception as e:
        print(f"[WARN] build_examples failed: {e} — using fallback")
        return FALLBACK_EXAMPLES


# Build once at startup
EXAMPLES = build_examples()

SELECTED = {"key": None}


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────
def predict_row(row):
    mdl = joblib.load(MODEL_PATH)
    scl = joblib.load(SCALER_PATH)
    df  = pd.DataFrame([row], columns=COLS)
    df[["Time","Amount"]] = scl.transform(df[["Time","Amount"]])
    prob = float(mdl.predict_proba(df)[0][1])
    pred = int(mdl.predict(df)[0])

    if pred == 1:
        html = f"""
<div style='background:linear-gradient(90deg,#c0392b,#e74c3c);
            border-radius:14px;padding:28px;text-align:center;
            font-size:32px;font-weight:900;color:white;
            box-shadow:0 8px 36px rgba(231,76,60,0.6);'>
  ⚠️ FRAUD DETECTED
</div>
<div style='margin-top:14px;padding:16px 22px;background:#2c0f0f;
            border:1px solid #e74c3c;border-radius:10px;
            color:white;font-size:15px;line-height:2.4;'>
  <b>Fraud Probability :</b> {prob*100:.2f}%<br>
  <b>Safe Probability  :</b> {(1-prob)*100:.2f}%<br>
  <b>Action            :</b> 🚨 Flag for manual review
</div>"""
    else:
        html = f"""
<div style='background:linear-gradient(90deg,#27ae60,#2ecc71);
            border-radius:14px;padding:28px;text-align:center;
            font-size:32px;font-weight:900;color:white;
            box-shadow:0 8px 36px rgba(46,204,113,0.6);'>
  ✅ SAFE TRANSACTION
</div>
<div style='margin-top:14px;padding:16px 22px;background:#0f2c1a;
            border:1px solid #2ecc71;border-radius:10px;
            color:white;font-size:15px;line-height:2.4;'>
  <b>Fraud Probability :</b> {prob*100:.2f}%<br>
  <b>Safe Probability  :</b> {(1-prob)*100:.2f}%<br>
  <b>Action            :</b> ✅ Transaction approved
</div>"""

    fig, ax = plt.subplots(figsize=(6, 0.9))
    clr = "#e74c3c" if prob > 0.4 else "#2ecc71"
    ax.barh([0],[prob],   color=clr,      height=0.55, zorder=3)
    ax.barh([0],[1-prob], color="#2a2a3e",height=0.55, left=prob, zorder=2)
    ax.set_xlim(0,1); ax.set_yticks([])
    ax.set_xticks([0,.25,.5,.75,1])
    ax.set_xticklabels(["0%","25%","50%","75%","100%"],color="white",fontsize=10)
    ax.tick_params(colors="white")
    ax.set_title(f"Fraud Probability: {prob*100:.1f}%",color="white",fontsize=11,pad=8)
    for s in ax.spines.values(): s.set_visible(False)
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#1a1a2e")
    plt.tight_layout()
    return html, fig


# ─────────────────────────────────────────────────────────────
# LOAD BUTTONS
# ─────────────────────────────────────────────────────────────
def load_normal():
    SELECTED["key"] = "normal"
    v = list(EXAMPLES["normal"])
    s = "<div style='background:#2ecc7122;border-left:5px solid #2ecc71;padding:10px 16px;border-radius:8px;color:white;font-size:15px;font-weight:600;margin:8px 0;'>📌 Normal transaction loaded ✅ — ab Analyse dabao</div>"
    return [s] + v

def load_fraud():
    SELECTED["key"] = "fraud"
    v = list(EXAMPLES["fraud"])
    s = "<div style='background:#e74c3c22;border-left:5px solid #e74c3c;padding:10px 16px;border-radius:8px;color:white;font-size:15px;font-weight:600;margin:8px 0;'>📌 Fraud transaction loaded 🚨 — ab Analyse dabao</div>"
    return [s] + v

def load_suspicious():
    SELECTED["key"] = "suspicious"
    v = list(EXAMPLES["suspicious"])
    s = "<div style='background:#f39c1222;border-left:5px solid #f39c12;padding:10px 16px;border-radius:8px;color:white;font-size:15px;font-weight:600;margin:8px 0;'>📌 Suspicious transaction loaded 🟡 — ab Analyse dabao</div>"
    return [s] + v


def analyse_click():
    key = SELECTED["key"]
    if key is None:
        err = "<div style='background:#2a2a3e;border:1px solid #888;border-radius:10px;padding:20px;text-align:center;color:#aaa;font-size:15px;'>⚠️ Pehle Load Normal / Load Fraud / Load Suspicious dabao</div>"
        return err, gr.update(visible=False)
    row = EXAMPLES[key]
    html, fig = predict_row(row)
    return html, gr.update(visible=True, value=fig)


# ─────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────
def build_dashboard():
    if not os.path.exists(DATA_PATH):
        return "<p style='color:#aaa;'>Run train.py first.</p>", None, None
    df = pd.read_csv(DATA_PATH, nrows=50_000)
    total=len(df); nf=int(df["Class"].sum()); ns=total-nf
    rate=nf/total*100
    fa=df[df["Class"]==1]["Amount"].mean()
    na=df[df["Class"]==0]["Amount"].mean()
    kpi=f"""
<div style='display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 18px;'>
  <div style='background:#1a2a3a;border:1px solid #3498db;border-radius:12px;padding:16px 22px;color:white;text-align:center;min-width:120px;'>
    <div style='font-size:24px;font-weight:800;color:#3498db;'>{total:,}</div>
    <div style='font-size:12px;color:#aaa;margin-top:4px;'>Total Transactions</div></div>
  <div style='background:#3a1e1e;border:1px solid #e74c3c;border-radius:12px;padding:16px 22px;color:white;text-align:center;min-width:120px;'>
    <div style='font-size:24px;font-weight:800;color:#e74c3c;'>{nf:,}</div>
    <div style='font-size:12px;color:#aaa;margin-top:4px;'>🚨 Fraud Cases</div></div>
  <div style='background:#1e3a2f;border:1px solid #2ecc71;border-radius:12px;padding:16px 22px;color:white;text-align:center;min-width:120px;'>
    <div style='font-size:24px;font-weight:800;color:#2ecc71;'>{ns:,}</div>
    <div style='font-size:12px;color:#aaa;margin-top:4px;'>✅ Legitimate</div></div>
  <div style='background:#2a1e3a;border:1px solid #9b59b6;border-radius:12px;padding:16px 22px;color:white;text-align:center;min-width:120px;'>
    <div style='font-size:24px;font-weight:800;color:#9b59b6;'>{rate:.3f}%</div>
    <div style='font-size:12px;color:#aaa;margin-top:4px;'>Fraud Rate</div></div>
  <div style='background:#2a2a1e;border:1px solid #f39c12;border-radius:12px;padding:16px 22px;color:white;text-align:center;min-width:120px;'>
    <div style='font-size:24px;font-weight:800;color:#f39c12;'>${fa:.2f}</div>
    <div style='font-size:12px;color:#aaa;margin-top:4px;'>Avg Fraud Amt</div></div>
  <div style='background:#1a2a3a;border:1px solid #1abc9c;border-radius:12px;padding:16px 22px;color:white;text-align:center;min-width:120px;'>
    <div style='font-size:24px;font-weight:800;color:#1abc9c;'>${na:.2f}</div>
    <div style='font-size:12px;color:#aaa;margin-top:4px;'>Avg Normal Amt</div></div>
</div>"""
    fig1,ax1=plt.subplots(figsize=(5,3.5))
    bars=ax1.bar(["Not Fraud","Fraud"],[ns,nf],color=["#2ecc71","#e74c3c"],edgecolor="white",linewidth=1.2)
    for b,v2 in zip(bars,[ns,nf]):
        ax1.text(b.get_x()+b.get_width()/2,b.get_height()+max(ns,nf)*0.02,f"{v2:,}",ha="center",va="bottom",color="white",fontsize=11,fontweight="bold")
    ax1.set_title("Class Distribution",color="white",fontsize=13,fontweight="bold",pad=10)
    ax1.set_ylabel("Count",color="white"); ax1.tick_params(colors="white")
    ax1.set_facecolor("#1a1a2e"); fig1.patch.set_facecolor("#1a1a2e")
    for s in ax1.spines.values(): s.set_color("#444")
    plt.tight_layout()
    fig2,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax2,cls,lbl,clr in zip(axes,[0,1],["Not Fraud","Fraud"],["#2ecc71","#e74c3c"]):
        ax2.hist(df[df["Class"]==cls]["Amount"],bins=50,color=clr,alpha=0.85,edgecolor="white",linewidth=0.3)
        ax2.set_title(f"Amount – {lbl}",color="white",fontsize=11,fontweight="bold")
        ax2.set_xlabel("Amount ($)",color="white"); ax2.set_ylabel("Count",color="white")
        ax2.set_yscale("log"); ax2.tick_params(colors="white")
        ax2.set_facecolor("#1a1a2e")
        for s in ax2.spines.values(): s.set_color("#444")
    fig2.patch.set_facecolor("#1a1a2e"); plt.tight_layout()
    return kpi, fig1, fig2


# ─────────────────────────────────────────────────────────────
# BULK CSV
# ─────────────────────────────────────────────────────────────
def predict_bulk(file, threshold):
    if file is None:
        return "<p style='color:#e74c3c;'>⚠️ Upload a CSV first.</p>", None, None
    mdl=joblib.load(MODEL_PATH); scl=joblib.load(SCALER_PATH)
    df_up=pd.read_csv(file.name)
    missing=[c for c in COLS if c not in df_up.columns]
    if missing:
        return f"<p style='color:#e74c3c;'>❌ Missing: {', '.join(missing)}</p>",None,None
    X=df_up[COLS].copy(); X[["Time","Amount"]]=scl.transform(X[["Time","Amount"]])
    probas=mdl.predict_proba(X)[:,1]; preds=(probas>=threshold).astype(int)
    df_up["Prediction"]=["🚨 Fraud" if p==1 else "✅ Safe" for p in preds]
    df_up["Fraud_Probability%"]=(probas*100).round(2)
    total=len(df_up); nf=int(preds.sum()); ns=total-nf
    acc_line=""
    if "Class" in df_up.columns:
        from sklearn.metrics import accuracy_score
        acc=accuracy_score(df_up["Class"].astype(int),preds)*100
        acc_line=f"<br><b style='color:#2ecc71;'>✅ Accuracy: {acc:.2f}%</b>"
    summary=f"""
<div style='display:flex;gap:12px;flex-wrap:wrap;margin:10px 0;'>
  <div style='background:#1a2a3a;border:1px solid #3498db;border-radius:10px;padding:14px 20px;color:white;text-align:center;'>
    <div style='font-size:22px;font-weight:800;color:#3498db;'>{total:,}</div><div style='font-size:12px;color:#aaa;'>Total</div></div>
  <div style='background:#3a1e1e;border:1px solid #e74c3c;border-radius:10px;padding:14px 20px;color:white;text-align:center;'>
    <div style='font-size:22px;font-weight:800;color:#e74c3c;'>{nf:,}</div><div style='font-size:12px;color:#aaa;'>🚨 Fraud</div></div>
  <div style='background:#1e3a2f;border:1px solid #2ecc71;border-radius:10px;padding:14px 20px;color:white;text-align:center;'>
    <div style='font-size:22px;font-weight:800;color:#2ecc71;'>{ns:,}</div><div style='font-size:12px;color:#aaa;'>✅ Safe</div></div>
  <div style='background:#2a2a3e;border:1px solid #888;border-radius:10px;padding:14px 20px;color:white;text-align:center;'>
    <div style='font-size:22px;font-weight:800;'>{nf/total*100:.2f}%</div><div style='font-size:12px;color:#aaa;'>Fraud Rate</div></div>
</div>{acc_line}"""
    fig,ax=plt.subplots(figsize=(4,4))
    ax.pie([ns,max(nf,0.001)],labels=[f"Safe\n{ns:,}",f"Fraud\n{nf:,}"],
           colors=["#2ecc71","#e74c3c"],autopct="%1.1f%%",startangle=140,
           textprops={"color":"white","fontsize":11},wedgeprops={"linewidth":2,"edgecolor":"white"})
    ax.set_title("Fraud vs Safe",color="white",fontsize=13,fontweight="bold")
    fig.patch.set_facecolor("#1a1a2e"); plt.tight_layout()
    out=os.path.join(BASE_DIR,"fraud_results.csv"); df_up.to_csv(out,index=False)
    return summary, fig, out


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
css = """
body,.gradio-container{background:#0f0c29 !important;color:#f0f0f0;}
button.primary{background:linear-gradient(90deg,#6a11cb,#2575fc) !important;border:none !important;font-weight:700 !important;font-size:15px !important;}
button.secondary{background:#2a2a3e !important;color:white !important;border:1px solid #666 !important;border-radius:8px !important;font-size:14px !important;}
label{color:#ccc !important;font-size:13px !important;}
input[type=number]{background:#1e1e2e !important;color:white !important;border:1px solid #444 !important;border-radius:6px !important;}
"""

with gr.Blocks(css=css, title="🛡️ Credit Card Fraud Detector") as demo:
    gr.HTML("""
<div style='text-align:center;padding:22px 0 8px;background:linear-gradient(135deg,#0f0c29,#302b63);border-radius:12px;margin-bottom:10px;'>
  <h1 style='color:white;font-size:30px;margin:0;'>🛡️ Credit Card Fraud Detection System</h1>
  <p style='color:#aaa;font-size:14px;margin:6px 0 0;'>Random Forest · Real-time ML · Python</p>
</div>""")

    with gr.Tabs():

        with gr.Tab("🏠 Dashboard"):
            gr.HTML("<p style='color:#aaa;margin:6px 0 10px;'>Dataset statistics dekhne ke liye click karo.</p>")
            btn_dash = gr.Button("🔄 Load Dashboard", variant="primary")
            dash_kpi = gr.HTML()
            with gr.Row():
                dash_f1 = gr.Plot(label="Class Distribution")
                dash_f2 = gr.Plot(label="Amount Distribution")
            btn_dash.click(fn=build_dashboard, inputs=[], outputs=[dash_kpi,dash_f1,dash_f2])

        with gr.Tab("🔮 Predict Transaction"):
            gr.HTML("""
<div style='background:#1a1a2e;border:1px solid #3498db;border-radius:10px;padding:14px 18px;margin:8px 0 14px;'>
  <p style='color:white;font-size:15px;font-weight:700;margin:0 0 6px;'>Kaise use karein:</p>
  <p style='color:#aaa;font-size:13px;margin:0;line-height:1.8;'>
    <b style='color:white;'>Step 1</b> — Load button dabao (values neeche inputs mein dikhenge)<br>
    <b style='color:white;'>Step 2</b> — ▶️ Analyse Transaction dabao
  </p>
</div>""")

            gr.HTML("<p style='color:#ccc;font-size:14px;font-weight:700;margin:4px 0 8px;'>▼ Step 1 — Load karo:</p>")
            with gr.Row():
                btn_n = gr.Button("✅ Load Normal",     variant="secondary", size="lg")
                btn_f = gr.Button("🚨 Load Fraud",      variant="secondary", size="lg")
                btn_s = gr.Button("🟡 Load Suspicious", variant="secondary", size="lg")
            load_status = gr.HTML(value="")

            gr.HTML("<hr style='border-color:#333;margin:14px 0;'>")
            gr.HTML("<p style='color:#aaa;font-size:13px;margin:4px 0 8px;'>Values neeche load hongi (reference ke liye):</p>")

            inp_time = gr.Number(label="⏱️ Time (seconds)", value=0, precision=2, interactive=True)
            gr.HTML("<p style='color:#aaa;font-size:13px;margin:8px 0 2px;'>🧬 V1 – V28</p>")
            v_inputs = []
            for rs in range(1, 29, 4):
                with gr.Row():
                    for i in range(rs, min(rs+4, 29)):
                        v_inputs.append(gr.Number(label=f"V{i}", value=0.0, precision=4, interactive=True))
            inp_amount = gr.Number(label="💰 Amount (USD)", value=0, precision=2, interactive=True)

            all_inputs = [inp_time] + v_inputs + [inp_amount]

            gr.HTML("<hr style='border-color:#333;margin:14px 0;'>")
            gr.HTML("<p style='color:#ccc;font-size:14px;font-weight:700;margin:4px 0 8px;'>▼ Step 2 — Analyse karo:</p>")
            btn_analyse = gr.Button("▶️ Analyse Transaction", variant="primary", size="lg")
            gr.HTML("<hr style='border-color:#333;margin:14px 0;'>")
            out_html  = gr.HTML(value="")
            out_gauge = gr.Plot(label="Fraud Probability", visible=False)

            load_outs = [load_status] + all_inputs

            btn_n.click(fn=load_normal,     inputs=[], outputs=load_outs)
            btn_f.click(fn=load_fraud,      inputs=[], outputs=load_outs)
            btn_s.click(fn=load_suspicious, inputs=[], outputs=load_outs)

            btn_analyse.click(
                fn      = analyse_click,
                inputs  = [],
                outputs = [out_html, out_gauge],
            )

        with gr.Tab("📁 Bulk CSV"):
            gr.HTML("<p style='color:#aaa;margin:6px 0 10px;'>CSV upload karo: Time, V1–V28, Amount.</p>")
            with gr.Row():
                csv_up = gr.File(label="📂 Upload CSV", file_types=[".csv"])
                thresh = gr.Slider(0.10,0.90,value=0.30,step=0.05,label="🎯 Fraud Threshold")
            btn_bulk  = gr.Button("🔍 Analyse All Transactions", variant="primary")
            bulk_html = gr.HTML()
            bulk_plot = gr.Plot()
            bulk_file = gr.File(label="⬇️ Download Results")
            btn_bulk.click(fn=predict_bulk,inputs=[csv_up,thresh],outputs=[bulk_html,bulk_plot,bulk_file])

        with gr.Tab("📊 Model Reports"):
            rm={"confusion_matrix.png":"🔲 Confusion Matrix","roc_auc_curve.png":"📈 ROC-AUC Curve","feature_importance.png":"🏆 Feature Importance","class_distribution.png":"📊 Class Distribution"}
            found=[f for f in rm if os.path.exists(os.path.join(PLOTS_DIR,f))]
            if found:
                for i in range(0,len(found),2):
                    with gr.Row():
                        for fname in found[i:i+2]:
                            gr.Image(value=os.path.join(PLOTS_DIR,fname),label=rm[fname])
            else:
                gr.HTML("<p style='color:#aaa;'>Run python train.py to generate plots.</p>")

        with gr.Tab("📖 About"):
            gr.Markdown("""
## 🛡️ Credit Card Fraud Detection
| | |
|--|--|
| Model | Random Forest (100 trees) |
| Examples | Auto-picked from dataset using trained model |
| UI | Gradio |

```bash
pip install gradio scikit-learn pandas numpy joblib matplotlib
python train.py
python app_gradio.py
```
            """)

    gr.HTML("<p style='text-align:center;color:#444;font-size:12px;margin:14px 0 4px;'>Credit Card Fraud Detection · Random Forest · Gradio</p>")

if __name__ == "__main__":
    demo.launch()