# app.py
import xgboost as xgb, numpy as np, pandas as pd, streamlit as st
from pathlib import Path

MODEL_PATH = Path("models/xgb_quality_smote.json")
_THRESH    = 0.0025   # your tuned threshold

# ---- load json -------------------------------------------------
MODEL = xgb.XGBClassifier()
MODEL.load_model(MODEL_PATH)             

def prob_fail(x: np.ndarray) -> float:
    return float(MODEL.predict_proba(x.reshape(1, -1))[0, 1])

def predict_fail(x: np.ndarray) -> int:
    return int(prob_fail(x) >= _THRESH)

# ------------------------------------------------------------------

st.title("Quality Alarm Dashboard")

st.info(f"Model : XGB • Threshold = {_THRESH:.4f}")

uploaded = st.file_uploader("Upload *scaled* feature file (CSV or Parquet)")
if not uploaded:
    st.stop()


# --- read upload (csv or parquet) ---------------------------------
if uploaded.name.endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_parquet(uploaded)

has_label   = "label" in df.columns
features    = df.drop(columns="label", errors="ignore")

# --- model inference ----------------------------------------------
probs = MODEL.predict_proba(features.to_numpy(dtype=np.float32))[:, 1]
preds = (probs >= _THRESH).astype(int)

df["P_fail"]  = np.round(probs, 3)
df["Pred_NG"] = preds

# sort by risk
df = df.sort_values("P_fail", ascending=False).reset_index(drop=True)

# --- style helper --------------------------------------------------
def color_row(row):
    return ["background-color: salmon" if row["Pred_NG"] else "" for _ in row]

styled = (df.style.hide(axis="index")
              .apply(color_row, axis=1)
              .format({"P_fail": "{:.3f}"}))

st.dataframe(styled, height=600)
st.markdown("*Salmon = NG alarm*")

# download button
st.download_button("Download predictions CSV",
                   df.to_csv(index=False).encode(),
                   "predictions.csv",
                   mime="text/csv")

# ---- quick metrics -------------------------------------------------
if has_label:
    from sklearn.metrics import precision_score, recall_score
    recall    = recall_score(df["label"], preds)
    precision = precision_score(df["label"], preds)
    st.subheader("Metrics (file)")
    st.metric("Recall",    f"{recall*100:.1f} %")
    st.metric("Precision", f"{precision*100:.1f} %")