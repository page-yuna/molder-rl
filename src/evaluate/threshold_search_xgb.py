import pathlib as pl, pandas as pd, numpy as np, joblib, xgboost as xgb
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report

MODEL   = "models/xgb_quality_best.json"          # <-- change if .pkl
TEST    = pl.Path("data/interim/features_test_scaled.parquet")

df      = pd.read_parquet(TEST)
X, y    = df.drop(columns="label"), df["label"]

# load model
model = xgb.XGBClassifier()
model.load_model(MODEL)

prob  = model.predict_proba(X)[:,1]
prec, rec, thr = precision_recall_curve(y, prob)

best = None
for p,r,t in zip(prec, rec, thr):
    if r >= 0.70:                      # NEW target: recall ≥ 70 %
        f1 = 2*p*r/(p+r+1e-9)
        if best is None or f1 > best[2]:
            best = (p, r, f1, t)

print(f"Chosen threshold {best[3]:.4f}  ->  P={best[0]:.2f}  R={best[1]:.2f}  F1={best[2]:.2f}\n")

# ---------- QUICK THRESHOLD CHECK  ----------------------------------
from sklearn.metrics import precision_score, recall_score
for test_thr in [0.0020, 0.0025, 0.0030]:
    pred_test = (prob >= test_thr).astype(int)
    p_test = precision_score(y, pred_test)
    r_test = recall_score(y, pred_test)
    print(f"thr={test_thr:.4f}  P={p_test:.3f}  R={r_test:.3f}")
# --------------------------------------------------------------------

pred = (prob >= best[3]).astype(int)
print("Confusion matrix [actual 1/0 rows, predicted 1/0 cols]")
print(confusion_matrix(y, pred, labels=[1, 0]))
print("\n", classification_report(y, pred, digits=4))


