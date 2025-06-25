# run in a throw-away notebook or new script
import numpy as np, pandas as pd, pathlib as pl, joblib
from sklearn.metrics import precision_recall_curve, f1_score

TEST = pl.Path("data/interim/features_test_scaled.parquet")
df    = pd.read_parquet(TEST)
X, y = df.drop(columns="label"), df["label"]

clf   = joblib.load("models/xgb_quality.pkl")
prob  = clf.predict_proba(X)[:,1]

prec, rec, thr = precision_recall_curve(y, prob)

best = None
for p,r,t in zip(prec, rec, thr):
    if r >= 0.60:                        # target recall ≥ 0.60
        f1 = 2*p*r/(p+r+1e-9)
        if best is None or f1 > best[2]:
            best = (p, r, f1, t)

print(f"Chosen threshold {best[3]:.3f}  →  P={best[0]:.2f}  R={best[1]:.2f}  F1={best[2]:.2f}")
