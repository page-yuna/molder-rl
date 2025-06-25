# src/models/train_xgb_grid.py
import pandas as pd, pathlib as pl, xgboost as xgb
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

TRAIN = pl.Path("data/interim/features_train_scaled.parquet")
TEST  = pl.Path("data/interim/features_test_scaled.parquet")

dft = pd.read_parquet(TRAIN)
Xtr, ytr = dft.drop(columns="label"), dft["label"]

# ── B  oversample minority (SMOTE-NC; here all features numeric) ──────────
sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(Xtr, ytr)

# ── A  grid search ───────────────────────────────────────────────────────
param = {
    "max_depth":    [4, 6, 8],
    "learning_rate":[0.05, 0.1],
    "n_estimators": [400, 800],
    "subsample":    [0.8, 1.0]
}
base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42)
gs = GridSearchCV(base, param, scoring="average_precision", cv=3, verbose=1)
gs.fit(X_bal, y_bal)

best = gs.best_estimator_
print("Best params:", gs.best_params_)

best.save_model("models/xgb_quality_best.json")
