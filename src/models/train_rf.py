import pandas as pd, pathlib as pl, json, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.tensorboard import SummaryWriter

# paths
DIR = pl.Path("data/interim")
X_train = pd.read_parquet(DIR / "features_train_scaled.parquet")
X_test  = pd.read_parquet(DIR / "features_test_scaled.parquet")

y_train = X_train.pop("label")
y_test  = X_test.pop("label")

with open(DIR / "class_weights.json") as f:
    class_w = {int(k): v for k, v in json.load(f).items()}

# model
rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight=class_w,
        random_state=42
     ).fit(X_train, y_train)

# predictions & metrics
y_pred = rf.predict(X_test)
report = classification_report(y_test, y_pred, digits=4, output_dict=True)
auc    = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])

print("F1-pos :", report['1']['f1-score'] , "  AUC :", auc)

# tensorboard
tb = SummaryWriter("runs/baseline_rf")
tb.add_scalar("F1_fail", report['1']['f1-score'], 0)
tb.add_scalar("AUC", auc, 0)
tb.close()

# save model
joblib.dump(rf, "rf_baseline.pkl")
