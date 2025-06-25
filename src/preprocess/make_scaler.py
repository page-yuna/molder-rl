import pandas as pd, pathlib as pl
from sklearn.preprocessing import StandardScaler
import joblib

# 1. load train set only
train = pd.read_parquet("data/interim/features_train.parquet")

# 2. separate X / y
X_train = train.drop(columns=["label"])
y_train = train["label"]

# 3. fit StandardScaler (mean=0, std=1 per feature)
scaler = StandardScaler().fit(X_train)

# 4. transform train and test -> save for quick reload
out_dir = pl.Path("data/interim")

pd.DataFrame(scaler.transform(X_train), columns=X_train.columns).assign(label=y_train)\
  .to_parquet(out_dir / "features_train_scaled.parquet", index=False)

test = pd.read_parquet("data/interim/features_test.parquet")
X_test_scaled = scaler.transform(test.drop(columns=["label"]))
pd.DataFrame(X_test_scaled, columns=X_train.columns).assign(label=test["label"])\
  .to_parquet(out_dir / "features_test_scaled.parquet", index=False)

# 5. persist the scaler for RL environment & inference
joblib.dump(scaler, "data/interim/standard_scaler.pkl")

# 6. compute class weights
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
w0 = len(y_train) / (2 * n_neg)
w1 = len(y_train) / (2 * n_pos)
print(f"class_weights = {{0: {w0:.2f}, 1: {w1:.2f}}}")

# save weights to a tiny JSON
import json, os
with open("data/interim/class_weights.json", "w") as f:
    json.dump({0: w0, 1: w1}, f)
