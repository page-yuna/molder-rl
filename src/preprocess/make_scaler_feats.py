# src/preprocess/make_scaler_feats.py
"""
Fit a StandardScaler on the TRAIN set (numeric features only),
apply it to both train & test, and save *_scaled.parquet
plus the scaler object (joblib) for later inference.
"""

import pathlib as pl
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

BASE = pl.Path("data/interim")

TRAIN_IN  = BASE / "features_train_wfeats.parquet"   # input with 18 features + label
TEST_IN   = BASE / "features_test_wfeats.parquet"

TRAIN_OUT = BASE / "features_train_scaled.parquet"
TEST_OUT  = BASE / "features_test_scaled.parquet"
SCALER_OUT = BASE / "scaler.joblib"

# ── load -----------------------------------------------------------------
df_train = pd.read_parquet(TRAIN_IN)
df_test  = pd.read_parquet(TEST_IN)

# separate features / label
X_train = df_train.drop(columns="label")
X_test  = df_test.drop(columns="label")

y_train = df_train["label"]
y_test  = df_test["label"]

# ── fit scaler on training features --------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# back to DataFrame with same columns
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns,  index=X_test.index)

df_train_scaled = pd.concat([X_train_scaled, y_train], axis=1)
df_test_scaled  = pd.concat([X_test_scaled,  y_test],  axis=1)

# ── save -----------------------------------------------------------------
df_train_scaled.to_parquet(TRAIN_OUT)
df_test_scaled.to_parquet(TEST_OUT)
joblib.dump(scaler, SCALER_OUT)

print("- Scaled train :", df_train_scaled.shape, "->", TRAIN_OUT.name)
print("- Scaled test  :", df_test_scaled.shape,  "->", TEST_OUT .name)
print("- Scaler saved :", SCALER_OUT)
