"""
Add (or verify) extra features for the XGBoost model.
We already have:
    • delta_mold_temp  : span of the 12 mould-temperature sensors
    • outlier_flag     : binary flag from Week 1

All we do is **check** that both columns exist and then write *_wfeats.parquet*.
No further calculations are needed.
"""

import pandas as pd
import pathlib as pl

BASE  = pl.Path("data/interim")
TRAIN = BASE / "features_train.parquet"     # raw, before scaling
TEST  = BASE / "features_test.parquet"

REQ_COLS = {"delta_mold_temp", "outlier_flag"}

def verify_feats(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = REQ_COLS.difference(df.columns)
    if missing:
        raise KeyError(f"{name}: missing required column(s): {', '.join(missing)}")
    return df

for path in (TRAIN, TEST):
    df  = pd.read_parquet(path)
    df  = verify_feats(df, path.name)
    out = path.with_name(path.stem + "_wfeats.parquet")
    df.to_parquet(out)
    print("✔︎ wrote", out)
