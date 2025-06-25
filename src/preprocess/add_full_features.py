import pandas as pd
import pathlib as pl

BASE = pl.Path("data/interim")
TRAIN_IN = BASE / "features_train.parquet"
TEST_IN = BASE / "features_test.parquet"

TRAIN_OUT = BASE / "features_train_full.parquet"
TEST_OUT = BASE / "features_test_full.parquet"

# List all columns to keep for the full model
FEATURES = [
    "cycle_time",
    "filling_time",
    "plasticizing_time",
    "clamp_close_time",
    "max_inj_pressure",
    "max_back_pressure",
    "avg_back_pressure",
    "max_screw_rpm",
    "avg_screw_rpm",
    "max_inj_speed",
    "cushion_error",
    "switch_over_error",
    "barrel_temp_avg",
    "hopper_temp",
    "mold_temp_avg",
    "delta_mold_temp",
    "outlier_flag",
    "label"
]

def filter_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return df[FEATURES].copy()

for input_path, output_path in [(TRAIN_IN, TRAIN_OUT), (TEST_IN, TEST_OUT)]:
    df = pd.read_parquet(input_path)
    df_full = filter_features(df)
    df_full.to_parquet(output_path)
    print(f"Wrote {output_path}")
