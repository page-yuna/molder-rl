import pathlib as pl
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

BASE = pl.Path("data/interim")
TRAIN_IN = BASE / "features_train_full.parquet"
TEST_IN = BASE / "features_test_full.parquet"

TRAIN_OUT = BASE / "features_train_full_scaled.parquet"
TEST_OUT = BASE / "features_test_full_scaled.parquet"
SCALER_PATH = BASE / "scaler_full.pkl"

def fit_scaler(df):
    features = df.drop(columns="label")
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler

def scale_and_save(df, scaler):
    features = df.drop(columns="label")
    scaled_features = scaler.transform(features)
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns, index=df.index)
    df_scaled["label"] = df["label"].values
    return df_scaled

def main():
    # Load train data
    df_train = pd.read_parquet(TRAIN_IN)
    scaler = fit_scaler(df_train)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    # Scale train data and save
    df_train_scaled = scale_and_save(df_train, scaler)
    df_train_scaled.to_parquet(TRAIN_OUT)
    print(f"Scaled train data saved to {TRAIN_OUT}")

    # Load test data
    df_test = pd.read_parquet(TEST_IN)
    # Scale test data and save
    df_test_scaled = scale_and_save(df_test, scaler)
    df_test_scaled.to_parquet(TEST_OUT)
    print(f"Scaled test data saved to {TEST_OUT}")

if __name__ == "__main__":
    main()
