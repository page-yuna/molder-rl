import pandas as pd, pathlib as pl
from sklearn.model_selection import train_test_split

# 1. load the 18-column feature parquet
df = pd.read_parquet("data/interim/molding_features.parquet")

# 2. stratified 80 / 20 split
train, test = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

# 3. save
out_dir = pl.Path("data/interim")
train.to_parquet(out_dir / "features_train.parquet", index=False)
test.to_parquet(out_dir / "features_test.parquet",  index=False)

print("Train shape :", train.shape,  " label counts:", train['label'].value_counts().to_dict())
print("Test  shape :", test.shape,   " label counts:", test['label'].value_counts().to_dict())
