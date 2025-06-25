import pandas as pd, numpy as np, pathlib as pl

df = pd.read_parquet("data/interim/molding_labeled.parquet")
key_cols = ["Max_Injection_Pressure", "Cycle_Time"]   # <-- updated names

def iqr_mask(series):
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper)

# boolean mask per column
flags = pd.DataFrame({c: iqr_mask(df[c]) for c in key_cols})

# cycle flagged if any key sensor is outlier
df["outlier_flag"] = flags.any(axis=1).astype("int8")

# save updated table
df.to_parquet("data/interim/molding_labeled_w_outlier.parquet", index=False)

# summary CSV
summary = flags.sum().rename("n_outliers").to_frame()
summary.to_csv("reports/quality/outlier_summary.csv")

print("Outliers per column:")
print(summary)
print("Total flagged cycles:", df["outlier_flag"].sum())