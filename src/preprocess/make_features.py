import pandas as pd, numpy as np, pathlib as pl

RAW = pl.Path("data/interim/molding_labeled_w_outlier.parquet")
df  = pd.read_parquet(RAW)

TARGET_CUSHION = 5.0     # mm (use real spec if known)
VP_SETPOINT    = 15.0    # mm switchover

df_feat = pd.DataFrame({
    "cycle_time"        : df["Cycle_Time"],
    "filling_time"      : df["Filling_Time"],
    "plasticizing_time" : df["Plasticizing_Time"],
    "clamp_close_time"  : df["Clamp_Close_Time"],
    "max_inj_pressure"  : df["Max_Injection_Pressure"],
    "max_back_pressure" : df["Max_Back_Pressure"],
    "avg_back_pressure" : df["Average_Back_Pressure"],
    "max_screw_rpm"     : df["Max_Screw_RPM"],
    "avg_screw_rpm"     : df["Average_Screw_RPM"],
    "max_inj_speed"     : df["Max_Injection_Speed"],
    "cushion_error"     : df["Cushion_Position"] - TARGET_CUSHION,
    "switch_over_error" : df["Switch_Over_Position"] - VP_SETPOINT,
    "barrel_temp_avg"   : df[[f"Barrel_Temperature_{i}" for i in range(1,8)]].mean(axis=1),
    "hopper_temp"       : df["Hopper_Temperature"],
    "mold_temp_avg"     : df[[f"Mold_Temperature_{i}" for i in range(1,13)]].mean(axis=1),
    "delta_mold_temp"   : df[[f"Mold_Temperature_{i}" for i in range(1,13)]].max(axis=1)
                          - df[[f"Mold_Temperature_{i}" for i in range(1,13)]].min(axis=1),
    "outlier_flag"      : df["outlier_flag"],
    "label"             : df["label"],
})

df_feat.to_parquet("data/interim/molding_features.parquet", index=False)
print("Features saved:", df_feat.shape)
