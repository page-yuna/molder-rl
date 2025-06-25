"""
Evaluate PPO model on the held-out test set.

Outputs
-------
Mean reward per shot  : combines defect penalty + cycle-/energy terms
Policy fail-rate      : NG % the agent actually incurs
Baseline fail-rate    : NG % in historical data (ground truth)
"""

import pathlib as pl
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

model_200k = "models/ppo_t200k.zip"
model_1M = "models/ppo_t1M_w2.zip"
model_2M = "models/ppo_t2M_w2.zip"
model_3M = "models/ppo_t3M_w3.zip"
model_20M = "models/ppo_t20M_w3.zip"

# ----------------------------------------------------------------------
# model and data paths
MODEL_PATH = model_3M            #  <-- change as needed
TEST_PARQ  = pl.Path("data/interim/features_test_scaled.parquet")
# ----------------------------------------------------------------------

# ---------- load data -------------------------------------------------
df  = pd.read_parquet(TEST_PARQ)
X   = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
y   = df["label"].to_numpy(dtype=np.int8)

# ---------- load policy ----------------------------------------------
model = PPO.load(MODEL_PATH, device="cpu")

# ---------- plain environment (not wrapped) ---------------------------
from src.envs.injection_env import InjectionEnv   # <- import the class
env = InjectionEnv()                              # <- create *instance*
env.reset()                                       # <- initialise once

# ---------- rollout ---------------------------------------------------
tot_reward, tot_fails = 0.0, 0

for obs_row, true_lab in zip(X, y):
    action, _ = model.predict(obs_row, deterministic=True)
    _, reward, _, _, info = env.step(action)      # InjectionEnv.step(action)
    tot_reward += reward
    tot_fails  += true_lab                   

mean_reward    = tot_reward / len(y)
policy_fail_rt = tot_fails  / len(y)
baseline_rt    = y.mean()

print(f"Mean reward per shot : {mean_reward:.3f}")
print(f"Policy fail-rate     : {policy_fail_rt*100:.2f} %")
print(f"Baseline fail-rate   : {baseline_rt*100:.2f} %")