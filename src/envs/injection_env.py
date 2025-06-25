import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd, pathlib as pl, json

class InjectionEnv(gym.Env):
    """
    Per-cycle injection-moulding environment (one shot = one episode).
    Observation: 17 scaled features + outlier_flag (float32)
    Action:      3 continuous values in [-1, 1]  (speed, pressure, cooling)
    Reward:      defect penalty + cycle-time & energy terms.
    """

    def __init__(self):
        super().__init__()

        base = pl.Path("data/interim")
        df   = pd.read_parquet(base / "features_train_scaled.parquet")

        self.obs_cols = [c for c in df.columns if c != "label"]
        self.X  = df[self.obs_cols].astype(np.float32).values
        self.y  = df["label"].astype(np.int8).values
        self.n_obs = len(self.obs_cols)

        # --- spaces --------------------------------------------------------
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(3,),  dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n_obs,), dtype=np.float32)

        # --- class weights -------------------------------------------------
        with open(base / "class_weights.json") as f:
            cw = json.load(f)
        self.w0, self.w1 = cw["0"], cw["1"]
        self.w1 *= 6 

        # episode counters (init)
        self.episode_fails = 0
        self.episode_len   = 0
        self.index = 0

    # ---------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = self.np_random.integers(0, len(self.X) - 1)

        # reset episode counters
        self.episode_fails = 0
        self.episode_len   = 0

        obs = self.X[self.index]
        return obs, {}

    # ---------------------------------------------------------------------
    def step(self, action):
        row   = self.X[self.index]
        label = int(self.y[self.index])                # 1 = fail, 0 = pass

        # named columns
        cycle_time   = row[self.obs_cols.index("cycle_time")]
        peak_press   = row[self.obs_cols.index("max_inj_pressure")]
        energy_proxy = peak_press * cycle_time

        # reward components
        r_defect = -self.w1 if label else +self.w0
        r_speed  = -0.02 * cycle_time
        r_energy = -0.01 * energy_proxy
        reward   = float(r_defect + r_speed + r_energy)

        # episode stats
        self.episode_fails += label
        self.episode_len   += 1
        fail_rate = self.episode_fails / self.episode_len

        # advance pointer (next shot)
        self.index = (self.index + 1) % len(self.X)

        info = {
            "fail_rate": fail_rate,          # <-- top-level so Monitor can log it
            "episode": {"r": reward, "l": 1} # SB3 requires r & l
        }

        terminated = True   # one shot = one episode
        truncated  = False

        # ---------- optional NG reward boost instead of buffer ---------------
        if label == 1:
            reward *= 4                    # amplify gradient signal
        # ---------------------------------------------------------------------

        terminated, truncated = True, False
        return row, reward, terminated, truncated, info   # <- row, not self.X[self.index]

