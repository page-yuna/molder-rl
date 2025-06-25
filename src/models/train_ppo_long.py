import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.envs.injection_env import InjectionEnv
import pathlib as pl
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.monitor import Monitor


LOG_DIR = "runs/ppo_t3M_w3"
MODEL_SAVE = "models/ppo_t3M_w3"
TOTAL_STEPS = 3_000_000


# 1. wrap env for SB3
make_env = lambda: Monitor(
    InjectionEnv(),
    info_keywords=("fail_rate",))   # <-- wrap with Monitor
env = DummyVecEnv([make_env])

# 2. define PPO policy (small network)
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=5e-5,
    n_steps=1024,
    batch_size=512,
    n_epochs=8,
    policy_kwargs=dict(net_arch=[128, 64]),
    verbose=1,
    tensorboard_log=LOG_DIR,
    seed=42
)

# 3. train for a very short run (just sanity)
model.learn(total_timesteps=TOTAL_STEPS)

# 4. save
pl.Path("models").mkdir(exist_ok=True)
model.save(MODEL_SAVE)

# 5. log final reward to TB
writer = SummaryWriter(LOG_DIR) 
writer.add_scalar("timesteps", TOTAL_STEPS, 0)
writer.close()

print("Training done : model saved to " + MODEL_SAVE + ".zip")
