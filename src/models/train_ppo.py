import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.envs.injection_env import InjectionEnv
import pathlib as pl
from torch.utils.tensorboard import SummaryWriter

# 1. wrap env for SB3
make_env = lambda: InjectionEnv()
env = DummyVecEnv([make_env])

# 2. define PPO policy (small network)
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=256,
    n_epochs=5,
    policy_kwargs=dict(net_arch=[128, 64]),
    verbose=1,
    tensorboard_log="runs/ppo_baseline",
    seed=42
)

# 3. train for a very short run (just sanity)
model.learn(total_timesteps=5_000)

# 4. save
pl.Path("models").mkdir(exist_ok=True)
model.save("models/ppo_t5000")

# 5. log final reward to TB
writer = SummaryWriter("runs/ppo_baseline")
writer.add_scalar("timesteps", 5_000, 0)
writer.close()

print("Training done – model saved to models/ppo_t5000.zip")
