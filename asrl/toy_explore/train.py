import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from asrl.toy_explore.env import GridExploreEnv


def train(cfg: dict, total_timesteps):
    vec_env = make_vec_env(GridExploreEnv, n_envs=32, env_kwargs={
        'grid_size': (8, 8),
        'max_steps': 200
    })

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    model = PPO("MlpPolicy", vec_env, **cfg, verbose=1, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps, log_interval=10, progress_bar=True)
    model.save("ppo_grid_explore")

if __name__ == "__main__":
    with open("/home/dev/workspace/asrl/toy_explore/config/params.yml", "r") as f:
        config = yaml.safe_load(f)
    
    # Run training
    train(config['ppo'],
          total_timesteps=1_000_000)