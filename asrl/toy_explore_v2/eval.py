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

from asrl.toy_explore_v2.env import GridExploreEnvTeleport
from stable_baselines3 import PPO


if __name__ == "__main__":
    # Load from file
    model = PPO.load("/home/dev/workspace/asrl/toy_explore_v2/ppo_grid_explore_cnn.zip")

    env = GridExploreEnvTeleport()
    obs, _ = env.reset()

    done, truncated = False, False

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(1 / 30.0)

    env.close()