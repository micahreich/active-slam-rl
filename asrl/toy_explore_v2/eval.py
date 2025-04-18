import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from asrl.toy_explore_v2.env import GridExploreEnvTeleport


if __name__ == "__main__":
    # Load from file
    model = PPO.load("/home/dev/workspace/asrl/toy_explore_v2/ppo_grid_explore_cont_cnn.zip")

    env = GridExploreEnvTeleport()
    obs, _ = env.reset(seed=12)

    done, truncated = False, False
    ep_len = 0
    ep_reward = 0

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        # action = env.action_space.sample()
        
        obs, reward, done, truncated, info = env.step(action)
        ep_len += 1
        ep_reward += reward
        env.render()
        time.sleep(1 / 10.0)

    print(f"Episode length: {ep_len}, reward: {ep_reward}")
    env.close()