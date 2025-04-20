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
from asrl.slam_sim.gym_env_exploration import GymExploreEnv

import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load from file
    model = PPO.load("/home/dev/workspace/asrl/slam_explore_v2/2025-04-20_04-03-50/ppo_slam_explore.zip")
    env_kwargs={
        'max_steps': 500,
        'percentage_of_map_to_explore': 0.5,
        'map_name': 'box2',
        'og_map_resolution': 0.2,
        'dt': 0.1,
        'og_map_shape': (100, 100),
    }
    
    env = GymExploreEnv(**env_kwargs, render_mode='human')
    obs, _ = env.reset(seed=1337)
    env.render()
    
    done, truncated = False, False
    ep_len = 0
    ep_reward = 0

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        # action = np.random.uniform(0, 1, size=(2,))
        
        obs, reward, done, truncated, info = env.step(action)
        ep_len += 1
        ep_reward += reward
        
        env.render()
        time.sleep(1 / 2.0)

    print(f"Episode length: {ep_len}, reward: {ep_reward}")
    env.close()