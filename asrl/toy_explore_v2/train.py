import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import gymnasium as gym
from gymnasium import spaces

from asrl.toy_explore_v2.env import GridExploreEnvTeleport

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
            
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def train(cfg: dict, total_timesteps):
    vec_env = make_vec_env(GridExploreEnvTeleport, n_envs=32, env_kwargs={
        'grid_size': (64, 64),
        'gaussian_sigma': 5.0,
        'map_value_max': 1.0,
        'max_steps': 200,
    })

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),  
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    model = PPO("CnnPolicy", vec_env, **cfg, verbose=1, policy_kwargs=policy_kwargs)
    print(model.policy)
    
    model.learn(total_timesteps, log_interval=2, progress_bar=True)
    model.save("ppo_grid_explore_cont_cnn")


if __name__ == "__main__":
    with open("/home/dev/workspace/asrl/toy_explore_v2/config/params.yml", "r") as f:
        config = yaml.safe_load(f)
    
    # Run training
    train(config['ppo'], total_timesteps=1_000_000)