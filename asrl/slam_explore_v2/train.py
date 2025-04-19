import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import gymnasium as gym
from gymnasium import spaces

from asrl.slam_sim.gym_env_exploration import GymExploreEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_obs_channels, H, W = observation_space.shape
        n_input_channels = n_obs_channels + 2
        
        # col coordinate: 0 at left, 1 at right
        row_coords = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(1, 1, H, W)
        col_coords = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(1, 1, H, W)
        
        self.register_buffer("row_coords", row_coords)
        self.register_buffer("col_coords", col_coords)
        
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
            sample = torch.as_tensor(observation_space.sample()[None])
            sample = self.expand_observations(sample.float())

            n_flatten = self.cnn(sample).shape[1]
            
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def expand_observations(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]

        rows = self.row_coords.expand(B, -1, -1, -1)
        cols = self.col_coords.expand(B, -1, -1, -1)
        return torch.cat([observations, rows, cols], dim=1)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self.expand_observations(observations)

        return self.linear(self.cnn(observations))


def train(cfg: dict, total_timesteps):
    vec_env = make_vec_env(GymExploreEnv, n_envs=32, env_kwargs={
        'max_steps': 100,
        'percentage_of_map_to_explore': 0.95,
        'map_name': 'box2',
        'og_map_resolution': 0.2,
        'dt': 0.1,
        'og_map_shape': (100, 100),
    })

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=[128, 128],
        share_features_extractor=True,
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    model = PPO("CnnPolicy", vec_env, **cfg, verbose=1, policy_kwargs=policy_kwargs)
    print(model.policy)
    
    model.learn(total_timesteps, log_interval=1, progress_bar=True)
    model.save("ppo_slam_explore")


if __name__ == "__main__":
    with open("/home/dev/workspace/asrl/slam_explore_v2/config/params.yml", "r") as f:
        config = yaml.safe_load(f)
    
    # Run training
    train(config['ppo'], total_timesteps=1_500_000)