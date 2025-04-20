import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime

from asrl.slam_sim.gym_env_exploration import GymExploreEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor, VecNormalize
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
        
        # CNN backbone
        self.cnn = nn.Sequential(
            # Block1:  in_ch→64  
            nn.Conv2d(n_input_channels,  32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),  # ↓2
            nn.BatchNorm2d(32), nn.ReLU(),

            # Block2: 64→128
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),  # ↓2
            nn.BatchNorm2d(64), nn.ReLU(),

            # Dilated “wide” context 
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64), nn.ReLU(),

            # Global summary
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, n_obs_channels, H, W)
            dummy = self._add_coordinates(dummy)
            n_flatten = self.cnn(dummy).shape[1]
            
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def _add_coordinates(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        rows = self.row_coords.expand(B, -1, -1, -1)
        cols = self.col_coords.expand(B, -1, -1, -1)
        return torch.cat([obs, rows, cols], dim=1)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self._add_coordinates(observations)
        x = self.cnn(x)
        return self.linear(x)


def train(cfg: dict, total_timesteps):
    vec_env = make_vec_env(
        GymExploreEnv,
        n_envs=32,
        env_kwargs={
            'max_steps': 1000,
            'percentage_of_map_to_explore': 0.5,
            'map_name': 'box2',
            'og_map_resolution': 0.2,
            'dt': 0.1,
            'og_map_shape': (100, 100),
        })
    
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,       # normalize observations to mean=0, std=1
        norm_reward=True,    # normalize rewards to mean=0, std=1
        clip_reward=10.0,
        gamma=cfg['gamma']
    )

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=[256, 256],
        share_features_extractor=True,
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    model = PPO("CnnPolicy", vec_env, **cfg, verbose=1, policy_kwargs=policy_kwargs)
    total_params = sum(p.numel() for p in model.policy.features_extractor.parameters() if p.requires_grad)
    
    print(model.policy)
    print(total_params)
    
    model.learn(total_timesteps, log_interval=1, progress_bar=True)
    
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    model.save(f"{timestamp_str}/ppo_slam_explore")
    vec_env.save(f"{timestamp_str}/vecnormalize.pkl")


if __name__ == "__main__":
    with open("/home/dev/workspace/asrl/slam_explore_v2/config/params.yml", "r") as f:
        config = yaml.safe_load(f)
    
    # Run training
    train(config['ppo'], total_timesteps=1_500_000)