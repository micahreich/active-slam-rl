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

from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        image_shape = observation_space.spaces["image"].shape  # (2, H, W)
        frontiers_shape = observation_space.spaces["frontiers"].shape  # (k, 2)
        n_obs_channels, H, W = image_shape
        n_input_channels = n_obs_channels + 2  # adding row/col coords

        self.k = frontiers_shape[0]

        # Coord channels
        row_coords = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(1, 1, H, W).clone()
        col_coords = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(1, 1, H, W).clone()
        
        self.register_buffer("row_coords", row_coords)
        self.register_buffer("col_coords", col_coords)

        # CNN for image + coords
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=2),
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

        # FC for frontiers
        self.frontier_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.k * 2, 128),
            nn.ReLU()
        )

        # compute output dims
        with torch.no_grad():
            sample_image = torch.as_tensor(observation_space.spaces["image"].sample()[None]).float()
            sample_image = self.expand_observations(sample_image)
            cnn_out_dim = self.cnn(sample_image).shape[1]
            frontier_out_dim = self.frontier_fc(torch.as_tensor(observation_space.spaces["frontiers"].sample()[None]).float()).shape[1]

        total_out_dim = cnn_out_dim + frontier_out_dim
        self.final = nn.Sequential(
            nn.Linear(total_out_dim, features_dim),
            nn.ReLU()
        )

    def expand_observations(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        rows = self.row_coords.expand(B, -1, -1, -1)
        cols = self.col_coords.expand(B, -1, -1, -1)
        return torch.cat([image, rows, cols], dim=1)

    def forward(self, observations: dict) -> torch.Tensor:
        image = observations["image"]
        frontiers = observations["frontiers"]

        x_img = self.cnn(self.expand_observations(image))
        x_frontiers = self.frontier_fc(frontiers)

        return self.final(torch.cat([x_img, x_frontiers], dim=1))


def train(cfg: dict, total_timesteps):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = "/home/dev/workspace/asrl/slam_explore_v3"
    tb_dir = f"{path}/{now}/tensorboard"
    
    env = GymExploreEnv(
        max_steps=1000,
        percentage_of_map_to_explore=0.95,
        map_name='box2',
        og_map_resolution=0.2,
        dt=0.1,
        k=20,
        og_map_shape=(100, 100)
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    model = DQN("MultiInputPolicy", env, **cfg,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tb_dir,
                device="cuda")

    new_logger = configure(tb_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    try:
        model.learn(total_timesteps, log_interval=100, progress_bar=True)
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model...")
    finally:
        model.save(f"{path}/{now}/dqn_slam_explore")

        with open(f'{path}/{now}/params.yml', 'w') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    with open("/home/dev/workspace/asrl/slam_explore_v3/config/params.yml", "r") as f:
        config = yaml.safe_load(f)
    
    # Run training
    train(config['dqn'], total_timesteps=500_000)