import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import gymnasium as gym

from asrl.slam_sim.gym_env_exploration import GymExploreEnv

class CNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] 
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

def train(cfg, total_timesteps):
    env_kwargs = {
        'episode_maxlen_steps': cfg['episode_maxlen_steps'],
        'percentage_of_map_to_explore': cfg['percentage_of_map_to_explore'],
        'map_name': cfg['map_name'],
        'og_map_resolution': cfg['og_map_resolution'],
        'omega': cfg['omega'],
        'v': cfg['v'],
        'dt': cfg['dt'],
        'travel_cut_short_dist_m': cfg['travel_cut_short_dist_m'],
        'og_map_shape': cfg['og_map_shape'], 
    }
    
    vec_env = make_vec_env(lambda: GymExploreEnv(**env_kwargs), n_envs=16)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    policy_kwargs = dict(
        features_extractor_class=CNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    
    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        **cfg['ppo']
    )
    model.learn(total_timesteps, log_interval=10, progress_bar=True)
    model.save("ppo_slam_explore")

if __name__ == "__main__":
    with open("config/params.yml", "r") as f:
        config = yaml.safe_load(f)
    
    train(config, total_timesteps=50_000_000)