import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from asrl.sac_explore import utils


class QCritic(nn.Module):
    def __init__(self, og_map_size, pose_dim, action_dim,
                 og_map_embedding_size=128,
                 encoder_channels=[16, 32, 64],
                 hidden_dim=256,
                 hidden_depth=2):
        super().__init__()
        
        c, h, w = og_map_size
        self.encoder = utils.CNNEncoder(h, w, channels=encoder_channels, in_channels=c, output_dim=og_map_embedding_size)
        self.net = utils.MLP(
            input_dim=pose_dim + og_map_embedding_size + action_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            hidden_depth=hidden_depth,
            output_mod=None
        )
    
    def forward(self, obs, action):
        pose, og_map = obs
        
        encoded_map = self.encoder(og_map)
        encoded_obs_action = torch.cat([pose, encoded_map, action], dim=-1)
        q = self.net(encoded_obs_action)

        return q

class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, og_map_size, pose_dim, action_dim,
                 og_map_embedding_size=128,
                 encoder_channels=[16, 32, 64],
                 hidden_dim=256,
                 hidden_depth=2):
        super().__init__()
        
        self.Q1 = QCritic(og_map_size, pose_dim, action_dim,
                          og_map_embedding_size=og_map_embedding_size,
                          encoder_channels=encoder_channels,
                          hidden_dim=hidden_dim,
                          hidden_depth=hidden_depth)
        
        self.Q2 = QCritic(og_map_size, pose_dim, action_dim,
                          og_map_embedding_size=og_map_embedding_size,
                          encoder_channels=encoder_channels,
                          hidden_dim=hidden_dim,
                          hidden_depth=hidden_depth)
        
        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)