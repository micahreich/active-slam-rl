import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

from asrl.sac_explore import utils


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [pyd.transforms.TanhTransform(),
                      pyd.transforms.AffineTransform(loc=0.5, scale=0.5)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    def __init__(self, og_map_size, pose_dim, action_dim,
                 og_map_embedding_size=128,
                 encoder_channels=[16, 32, 64],
                 hidden_dim=256,
                 hidden_depth=2,
                 log_std_bounds=(LOG_STD_MIN, LOG_STD_MAX)):
        super().__init__()
        
        c, h, w = og_map_size
        self.encoder = utils.CNNEncoder(h, w, channels=encoder_channels, in_channels=c, output_dim=og_map_embedding_size)
        self.net = utils.MLP(
            input_dim=pose_dim + og_map_embedding_size,
            hidden_dim=hidden_dim,
            output_dim=2 * action_dim,
            hidden_depth=hidden_depth,
            output_mod=None
        )
        
        self.log_std_bounds = log_std_bounds
        
        self.apply(utils.weight_init)
    
    def forward(self, obs):
        pose, og_map = obs
        
        encoded_map = self.encoder(og_map)
        encoded_obs = torch.cat([pose, encoded_map], dim=-1)
        
        mu, log_std = self.net(encoded_obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        
        std = log_std.exp()
        
        self.outputs['mu'] = mu
        self.outputs['std'] = std
        
        dist = SquashedNormal(mu, std)
        
        return dist
    
    def sample(self, obs,
               action_low=torch.tensor([-torch.pi, 0.0]),
               action_high=torch.tensor([torch.pi, 1e2])):
        dist = self(obs)
        action = dist.rsample()
        
        return action * (action_high - action_low) + action_low

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.net):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)