import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

from asrl.sac_explore import utils


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale, affine_bias, affine_scale):
        self.loc = loc
        self.scale = scale
        
        self.base_dist = pyd.Normal(loc, scale)
        transforms = [pyd.transforms.TanhTransform(),
                      pyd.transforms.AffineTransform(loc=affine_bias,
                                                     scale=affine_scale)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    def __init__(self, og_map_shape, pose_dim, action_dim,
                 og_map_embedding_size,
                 encoder_channels,
                 hidden_dim,
                 hidden_depth,
                 log_std_bounds,
                 action_bounds):
        super().__init__()
        
        self.encoder = utils.CNNEncoder(og_map_shape, channels=encoder_channels, output_dim=og_map_embedding_size)
        self.net = utils.MLP(
            input_dim=pose_dim + og_map_embedding_size,
            hidden_dim=hidden_dim,
            output_dim=2 * action_dim,
            hidden_depth=hidden_depth,
            output_mod=None
        )
        
        self.log_std_bounds = log_std_bounds
        action_low, action_high = action_bounds
        
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_high - action_low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_high + action_low) / 2.0,
                dtype=torch.float32,
            ),
        )
        
        self.apply(utils.weight_init)
    
    def forward(self, obs):
        og_map, pose = obs['og_map'], obs['pose']
        
        encoded_map = self.encoder(og_map)
        encoded_obs = torch.cat([pose, encoded_map], dim=-1)
        
        mu, log_std = self.net(encoded_obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds        
        log_std_scale = 1/2 * (log_std_max - log_std_min)
        log_std_bias = 1/2 * (log_std_max + log_std_min)
        
        log_std = torch.tanh(log_std)
        log_std = utils.scale_tanh(log_std, log_std_scale, log_std_bias)
        
        std = log_std.exp()
        
        dist = SquashedNormal(mu, std, self.action_bias, self.action_scale)
        return dist
    
    @staticmethod
    def from_dict(cfg):
        return DiagGaussianActor(
            og_map_shape=cfg['og_map_shape'],
            pose_dim=cfg['pose_dim'],
            action_dim=cfg['action_dim'],
            og_map_embedding_size=cfg['og_map_embedding_size'],
            encoder_channels=cfg['encoder_channels'],
            hidden_dim=cfg['hidden_dim'],
            hidden_depth=cfg['hidden_depth'],
            log_std_bounds=cfg['log_std_bounds'],
            action_bounds=(
                np.array(cfg['action_bounds']['low']),
                np.array(cfg['action_bounds']['high']),
            )
        )
    
        # action = dist.rsample()
        # log_prob = dist.log_prob(action)
        # log_prob = log_prob.sum(-1, keepdim=True)
        
        # action = self._scale_tanh(action, self.action_scale, self.action_bias)
        # mean = self._scale_tanh(dist.mean, self.action_scale, self.action_bias)
        
        # return action, log_prob, mean
    
    # def act(self, obs):
    #     dist = self(obs)

    #     action = dist.sample()
    #     action = self._scale_tanh(action, self.action_scale, self.action_bias)
    #     mean = self._scale_tanh(dist.mean, self.action_scale, self.action_bias)
        
    #     return action, mean


    # def log(self, logger, step):
    #     for k, v in self.outputs.items():
    #         logger.log_histogram(f'train_actor/{k}_hist', v, step)

    #     for i, m in enumerate(self.net):
    #         if type(m) == nn.Linear:
    #             logger.log_param(f'train_actor/fc{i}', m, step)