import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

from asrl.sac_explore import utils


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
                 encoder_kernel_sizes,
                 hidden_dim,
                 hidden_depth,
                 log_std_bounds,
                 action_bounds):
        super().__init__()
        
        self.encoder = utils.CNNEncoder(og_map_shape,
                                        channels=encoder_channels,
                                        kernel_sizes=encoder_kernel_sizes,
                                        output_dim=og_map_embedding_size)
        self.net = utils.MLP(
            input_dim=pose_dim + og_map_embedding_size,
            hidden_dim=hidden_dim,
            output_dim=2 * action_dim,
            hidden_depth=hidden_depth,
            output_mod=None
        )
        
        # nn.init.zeros_(self.net[-1].weight)
        # nn.init.zeros_(self.net[-1].bias)
        
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
            
    def forward(self, obs):
        og_map, pose = obs['og_map'], obs['pose']
        
        assert not torch.any(torch.isnan(og_map)), "og_map contains NaN values"
        assert not torch.any(torch.isnan(pose)), "pose contains NaN values"
        assert not torch.any(torch.isinf(og_map)), "og_map contains Inf values"
        assert not torch.any(torch.isinf(pose)), "pose contains Inf values"
        
        encoded_map = self.encoder(og_map)
        
        assert not torch.any(torch.isnan(encoded_map)), "encoded_map has NaNs"
        
        encoded_obs = torch.cat([encoded_map, pose], dim=-1)
        
        mu, log_std = self.net(encoded_obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds        
        log_std_scale = 1/2 * (log_std_max - log_std_min)
        log_std_bias = 1/2 * (log_std_max + log_std_min)
        
        log_std = torch.tanh(log_std)
        log_std = utils.scale_tanh(log_std, log_std_scale, log_std_bias)
        
        std = log_std.exp()
        
        assert not torch.any(torch.isnan(mu)), "mu contains NaN values"
        assert not torch.any(torch.isnan(std)), "std contains NaN values"
        assert not torch.any(torch.isinf(mu)), "mu contains Inf values"
        assert not torch.any(torch.isinf(std)), "std contains Inf values"
        
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
            encoder_kernel_sizes=cfg['encoder_kernel_sizes'],
            hidden_dim=cfg['hidden_dim'],
            hidden_depth=cfg['hidden_depth'],
            log_std_bounds=cfg['log_std_bounds'],
            action_bounds=(
                np.array(cfg['action_bounds']['low']),
                np.array(cfg['action_bounds']['high']),
            )
        )


if __name__ == "__main__":
    actor = DiagGaussianActor(
        og_map_shape=[1, 128, 128],
        pose_dim=3,
        action_dim=2,
        og_map_embedding_size=128,
        encoder_channels=[32, 64, 64],
        encoder_kernel_sizes=[8, 4, 4],
        hidden_dim=128,
        hidden_depth=2,
        log_std_bounds=(-20, 2),
        action_bounds=(
            np.array([-np.pi, 0.0]),
            np.array([np.pi, 10.0])
        )
    )
    
    print(actor)
    
    dummy_obs = {
        'og_map': torch.randn(1, 1, 128, 128),
        'pose': torch.randn(1, 3)
    }
    
    dist = actor(dummy_obs)