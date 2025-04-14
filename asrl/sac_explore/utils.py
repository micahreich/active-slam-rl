import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Utility to scale tanh outputs to a range
def scale_action(raw_action, low, high):
    return 0.5 * (high - low) * (raw_action + 1.0) + low

class CNNEncoder(nn.Module):
    def __init__(self, h, w, channels=[16, 32, 64], in_channels=1, output_dim=128, ksize=3):
        super().__init__()
        
        layers = []
        
        for i in range(len(channels) - 1):
            c_in = channels[i - 1] if i > 0 else in_channels
            c_out = channels[i]
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=ksize, stride=2, padding=ksize//2))
            layers.append(nn.ReLU())
            
            h = (h - ksize + 2 * (ksize // 2)) // 2 + 1
            w = (w - ksize + 2 * (ksize // 2)) // 2 + 1
        
        layers += [
            nn.Flatten(),
            nn.Linear(channels[-1] * (h * w), output_dim),
            nn.ReLU()
        ]
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        return self.net(x)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)