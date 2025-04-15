import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class CNNEncoder(nn.Module):
    def __init__(self, input_shape,
                 channels=[32, 64, 64],
                 output_dim=128):
        super().__init__()
        
        in_channels, h, w = input_shape
        layers = []
        
        for i, c_out in enumerate(channels):
            c_in = channels[i-1] if i > 0 else in_channels

            layers.append(nn.Conv2d(c_in, c_out, kernel_size=5, stride=2, padding=2))
            layers.append(nn.ReLU())

        self.conv = nn.Sequential(*layers, nn.Flatten())

        # dummy forward to get conv output shape
        with torch.inference_mode():
            dummy = torch.zeros(1, *input_shape)
            conv_out_dim = self.conv(dummy).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, output_dim),
        )
            
    def forward(self, x):
        x = self.conv(x / 255.0)        
        x = self.fc(x)
        return x


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


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def scale_tanh(x, scale, bias):
    return x * scale + bias


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    input_shape = (1, 128, 128)
    encoder = CNNEncoder(input_shape)
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(encoder)
    print(f"Total parameters: {total_params}")
    
    xs = 255 * torch.ones(1, *input_shape)
    out = encoder(xs)
    print(out.shape)