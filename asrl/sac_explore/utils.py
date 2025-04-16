import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def nan_check_hook(name):
    def hook(module, input, output):
        if torch.isnan(output).any():
            print(f"🚨 NaNs in {name}")
    return hook


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNEncoder(nn.Module):
    def __init__(self, input_shape,
                 channels,
                 kernel_sizes,
                 output_dim):
        super().__init__()
        
        in_channels, h, w = input_shape
        layers = []

        # for i, c_out in enumerate(channels):
        #     c_in = channels[i-1] if i > 0 else in_channels

        #     # ksize = kernel_sizes[i]
        #     # stride = ksize // 2
        #     # padding = ksize // 2
            
        #     layers.append(nn.Conv2d(c_in, c_out, kernel_size=ksize, stride=stride, padding=padding))
        #     layers.append(nn.Mish())
        layers = [
            layer_init(nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            layer_init(nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
        ]
        self.conv = nn.Sequential(*layers, nn.Flatten())

        # dummy forward to get conv output shape
        with torch.inference_mode():
            dummy = torch.zeros(1, *input_shape)
            conv_out_dim = self.conv(dummy).shape[1]
        
        # self.norm = nn.LayerNorm(conv_out_dim)
        
        self.fc = nn.Sequential(
            layer_init(nn.Linear(conv_out_dim, output_dim)),
        )
        
    def forward(self, x):        
        assert not torch.any(torch.isnan(x)), "input x contains NaN values"
        
        x = (x - 127.5) / 127.5
        
        assert not torch.any(torch.isnan(x)), "input x normalized contains NaN values"
        assert x.device == next(self.parameters()).device, "Device mismatch!"
        
        y = self.conv(x)
        # y = x
        # for i, layer in enumerate(self.conv):
        #     y = layer(y)
        #     print(f"after layer {i} ({layer.__class__.__name__}): {y.shape}")
        
        err_msg = f"conv output contains NaN values [{x.min()=}, {x.max()=}, {x.dtype=}, {x.device=}]"
        assert not torch.any(torch.isnan(y)), err_msg
        
        # y = self.norm(y)
        
        # assert not torch.any(torch.isnan(y)), "norm output contains NaN values"
        
        y = self.fc(y)
        
        assert not torch.any(torch.isnan(y)), "fc output contains NaN values"
        
        return y


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

        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)
        
    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.Mish(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.Mish(inplace=True)]
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
    pass
    # """Custom weight init for Conv2D and Linear layers."""
    # if isinstance(m, nn.Linear):
    #     nn.init.xavier_normal_(m.weight.data)
    #     if hasattr(m.bias, 'data'):
    #         m.bias.data.fill_(0.0)


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
    encoder = CNNEncoder(input_shape,
                         channels=[32, 64, 64],
                         kernel_sizes=[8, 4, 4],
                         output_dim=256)
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(encoder)
    print(f"Total parameters: {total_params}")
    
    xs = 255 * torch.ones(1, *input_shape)
    out = encoder(xs)
    print(out.shape)