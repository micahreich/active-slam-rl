import torch
from torch import nn
import normflows as nf
import math

from asrl.realnvp2d.nf.masks import create_mid_channel_split_mask, create_temporal_checkerboard_mask
from asrl.realnvp2d.nf.resnet import ResidualTemporalBlock1D
from asrl.realnvp2d.nf.squeeze import SqueezeTemporal


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def init_realnvp2d(n_channels, seq_len, n_layers):
    assert is_power_of_two(seq_len) and seq_len > 4
    
    print(f"Initializing RealNVP2D with {n_layers} layers and {n_channels} channels for a sequence of length {seq_len}")
        
    n_splits = int(math.log2(seq_len // 4))
    
    latent_shape = (n_channels, seq_len)
    flows_reversed = []
    
    for i in range(n_splits):
        for j in range(n_layers):
            s = ResidualTemporalBlock1D(inp_channels=n_channels, out_channels=n_channels, kernel_size=5, init_zeros=True)
            t = ResidualTemporalBlock1D(inp_channels=n_channels, out_channels=n_channels, kernel_size=5, init_zeros=True)
            
            even = (j+i) % 2 == 0
            b = create_temporal_checkerboard_mask(seq_len, n_channels, even=even)

            print(f"Level {i}, Layer {j}; mask shape={b.shape}, even={even} (checkerboard)")
            
            flows_reversed += [nf.flows.MaskedAffineFlow(b, t, s)]
        
        if i < n_splits - 1:
            flows_reversed += [SqueezeTemporal()]
            n_channels *= 2
            seq_len //= 2
            
            latent_shape = (n_channels, seq_len)
            
            for j in range(n_layers):
                s = ResidualTemporalBlock1D(inp_channels=n_channels, out_channels=n_channels, kernel_size=5, init_zeros=True)
                t = ResidualTemporalBlock1D(inp_channels=n_channels, out_channels=n_channels, kernel_size=5, init_zeros=True)
                
                even = (j+i+1) % 2 == 0
                b = create_mid_channel_split_mask(seq_len, n_channels, even=even)
                
                print(f"Level {i}, Layer {j}; mask shape={b.shape}, even={even} (mid channel split)")
            
                flows_reversed += [nf.flows.MaskedAffineFlow(b, t, s)]
    
    q0 = nf.distributions.DiagGaussian(latent_shape)
    model = nf.NormalizingFlow(q0=q0, flows=reversed(flows_reversed))
    
    return model


if __name__ == "__main__":
    n_channels = 3
    seq_len = 128
    
    model = init_realnvp2d(n_channels=n_channels, seq_len=seq_len, n_layers=3)
    x = torch.randn(32, n_channels, seq_len)
    
    z = model.inverse(x)
    xbar = model.forward(z)
    
    print(f"x shape: {x.shape}")
    print(f"z shape: {z.shape}")
    print(f"xbar shape: {xbar.shape}")
    
    assert torch.allclose(x, xbar, atol=1e-5)
    print("Success!")