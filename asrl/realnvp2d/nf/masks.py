import torch

def create_temporal_checkerboard_mask(seq_len, n_channels, even=True) -> torch.Tensor:
    mask_zeros = torch.zeros(seq_len)
    mask_zeros[even::2] = 1
    
    mask_zeros = mask_zeros.repeat(n_channels, 1)
    assert mask_zeros.shape == (n_channels, seq_len)
    
    return mask_zeros

def create_mid_channel_split_mask(seq_len, n_channels, even=True) -> torch.Tensor:
    assert n_channels % 2 == 0, "Number of channels must be even for mid channel split mask"
    
    mask_zeros = torch.zeros((n_channels, seq_len))
    
    halves = [
        torch.arange(n_channels//2),
        torch.arange(n_channels//2, n_channels)
    ]
    mask_zeros[halves[even], :] = 1
    
    return mask_zeros

if __name__ == "__main__":
    mask = create_temporal_checkerboard_mask(10, 3, even=True)
    print(mask)
    mask = create_temporal_checkerboard_mask(10, 3, even=False)
    print(mask)
    
    mask = create_mid_channel_split_mask(10, 4, even=True)
    print(mask)
    mask = create_mid_channel_split_mask(10, 4, even=False)
    print(mask)