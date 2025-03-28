from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")


def rearrange_dims(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 2:
        return tensor[:, :, None]
    if len(tensor.shape) == 3:
        return tensor[:, :, None, :]
    elif len(tensor.shape) == 4:
        return tensor[:, :, 0, :]
    else:
        raise ValueError(f"`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.")


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish

    Parameters:
        inp_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        n_groups (`int`, default `8`): Number of groups to separate the channels into.
        activation (`str`, defaults to `mish`): Name of the activation function.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        activation: str = "mish",
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.mish = get_activation(activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        intermediate_repr = self.conv1d(inputs)
        output = self.mish(intermediate_repr)
        return output


class ResidualTemporalBlock1D(nn.Module):
    """
    Residual 1D block with temporal convolutions.

    Parameters:
        inp_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        embed_dim (`int`): Embedding dimension.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        activation (`str`, defaults `mish`): It is possible to choose the right activation function.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 5,
        cond_dim: Optional[int] = None,
        init_zeros=False,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.act = nn.Mish()
        
        if init_zeros:
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)
            
        # self.conv_in = Conv1dBlock(inp_channels, out_channels, kernel_size)
        # self.conv_out = Conv1dBlock(out_channels, out_channels, kernel_size)

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()
        )
        
        if cond_dim is not None:
            self.film = nn.Linear(cond_dim, out_channels * 2)  # For FiLM layer

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs : [ batch_size x inp_channels x horizon ]

        returns:
            out : [ batch_size x out_channels x horizon ]
        """
        
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        # out = self.bn2(out)
        
        if cond is not None:
            film_params = self.film(cond)[:, :, None] # (B, out_channels * 2, 1, 1)
            gamma, beta = film_params.chunk(2, dim=1)  # Split into gamma and beta
            out = out * (1.0 + gamma) + beta
        
        return out + self.residual_conv(x)
        
        # out = self.conv_in(inputs)
        # out = self.conv_out(out)
        # return out + self.residual_conv(inputs)