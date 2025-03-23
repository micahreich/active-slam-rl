import torch
import torch.nn as nn
import numpy as np
from normflows.distributions.base import ConditionalDiagGaussian


class ContextEncoderDiagGaussian(nn.Module):

    def __init__(self, shape, log_scale: nn.Module, loc: nn.Module):
        super().__init__()

        if isinstance(shape, int):
            self.shape = (shape, )
        if isinstance(shape, list):
            self.shape = tuple(shape)

        self.log_scale = log_scale
        self.loc = loc

    def forward(self, context):
        B = context.shape[0]

        mean = self.loc(context) if self.loc is not None else torch.zeros(
            (B, ) + self.shape, device=context.device)
        log_scale = self.log_scale(
            context) if self.log_scale is not None else torch.zeros(
                (B, ) + self.shape, device=context.device)

        return torch.cat([mean, log_scale], dim=-1)


class ConditionalDiagLinearGaussian(ConditionalDiagGaussian):
    """
    Conditional multivariate Gaussian distribution with diagonal
    covariance matrix, parameters are obtained by a context encoder,
    context meaning the variable to condition on
    """

    def __init__(self,
                 shape,
                 context_encoder: nn.Module,
                 sigma_min=-5,
                 sigma_max=-0.3):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          context_encoder: Computes mean and log of the standard deviation
          of the Gaussian, mean is the first half of the last dimension
          of the encoder output, log of the standard deviation the second
          half
        """
        super().__init__(shape, context_encoder)

        self.const = torch.tensor(-0.5 * np.prod(shape) * np.log(2 * np.pi))
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_mean_std(self, context):
        encoder_output = self.context_encoder(context)
        split_ind = encoder_output.shape[-1] // 2
        mean = encoder_output[..., :split_ind]
        log_scale = encoder_output[..., split_ind:]
        log_scale = torch.tanh(log_scale)
        log_scale = self.sigma_min + 0.5 * (self.sigma_max -
                                            self.sigma_min) * (log_scale + 1)
        sigma = log_scale.exp()
        return mean, sigma

    def forward(self, num_samples=1, context=None):
        eps = torch.randn((num_samples, ) + self.shape,
                          dtype=context.dtype,
                          device=context.device)
        mean, std = self.get_mean_std(context)
        z = mean + std * eps
        log_p = self.const - torch.sum(
            torch.log(std) + 0.5 * torch.pow(eps, 2),
            list(range(1, self.n_dim + 1)))
        return z, log_p

    def log_prob(self, z, context):
        mean, std = self.get_mean_std(context)
        log_p = self.const - torch.sum(
            torch.log(std) + 0.5 * torch.pow((z - mean) / std, 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p

    def get_qv(self, z, context):
        mean, std = self.get_mean_std(context)
        q = -torch.sum(0.5 * torch.pow(
            (z - mean) / std, 2), list(range(1, self.n_dim + 1)))
        v = self.const - torch.sum(torch.log(std),
                                   list(range(1, self.n_dim + 1)))
        return q, -v
