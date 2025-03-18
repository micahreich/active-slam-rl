import torch
import normflows
import numpy as np

class ConditionalDiagGaussianQV(normflows.distributions.base.ConditionalDiagGaussian):
    def __init__(self, shape, context_encoder):
        super().__init__(shape, context_encoder)
        self.const = torch.tensor(-0.5 * np.prod(shape) * np.log(2 * np.pi))
    
    def forward(self, num_samples=1, context=None, deterministic=False):
        encoder_output = self.context_encoder(context)
        split_ind = encoder_output.shape[-1] // 2
        mean = encoder_output[..., :split_ind]
        log_scale = encoder_output[..., split_ind:]

        if deterministic:
            eps = torch.zeros(
                (num_samples,) + self.shape, dtype=mean.dtype, device=mean.device
            )  # Zero noise for mean sampling
        else:
            eps = torch.randn(
                (num_samples,) + self.shape, dtype=mean.dtype, device=mean.device
            )

        z = mean + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), dim=list(range(1, self.n_dim + 1))
        )

        return z, log_p
    
    @torch.jit.export
    def get_qv(self, z, context):
        encoder_output = self.context_encoder(context)
        split_ind = encoder_output.shape[-1] // 2
        mean = encoder_output[..., :split_ind]
        log_scale = encoder_output[..., split_ind:]
        std = torch.exp(log_scale)
        
        q = -torch.sum(0.5 * torch.pow((z - mean) / std, 2),
                       list(range(1, self.n_dim + 1)))
        v = self.const - torch.sum(torch.log(std), list(range(1, self.n_dim + 1)))
        
        return z, q, -v