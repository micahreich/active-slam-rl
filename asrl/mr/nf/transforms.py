import torch
import numpy as np
from normflows.flows import Flow
from torch.distributions import Transform, TanhTransform, AffineTransform, ComposeTransform, transforms


class arcTanh(Flow):

    def __init__(self):
        super().__init__()
        self.eps = np.finfo(np.float32).eps.item()

    def forward(self, z):
        z_ = torch.tanh(z)
        log_det = torch.log(1 - z_.pow(2) + self.eps).sum(-1, keepdim=False)
        return z_, log_det

    @torch.jit.export
    def inverse(self, z):
        z_ = torch.atanh(z)
        log_det = -torch.log(1 - z.pow(2) + self.eps).sum(-1, keepdim=False)
        return z_, log_det


class Clip(Flow):

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, z_):
        # (Generation direaction) output must be [-1, 1] (this is defined according to the env.)
        z_ = torch.clamp(z_, -1, 1)
        return z_, torch.zeros(z_.shape[0], device=z_.device)

    @torch.jit.export
    def inverse(self, z):
        # (Density estimation direction) input must be [-1+esp, 1-esp] (prevent NAN outputs after preprocessing operation.)
        z = torch.clamp(z, -1 + self.eps, 1 - self.eps)
        return z, torch.zeros(z.shape[0], device=z.device)


class Clip(Transform):
    domain = transforms.constraints.real
    codomain = transforms.constraints.real
    bijective = True
    sign = 1

    def __init__(self, lo, hi):
        super().__init__()
        self.lo = lo
        self.hi = hi

    def _call(self, x):
        """ Forward transformation: y = s * tanh(x) + m """
        return x

    def _inverse(self, y):
        """ Inverse transformation: x = atanh((y - m) / s) """
        return torch.clip(y, self.lo, self.hi)

    def log_abs_det_jacobian(self, x, y):
        """ Log-determinant of the Jacobian """
        return torch.tensor(0)


class Preprocessing(Flow):

    def __init__(self):
        super().__init__()

        eps = 1e-6
        self.transform = ComposeTransform(
            parts=[TanhTransform(), Clip(-1 + eps, 1 - eps)])

    def forward(self, z, context=None):
        _z = self.transform(z)
        log_det = self.transform.log_abs_det_jacobian(z, _z).sum(-1,
                                                                 keepdim=False)
        return _z, log_det

    @torch.jit.export
    def inverse(self, z, context=None):
        _z = self.transform.inv(z)
        log_det = self.transform.inv.log_abs_det_jacobian(z, _z).sum(
            -1, keepdim=False)
        return _z, log_det

    @torch.jit.export
    def get_qv(self, z, context):
        z_, q = self.inverse(z, context)
        v = torch.zeros(z.shape[0], device=z.device)
        return z_, q, v
