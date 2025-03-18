import torch
import normflows
from typing import Optional, Union
from asrl.meow.utils import MLP
from torch.distributions.transforms import Transform, TanhTransform, AffineTransform


class MaskedConditionalAffineFlow(normflows.flows.Flow):
    def __init__(self, b: torch.Tensor, t=None, s=None):
        """
        __init__ Masked conditional affine flow layer from https://arxiv.org/pdf/1605.08803

        Parameters
        ----------
        b : torch.Tensor
            Mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
        t : torch.Tensor, optional
            Translation mapping, i.e. neural network, where first input dimension is batch dim, if None no translation is applied, by default None
        s : torch.Tensor, optional
            Scale mapping, i.e. neural network, where first input dimension is batch dim, if None no scale is applied, by default None
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer("b", self.b_cpu)
        self.s = s
        self.t = t

    def get_st(self, z_masked, context):
        tmp = torch.cat([z_masked, context], dim=1)
        scale = self.s(tmp) if self.s is not None else torch.zeros_like(z_masked)
        trans = self.t(tmp) if self.t is not None else torch.zeros_like(z_masked)
        return scale, trans

    def forward(self, z, context):
        z_masked = self.b * z
        scale, trans = self.get_st(z_masked, context)
        # print("forward call of masked affine layer")
        # print(f"  scale shape: {scale.shape}, trans shape: {trans.shape}, z_masked shape: {z_masked.shape}, b shape: {self.b.shape}")
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

    def inverse(self, z, context):
        z_masked = self.b * z
        scale, trans = self.get_st(z_masked, context)
        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

    @torch.jit.export
    def get_qv(self, z, context):
        z_, log_det = self.inverse(z, context)
        q = log_det
        v = torch.zeros(z.shape[0], device=z.device)
        return z_, q, v


class ConditionalRewardShifting(normflows.flows.Flow):
    def __init__(self, s1: MLP, s2: Optional[MLP] = None):
        """
        __init__ Conditional scaling layer to implement learned reward shifting

        Parameters
        ----------
        s1 : MLP
            MLP which outputs learned scaling factor for the first half of the context
        s2 : MLP, optional
            MLP which outputs learned scaling factor for the second half of the context, by default None;
            to be used when you have a shared-trunk Q-function architecture
        
        This is mostly for convenience, as it allows for the LRS term to be added to the Q, V values
        using the same interface as the flows
        """
        super().__init__()
        self.scale1 = s1
        self.scale2 = s2

    def forward(self, z, context):
        log_det = torch.zeros(z.shape[0], device=z.device)
        return z, log_det
    
    @torch.jit.export
    def inverse(self, z, context):
        log_det = torch.zeros(z.shape[0], device=z.device)
        return z, log_det

    @torch.jit.export
    def get_qv(self, z, context):
        if self.scale2 is not None:
            s1 = self.scale1(context[:context.shape[0]//2])
            s2 = self.scale2(context[:context.shape[0]//2])
            q = torch.cat((s1[:, 0], s2[:, 0]), dim=0)
            v = torch.cat((s1[:, 0], s2[:, 0]), dim=0)
        else:
            s1 = self.scale1(context)
            q = s1[:, 0]
            v = s1[:, 0]
        return z, q, v


class TransformFlow(normflows.flows.Flow):
    def __init__(self, transform: Transform):
        super().__init__()
        self.transform = transform
    
    def forward(self, z, context=None):
        h = self.transform(z)
        log_det = self.transform.log_abs_det_jacobian(z, h).sum(dim=-1, keepdims=False)
        return h, log_det
    
    def inverse(self, z, context=None):
        if type(self.transform) == TanhTransform:
            # Clamp to avoid numerical issues with arctanh, stay within (-1, 1)
            eps = 1e-5
            z = torch.clamp(z, -1 + eps, 1 - eps)
            
        h = self.transform.inv(z)
        log_det = -self.transform.log_abs_det_jacobian(h, z).sum(dim=-1, keepdims=False)
        return h, log_det
    
    @torch.jit.export
    def get_qv(self, z, context):
        z_, log_det = self.inverse(z, context)
        q = log_det
        v = torch.zeros(z.shape[0], device=z.device)
        return z_, q, v


# class NormalizeAction(normflows.flows.Flow):
#     def __init__(self, a_min: torch.Tensor, a_max: torch.Tensor):
#         super().__init__()
        
#         assert (a_max > a_min).all(), "a_min must be less than a_max"
        
#         self.tanh_transform = TanhTransform()
#         self.affine_transform = AffineTransform(loc=a_min, scale=a_max - a_min)
    
#     def forward(self, z, context=None):
#         log_det = torch.zeros(z.shape[0], device=z.device)
        
#         h = self.tanh_transform(z)
#         log_det += self.tanh_transform.log_abs_det_jacobian(z, h) 
#         z = h
        
#         h = self.affine_transform(z)
#         log_det += self.affine_transform.log_abs_det_jacobian(z, h)
#         z = h
        
#         return z, log_det
    
#     def inverse(self, z, context=None):
#         log_det = torch.zeros(z.shape[0], device=z.device)
        
#         h = self.affine_transform.inv(z)
#         log_det -= self.affine_transform.log_abs_det_jacobian(h, z)
#         z = h
        
#         h = self.tanh_transform.inv(z)
#         log_det -= self.tanh_transform.log_abs_det_jacobian(h, z)
#         z = h
        
#         return z, log_det
    
#     @torch.jit.export
#     def get_qv(self, z, context):
#         z_, log_det = self.inverse(z, context)
#         q = torch.zeros(z.shape[0], device=z.device)
#         v = torch.zeros(z.shape[0], device=z.device)
#         return z_, q, v