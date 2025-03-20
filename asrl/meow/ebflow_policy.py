import torch
from torch.distributions.transforms import TanhTransform, AffineTransform
from normflows.core import ConditionalNormalizingFlow
from normflows.distributions import BaseDistribution
from normflows.flows import Flow
from torch import nn
from typing import Optional, Tuple, List
# from normflows.nets import MLP
from asrl.meow.utils import get_device, MLP
from asrl.meow.flows import CondScaling, MaskedCondAffineFlow
from asrl.meow.transforms import Preprocessing
from asrl.meow.distributions import ConditionalDiagLinearGaussian
import inspect



def init_Flow(sigma_max, sigma_min, action_sizes, state_sizes):
    init_parameter = "zero"
    init_parameter_flow = "orthogonal"
    dropout_rate_flow = 0.1
    dropout_rate_scale = 0.0
    layer_norm_flow = True
    layer_norm_scale = False
    hidden_layers = 2
    flow_layers = 2
    hidden_sizes = 64
    scale_hidden_sizes = 256
    
    # Construct the prior distribution and the linear transformation
    prior_list = [state_sizes] + [hidden_sizes]*hidden_layers + [action_sizes]
    loc = None
    log_scale = MLP(prior_list, init=init_parameter)
    q0 = ConditionalDiagLinearGaussian(action_sizes, loc, log_scale, SIGMA_MIN=sigma_min, SIGMA_MAX=sigma_max)

    # Construct normalizing flow
    flows = []
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(action_sizes)])
    for i in range(flow_layers):
        layers_list = [action_sizes+state_sizes] + [hidden_sizes]*hidden_layers + [action_sizes]
        s = None
        t1 = MLP(layers_list, init=init_parameter_flow, dropout_rate=dropout_rate_flow, layernorm=layer_norm_flow)
        t2 = MLP(layers_list, init=init_parameter_flow, dropout_rate=dropout_rate_flow, layernorm=layer_norm_flow)
        flows += [MaskedCondAffineFlow(b, t1, s)]
        flows += [MaskedCondAffineFlow(1 - b, t2, s)]
    
    # Construct the reward shifting function
    scale_list = [state_sizes] + [scale_hidden_sizes]*hidden_layers + [1]
    learnable_scale_1 = MLP(scale_list, init=init_parameter, dropout_rate=dropout_rate_scale, layernorm=layer_norm_scale)
    learnable_scale_2 = MLP(scale_list, init=init_parameter, dropout_rate=dropout_rate_scale, layernorm=layer_norm_scale)
    flows += [CondScaling(learnable_scale_1, learnable_scale_2)]

    # Construct the preprocessing layer
    flows += [Preprocessing()]
    return flows, q0
    
    
class EBFlowPolicy(ConditionalNormalizingFlow):
    def __init__(self,
                 q0: BaseDistribution,
                 flows: List[Flow],
                 alpha: float):
        assert hasattr(q0, 'get_qv'), "The flow's q0 must implement get_qv method"
        
        for flow in flows:
            assert hasattr(flow, 'get_qv'), "All flows must implement get_qv method"
            
        super().__init__(q0, flows)
        
        self.alpha = alpha
    
        
    def sample(self, num_samples=1, context=None, deterministic=False):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          context: Batch of conditions/context

        Returns:
          Samples, log probability
        """
        if deterministic:
            sig = inspect.signature(self.q0.forward)
            assert 'deterministic' in sig.parameters, "The q0 distribution must implement deterministic sampling"
            
            z, log_q = self.q0(num_samples, context=context, deterministic=deterministic)
        else:
            z, log_q = self.q0(num_samples, context=context)
        
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        return z, log_q
    
    def get_qv(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[float, float]:
        """
        get_qv _summary_

        Parameters
        ----------
        s : torch.Tensor
            _description_
        a : torch.Tensor
            _description_

        Returns
        -------
        Tuple[float, float]
            _description_
        
        Our policy pi(a|s) is defined as:
            pi(a|s) = p_z(z) * exp(1/α * Q(s, a)) * exp(-1/α * V(s))
        Since we factorize pi as a normalizing flow model, we have:
            Q(s, a) = α * log (
                p_z(z) * prod over reverse nonlinear flow layers T^{-1} |det J_T^{-1}(x)|
            )
                    = α * [ log(p_z(z)) + sum over reverse nonlinear flow layers log(|det J_T^{-1}(x)|) ]
        and similarly for V(s):
            V(s) = -α * log (
                prod over reverse linear flow layers T^{-1} |det J_T^{-1}(x)|
            )
                 = α * sum over reverse linear flow layers log(|det J_T^{-1}(x)|)
        """
        assert s.device == a.device, "State and action must be on the same device"
        assert s.shape[0] == a.shape[0]

        batch_size = s.shape[0]
        device = s.device        
        
        q = torch.zeros(batch_size, device=device)
        v = torch.zeros(batch_size, device=device)
        z = a
        
        for flow in self.flows[::-1]:
            z, q_, v_ = flow.get_qv(z, context=s)
            q += q_
            v += v_
            
        z, q_, v_ = self.q0.get_qv(z, context=s)
        q += q_
        v += v_
        q = q * self.alpha
        v = v * self.alpha
    
        return q[:, None], v[:, None]

    @torch.jit.export
    def get_v(self, s):
        act = torch.zeros((s.shape[0], self.q0.shape[0]), device=s.device)
        v = torch.zeros((act.shape[0]), device=act.device)
        z = act
        for flow in self.flows[::-1]:
            z, _, v_ = flow.get_qv(z, context=s)
            v += v_
        z, _, v_ = self.q0.get_qv(z, context=s)
        v += v_
        v = v * self.alpha
        return v[:, None]
    