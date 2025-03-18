import torch
from torch.distributions.transforms import TanhTransform, AffineTransform
from normflows.core import ConditionalNormalizingFlow
from normflows.distributions import BaseDistribution
from normflows.flows import Flow
from torch import nn
from typing import Optional, Tuple, List
# from normflows.nets import MLP
from asrl.meow.utils import get_device, MLP
from asrl.meow.flows import MaskedConditionalAffineFlow, TransformFlow, ConditionalRewardShifting
from asrl.meow.distributions import ConditionalDiagGaussianQV
import inspect



def init_flow(state_dim: int,
              action_dim: int,
              action_range: Tuple[torch.Tensor, torch.Tensor],
              flow_layers: int = 2,
              context_encoder_hidden_sizes: int = 256,
              flow_hidden_sizes: int = 256,
              reward_shift_hidden_sizes: int = 256,
              hidden_layers: int = 2,
              dropout_rate_flow: float = 0.1,
              dropout_rate_reward_shift: float = 0.1,
              layer_norm_flow: bool = True,
              layer_norm_reward_shift: bool = True,
              init_parameter_context_encoder="orthogonal",
              init_parameter_flow="orthogonal",
              init_parameter_reward_shift="zero",
              device="cpu"):
    
    action_range = (torch.as_tensor(action_range[0]), torch.as_tensor(action_range[1]))
    assert (action_range[1] > action_range[0]).all(), "Action range must be valid, i.e. upper bound must be greater than lower bound in all dims"
    
    # Construct the prior distribution q0(z | s)
    prior_context_encoder_layer_dims = [state_dim] + [context_encoder_hidden_sizes]*hidden_layers + [2 * action_dim]
    prior_context_encoder = MLP(layers=prior_context_encoder_layer_dims,
                                init=init_parameter_context_encoder)
    q0 = ConditionalDiagGaussianQV(shape=(action_dim,), context_encoder=prior_context_encoder)
    
    # Construct the normalizing flow, x = T(u)
    flows = []
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(action_dim)])
    
    flow_st_encoder_layer_dims = [action_dim+state_dim] + [flow_hidden_sizes]*hidden_layers + [action_dim]
    
    for i in range(flow_layers):
        s = None
        t1 = MLP(flow_st_encoder_layer_dims,
                 dropout_rate=dropout_rate_flow,
                 init=init_parameter_flow,
                 layernorm=layer_norm_flow)
        t2 = MLP(flow_st_encoder_layer_dims,
                 dropout_rate=dropout_rate_flow,
                 init=init_parameter_flow,
                 layernorm=layer_norm_flow)
        flows += [MaskedConditionalAffineFlow(b, t1, s)]
        flows += [MaskedConditionalAffineFlow(1 - b, t2, s)]
    
    # Construct the reward shifting functions
    scale_layers_dim_list = [state_dim] + [reward_shift_hidden_sizes]*hidden_layers + [1]
    learnable_reward_shift_1 = MLP(scale_layers_dim_list,
                            dropout_rate=dropout_rate_reward_shift,
                            init=init_parameter_reward_shift,
                            layernorm=layer_norm_reward_shift)
    
    learnable_reward_shift_2 = MLP(scale_layers_dim_list,
                            dropout_rate=dropout_rate_reward_shift,
                            init=init_parameter_reward_shift,
                            layernorm=layer_norm_reward_shift)    

    flows += [ConditionalRewardShifting(learnable_reward_shift_1, learnable_reward_shift_2)]

    # Normalize the output to stay in action range: https://www.desmos.com/calculator/4boustmxgf
    affine_tf_width = action_range[1] - action_range[0]
    affine_loc = (1/2 * affine_tf_width + action_range[0]).to(device)
    affine_scale = (1/2 * affine_tf_width).to(device)
    
    flows += [TransformFlow(transform=TanhTransform()),
              TransformFlow(transform=AffineTransform(loc=affine_loc,
                                                      scale=affine_scale))]
    
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
    