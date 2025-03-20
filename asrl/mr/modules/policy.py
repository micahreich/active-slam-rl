import torch
from torch import nn
from asrl.mr.modules.utils import convert_to_shape_tuple
from nf.nets import MLP
from nf.transforms import Preprocessing
from nf.distributions import ConditionalDiagLinearGaussian, ContextEncoderDiagGaussian
from nf.flows import MaskedCondAffineFlow, CondRewardShift, Flow
from normflows import ConditionalNormalizingFlow
from typing import Tuple, List


def init_flow(args) -> Tuple[List[Flow], ConditionalDiagLinearGaussian]:
    # init_parameter = "zero"
    # init_parameter_flow = "orthogonal"
    # dropout_rate_flow = 0.1
    # dropout_rate_scale = 0.0
    # layer_norm_flow = True
    # layer_norm_scale = False
    # hidden_layers = 2
    # flow_layers = 2
    # hidden_size = 64
    # scale_hidden_size = 256

    # Construct the prior distribution and the linear transformation
    prior_list = [args.state_size] + \
                 [args.n_hidden_units_context_encoder] * args.n_hidden_layers_context_encoder + \
                 [args.action_size]

    loc = None
    log_scale = MLP(prior_list,
                    init=args.init_parameters_context_encoder,
                    layernorm=args.layernorm_context_encoder)
    context_encoder = ContextEncoderDiagGaussian(args.action_size, log_scale,
                                                 loc)
    q0 = ConditionalDiagLinearGaussian(args.action_size, context_encoder,
                                       args.sigma_min, args.sigma_max)

    # Construct normalizing flow
    flows = []
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(args.action_size)])
    for i in range(args.n_flow_layers):
        layers_list = [args.action_size + args.state_size] + \
                      [args.n_hidden_units_flow] * args.n_hidden_layers_flow + \
                      [args.action_size]
        s = None
        t1 = MLP(layers_list,
                 init=args.init_parameters_flow,
                 dropout_rate=args.dropout_rate_flow,
                 layernorm=args.layernorm_flow)
        t2 = MLP(layers_list,
                 init=args.init_parameters_flow,
                 dropout_rate=args.dropout_rate_flow,
                 layernorm=args.layernorm_flow)
        flows += [MaskedCondAffineFlow(b, t1, s)]
        flows += [MaskedCondAffineFlow(1 - b, t2, s)]

    # Construct the reward shifting function
    scale_list = [args.state_size] + \
                 [args.n_hidden_units_reward_shift] * args.n_hidden_layers_reward_shift + \
                 [1]

    learnable_scale_1 = MLP(scale_list,
                            init=args.init_parameters_reward_shift,
                            dropout_rate=args.dropout_rate_reward_shift,
                            layernorm=args.layernorm_reward_shift)
    learnable_scale_2 = MLP(scale_list,
                            init=args.init_parameters_reward_shift,
                            dropout_rate=args.dropout_rate_reward_shift,
                            layernorm=args.layernorm_reward_shift)
    flows += [CondRewardShift(learnable_scale_1, learnable_scale_2)]

    # Construct the preprocessing layer
    flows += [Preprocessing()]
    return flows, q0


class FlowPolicy(ConditionalNormalizingFlow):

    def __init__(self, args):
        flows, q0 = init_flow(args)
        super().__init__(q0, flows)

        self.device = args.device
        self.alpha = args.alpha
        self.action_shape = convert_to_shape_tuple(args.action_size)

        self.flows.to(self.device)
        self.q0.to(self.device)

    def sample(self, num_samples, context, deterministic=False):
        context = torch.as_tensor(context,
                                  dtype=torch.float32,
                                  device=self.device)

        if deterministic:
            # (Warning: This is only implemented for MEow with the additive coupling layers and the Gaussian prior)
            # (This is due to the fact that Gaussian + additive coupling layers have the property the MLE of the prior,
            # when transformed, is the MLE of the posterior)
            a, _ = self.q0.get_mean_std(context)
            log_q = self.q0.log_prob(a, context)

            a, log_det = self.forward_and_log_det(a, context)
            log_q -= log_det
        else:
            a, log_q = super().sample(num_samples, context)

        return a, log_q

    @torch.jit.export
    def get_qv(self, obs, act):
        N = obs.shape[0]
        q = torch.zeros((N), device=act.device)
        v = torch.zeros((N), device=act.device)
        z = act
        for flow in reversed(self.flows):
            z, q_, v_ = flow.get_qv(z, context=obs)
            q += q_
            v += v_
        q_, v_ = self.q0.get_qv(z, context=obs)
        q += q_
        v += v_
        q = q * self.alpha
        v = v * self.alpha
        return q[:, None], v[:, None]

    @torch.jit.export
    def get_v(self, obs):
        N = obs.shape[0]
        act = torch.zeros((N, ) + self.action_shape, device=self.device)
        v = torch.zeros((N), device=act.device)
        z = act
        for flow in reversed(self.flows):
            z, _, v_ = flow.get_qv(z, context=obs)
            v += v_
        _, v_ = self.q0.get_qv(z, context=obs)
        v += v_
        v = v * self.alpha
        return v[:, None]


# class FlowPolicyOLD(nn.Module):

#     def __init__(self, alpha, sigma_max, sigma_min, action_sizes, state_sizes,
#                  device):
#         super().__init__()
#         self.device = device
#         self.alpha = alpha
#         self.action_shape = action_sizes
#         flows, q0 = init_flow(sigma_max, sigma_min, action_sizes, state_sizes)
#         self.flows = nn.ModuleList(flows).to(self.device)
#         self.prior = q0.to(self.device)

#     def forward(self, obs, act):
#         log_q = torch.zeros(act.shape[0], dtype=act.dtype, device=act.device)
#         z = act
#         for flow in self.flows:
#             z, log_det = flow.forward(z, context=obs)
#             log_q -= log_det
#         return z, log_q

#     @torch.jit.export
#     def inverse(self, obs, act):
#         log_q = torch.zeros(act.shape[0], dtype=act.dtype, device=act.device)
#         z = act
#         for flow in self.flows[::-1]:
#             z, log_det = flow.inverse(z, context=obs)
#             log_q += log_det
#         return z, log_q

#     @torch.jit.ignore
#     def sample(self, num_samples, obs, deterministic=False):
#         obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
#         if deterministic:  # (Warning: This is only implemented for MEow with the additive coupling layers and the Gaussian prior)
#             # eps = torch.randn((num_samples,) + self.prior.shape, dtype=obs.dtype, device=obs.device)
#             act, _ = self.prior.get_mean_std(obs)
#             log_q = self.prior.log_prob(act, context=obs)
#         else:
#             act, log_q = self.prior.forward(num_samples=num_samples,
#                                             context=obs)
#         a, log_det = self.forward(obs=obs, act=act)
#         log_q -= log_det
#         return a, log_q

#     @torch.jit.export
#     def log_prob(self, obs, act):
#         z, log_q = self.inverse(obs=obs, act=act)
#         log_q += self.prior.log_prob(z, context=obs)
#         return log_q

#     @torch.jit.export
#     def get_qv(self, obs, act):
#         q = torch.zeros((act.shape[0]), device=act.device)
#         v = torch.zeros((act.shape[0]), device=act.device)
#         z = act
#         for flow in self.flows[::-1]:
#             z, q_, v_ = flow.get_qv(z, context=obs)
#             q += q_
#             v += v_
#         q_, v_ = self.prior.get_qv(z, context=obs)
#         q += q_
#         v += v_
#         q = q * self.alpha
#         v = v * self.alpha
#         return q[:, None], v[:, None]

#     @torch.jit.export
#     def get_v(self, obs):
#         act = torch.zeros((obs.shape[0], self.action_shape),
#                           device=self.device)
#         v = torch.zeros((act.shape[0]), device=act.device)
#         z = act
#         for flow in self.flows[::-1]:
#             z, _, v_ = flow.get_qv(z, context=obs)
#             v += v_
#         _, v_ = self.prior.get_qv(z, context=obs)
#         v += v_
#         v = v * self.alpha
#         return v[:, None]
