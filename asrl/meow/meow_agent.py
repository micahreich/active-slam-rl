from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from asrl.meow.ebflow_policy import EBFlowPolicy, init_flow
from asrl.meow.buffer import ReplayBuffer
import gymnasium as gym
from asrl.meow.training_loop import MEOWTrainingConfig, training_loop
import numpy as np


class MEOWAgent:
    def __init__(self,
                 nx,
                 nu,
                 alpha,
                 gamma,
                 action_range,
                 device,
                 config: MEOWTrainingConfig):
        
        self.nx = nx
        self.nu = nu
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.config = config
        
        flows, q0 = init_flow(
            state_dim=nx,
            action_dim=nu,
            action_range=action_range,
            device=self.device
        )
        
        policy = EBFlowPolicy(q0=q0, flows=flows, alpha=self.alpha)
        
        policy.flows.to(self.device)
        policy.q0.to(self.device)
        
        self.policy, self.policy_old = policy, deepcopy(policy)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.loss = nn.MSELoss()
        # self.buffer = ReplayBuffer(self.config.replay_buffer_size, nx, nu)
    
    def train(self, env: gym.Env):
        eval_envs = gym.make_vec(env.spec.id, num_envs=self.config.n_eval_envs)
        return training_loop(self, env, self.config, eval_envs)
    
    def get_action(self, s: np.ndarray, deterministic=False):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        num_samples = s.shape[0]

        # print(f"get action s shape: {s.shape} ({num_samples} samples)")
                
        a, log_q = self.policy.sample(num_samples, s, deterministic=deterministic)
        return a, log_q
    
    def _update_q(self, batch):
        s, a, r, s_n, d = batch
        
        # Compute the target Q values
        with torch.no_grad():
            self.policy.eval()
            
            v_old = self.policy_old.get_v(torch.cat((s_n, s_n), dim=0))
            exact_v_old = torch.min(v_old[:v_old.shape[0]//2], v_old[v_old.shape[0]//2:])
            target_q = r + (1-d) * self.gamma * exact_v_old
        
        
        self.policy.train()
        
        # Update the Q function with policy iteration
        current_q1, _ = self.policy.get_qv(torch.cat((s, s), dim=0), torch.cat((a, a), dim=0))
        target_q = torch.cat((target_q, target_q), dim=0)
        
        # print(f"current_q1: {current_q1.shape}, target_q: {target_q.shape}")
        
        c1_loss = F.mse_loss(current_q1, target_q)
        # c1_loss[c1_loss != c1_loss] = 0.0
        # c1_loss = c1_loss.mean()
        
        self.optim.zero_grad(set_to_none=True)
        c1_loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
            
        self.optim.step()
    
    @staticmethod
    def _soft_update(tgt: nn.Module, src: nn.Module, tau=0.005):
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1-tau) * tgt_param.data)
    
    def update(self, batch):
        # Sample batch from replay buffer
        s, a, r, s_n, d = batch['states'], batch['actions'], batch['rewards'], batch['next_states'], batch['dones']
        
        # Update policy
        self._update_q(batch=(s, a, r, s_n, d))
        self._soft_update(self.policy_old, self.policy, tau=self.config.tau)
