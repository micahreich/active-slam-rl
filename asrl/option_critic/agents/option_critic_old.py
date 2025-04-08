import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

from math import exp
import numpy as np

from utils import to_tensor


class OptionCriticAgent(nn.Module):
    def __init__(self,
                 in_features,
                 num_actions,
                 num_options,
                 temperature=1.0,
                 eps_start=1.0,
                 eps_min=0.1,
                 eps_decay=int(1e6),
                 eps_test=0.05,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 testing=False):
        super(OptionCriticAgent, self).__init__()
        
        self.in_features = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0
        
        self.num_embedding_features = 64
        
        self.state_embedding = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        
        # Q_\Omega(s, \omega) modeled as MLP which takes state as input and
        # outputs Q-values for each option
        
        self.Q = nn.Sequential(
            # nn.Linear(self.num_embedding_features, self.num_embedding_features),
            # nn.ReLU(),
            nn.Linear(64, num_options)
        )
        
        # \Beta_(s, \omega) modeled as MLP which takes state as input and
        # outputs termination probabilities for each option
        
        self.terminations = nn.Sequential(
            # nn.Linear(self.num_embedding_features, self.num_embedding_features),
            # nn.ReLU(),
            nn.Linear(64, num_options),
            # nn.Sigmoid()
        )
        
        # \pi_\omega(s) modeled as MLP which takes state as input and
        # outputs action probabilities for each option
        # self.pi_omega = nn.Sequential(
        #     nn.Linear(self.num_embedding_features + num_options, self.num_embedding_features),
        #     nn.ReLU(),
        #     nn.Linear(self.num_embedding_features, num_options),
        # )
        
        self.options_W = nn.Parameter(torch.zeros(num_options, self.num_embedding_features, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

    def get_state_embedding(self, obs):
        return self.state_embedding(obs)
    
    def get_Q(self, state_emb):
        return self.Q(state_emb)
    
    def get_Beta(self, state_emb):
        return self.Beta(state_emb)
    
    def get_action_dist(self, state_emb, option):
        # option_one_hot = F.one_hot(option, num_classes=self.num_options).float()
        # state_option = torch.cat((state_emb, option_one_hot), dim=-1)
        
        logits = state_emb @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)
        
        return action_dist
    
    # def get_option_termination(self, obs: np.ndarray, option: int):
    #     obs = to_tensor(obs, device=self.device)
    #     state_emb = self.get_state_embedding(obs)
        
    #     termination_probs = self.get_Beta(state_emb)
    #     option_termination_prob = termination_probs[option]
        
    #     return option_termination_prob.item()
    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()
    
    def get_action(self, obs: np.ndarray, option: int):
        obs = to_tensor(obs, device=self.device)
        state_emb = self.get_state_embedding(obs)
        option = to_tensor(option, device=self.device, dtype=torch.long)
        
        action_dist = self.get_action_dist(state_emb, option)
        action = action_dist.sample()
        
        return action.item()
    
    def greedy_option(self, obs: np.ndarray):
        obs = to_tensor(obs, device=self.device)        
        state_emb = self.get_state_embedding(obs)
        
        Qvals = self.get_Q(state_emb)
        Qvals_argmax = Qvals.argmax(dim=-1)
        
        return Qvals_argmax.item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps


def get_critic_loss(model: OptionCriticAgent, model_prime: OptionCriticAgent, data_batch, args):
    obs, options, actions, rewards, next_obs, dones = data_batch
    
    obs = to_tensor(obs, device=model.device)
    options = to_tensor(options, device=model.device, dtype=torch.long)
    rewards = to_tensor(rewards, device=model.device)
    next_obs = to_tensor(next_obs, device=model.device)
    dones = to_tensor(dones, device=model.device)
    
    masks = 1 - dones
    
    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    state_embs = model.get_state_embedding(obs)    
    Q = model.get_Q(state_embs)
    
    next_states_embs_prime = model_prime.get_state_embedding(next_obs)
    next_Q_prime = model_prime.get_Q(next_states_embs_prime)
    
    # Additionally, we need the beta probabilities of the next state
    next_states_embs = model.get_state_embedding(next_obs)
    next_options_term_prob = model.get_Beta(next_states_embs)[:, options]
    
    # Now we can calculate the update target gt
    gt = rewards + masks * args.gamma * \
        ((1 - next_options_term_prob) * next_Q_prime[:, options] + next_options_term_prob  * next_Q_prime.amax(dim=-1))

    # to update Q we want to use the actual network, not the prime
    td_err = F.mse_loss(Q[:, options], gt.detach())
    
    return td_err


def get_actor_loss(model: OptionCriticAgent, model_prime: OptionCriticAgent, data_batch, args):
    obs, options, actions, rewards, next_obs, dones = data_batch
    
    obs = to_tensor(obs, device=model.device).unsqueeze(0)
    options = to_tensor(options, device=model.device, dtype=torch.long).unsqueeze(0)
    actions = to_tensor(actions, device=model.device, dtype=torch.long).unsqueeze(0)
    rewards = to_tensor(rewards, device=model.device).unsqueeze(0)
    next_obs = to_tensor(next_obs, device=model.device).unsqueeze(0)
    dones = to_tensor(dones, device=model.device).unsqueeze(0)
    
    masks = 1 - dones
    
    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    state_embs = model.get_state_embedding(obs)
    next_state_embs = model.get_state_embedding(next_obs)
    Q = model.get_Q(state_embs)
    
    next_state_embs_prime = model_prime.get_state_embedding(next_obs)
    next_Q_prime = model_prime.get_Q(next_state_embs_prime)
    
    options_term_prob = model.get_Beta(state_embs)[:, options]
    next_options_term_prob = model.get_Beta(next_state_embs)[:, options]
    
    # Calculate the policy gradient for the intra-option policy
    gt = rewards + masks * args.gamma * \
        ((1 - next_options_term_prob) * next_Q_prime[:, options] + next_options_term_prob * next_Q_prime.amax(dim=-1))
    
    action_dists = model.get_action_dist(state_embs, options)
    action_logprobs = action_dists.log_prob(actions)
    action_entropies = action_dists.entropy()
    
    policy_loss = -action_logprobs * (gt - Q[:, options]).detach() #- args.entropy_reg * action_entropies
    
    # Calculate the policy gradient for the termination function
    termination_loss = options_term_prob * (
        (Q[:, options] - Q.amax(dim=-1)).detach() + args.termination_reg
    ) * masks
    
    # Combine the policy loss and termination loss
    actor_loss = policy_loss + termination_loss
    # actor_loss = actor_loss.sum()
    
    return actor_loss

def soft_update(target_net, source_net, tau):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)