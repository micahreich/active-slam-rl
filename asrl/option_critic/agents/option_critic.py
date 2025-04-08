import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

from math import exp
import numpy as np

from utils import to_tensor


class OptionCriticFeatures(nn.Module):
    def __init__(self,
                in_features,
                num_actions,
                num_options,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.1,
                eps_decay=int(1e5),
                ent_start=100,
                ent_min=0.1,
                ent_decay=int(1e5),
                eps_test=0.05,
                device='cpu',
                testing=False):

        super(OptionCriticFeatures, self).__init__()

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
        
        self.ent_start = ent_start
        self.ent_min   = ent_min
        self.ent_decay = ent_decay
        
        self.num_steps = 0
        
        self.features = nn.Sequential(
            nn.Linear(in_features, 60),
            nn.ReLU(),
            nn.Linear(60, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU()
        )


        # Option-value head (policy over options)
        self.Q = nn.Linear(200, num_options)

        # Option-termination head
        self.terminations = nn.Linear(200, num_options)
        
        self.options_W = nn.Parameter(torch.zeros(num_options, 200, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))
        
        torch.nn.init.xavier_uniform_(self.options_W)
        torch.nn.init.zeros_(self.options_b)

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)
    
    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()
    
    def get_terminations(self, state):
        return self.terminations(state).sigmoid() 

    def get_action(self, state, option):        
        logits = state.data @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy
    
    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if self.testing:
            return self.eps_test

        # Linear decay: eps = max(eps_min, eps_start - (step / decay) * (eps_start - eps_min))
        fraction = min(self.num_steps / self.eps_decay, 1.0)
        eps = self.eps_start + fraction * (self.eps_min - self.eps_start)
        self.num_steps += 1
        return eps

    @property
    def entropy_coeff(self):
        # Linear decay: coeff = max(end, start - (step / decay) * (start - end))
        fraction = min(self.num_steps / self.ent_decay, 1.0)
        coeff = self.ent_start + fraction * (self.ent_min - self.ent_start)
        return coeff
        

def critic_loss(model, model_prime, data_batch, args):
    obs, options, rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    options   = torch.LongTensor(options).to(model.device)
    rewards   = torch.FloatTensor(rewards).to(model.device)
    masks     = 1 - torch.FloatTensor(dones).to(model.device)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    states = model.get_state(to_tensor(obs)).squeeze(0)
    Q      = model.get_Q(states)

    
    # the update target contains Q_next, but for stable learning we use prime network for this
    next_states_prime = model_prime.get_state(to_tensor(next_obs)).squeeze(0)
    next_Q_prime      = model_prime.get_Q(next_states_prime).detach() # detach?

    # Additionally, we need the beta probabilities of the next state
    next_states            = model.get_state(to_tensor(next_obs)).squeeze(0)
    next_termination_probs = model.get_terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]
    
    next_Q = model.get_Q(next_states).detach()  # use online model for selection
    best_next_options = next_Q.argmax(dim=-1)
    next_Q_prime_vals = next_Q_prime[batch_idx, best_next_options]

    # Now we can calculate the update target gt
    gt = rewards + masks * args.gamma * \
        ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob  * next_Q_prime_vals)

    # to update Q we want to use the actual network, not the prime
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    return td_err

def actor_loss(obs, option, logp, entropy, entropy_coeff, reward, done, next_obs, model, model_prime, args):
    state = model.get_state(to_tensor(obs))
    next_state = model.get_state(to_tensor(next_obs))
    next_state_prime = model_prime.get_state(to_tensor(next_obs))

    option_term_prob = model.get_terminations(state)[:, option]
    next_option_term_prob = model.get_terminations(next_state)[:, option].detach()

    Q = model.get_Q(state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()

    # Target update gt
    gt = reward + (1 - done) * args.gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob  * next_Q_prime.max(dim=-1)[0])

    # The termination loss
    termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() + args.termination_reg) * (1 - done)
    
    # actor-critic policy gradient with entropy regularization
    policy_loss = -logp * (gt.detach() - Q[option]) - entropy_coeff * entropy
    actor_loss = termination_loss + policy_loss
    return actor_loss

def soft_update(target_net, online_net, tau=0.005):
    for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)