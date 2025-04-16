import numpy as np
from asrl.sac_explore import Agent
from asrl.sac_explore import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from asrl.sac_explore.actor import DiagGaussianActor
from asrl.sac_explore.critic import DoubleQCritic


class SACAgent(Agent):
    def __init__(self, cfg, device):
        super().__init__()

        self.device = torch.device(device)
        self.discount = cfg["discount"]
        self.critic_tau = cfg["critic_tau"]
        self.actor_update_frequency = cfg["actor_update_frequency"]
        self.critic_target_update_frequency = cfg["critic_target_update_frequency"]
        self.batch_size = cfg["batch_size"]
        self.learnable_temperature = cfg["learnable_temperature"]

        self.critic = DoubleQCritic.from_dict(cfg["critic"]).to(self.device)

        self.critic_target = DoubleQCritic.from_dict(cfg["critic"]).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor.from_dict(cfg["actor"]).to(self.device)

        self.log_alpha = torch.tensor(np.log(cfg["init_temperature"])).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -cfg["action_dim"]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"],)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"],)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg["alpha_lr"],)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
    
    def count_parameters(self):
        return utils.count_parameters(self.actor) + utils.count_parameters(self.critic)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):        
        obs = {
            "og_map": torch.FloatTensor(obs["og_map"]).to(self.device).unsqueeze(0),
            "pose": torch.FloatTensor(obs["pose"]).to(self.device).unsqueeze(0)
        }
        
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)
            target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        
        logger.log('train_critic/loss', critic_loss, step)
        logger.log('train_critic/target_Q', target_Q.mean(), step)
        logger.log('train_critic/current_Q1', current_Q1.mean(), step)
        logger.log('train_critic/current_Q2', current_Q2.mean(), step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # if self.learnable_temperature:
        #     alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        #     logger.log('train_alpha/loss', alpha_loss, step)
        #     logger.log('train_alpha/value', self.alpha, step)

        #     self.log_alpha_optimizer.zero_grad()
        #     alpha_loss.backward()
        #     self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)
        
        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)