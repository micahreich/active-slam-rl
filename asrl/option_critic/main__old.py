from copy import deepcopy

import numpy as np
import torch
from torch.distributions import Bernoulli
from asrl.option_critic import utils
from asrl.option_critic.agents.option_critic import OptionCriticAgent, actor_loss, critic_loss
import asrl.option_critic.envs
import gymnasium as gym

from asrl.option_critic.replay_buffer import ReplayBuffer
from dataclasses import dataclass

import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class OptionCriticArgs:
    num_options: int = 4
    num_env_steps: int = 4 * 100_000
    temp: float = 1.0
    eps_start: float = 1.0
    eps_min: float = 0.1
    eps_decay: float = 0.05 * num_env_steps
    max_history: int = 10000
    termination_reg: float = 0.01
    entropy_reg: float = 0.01
    update_freq: int = 4
    freeze_interval: int = 200
    tau: float = 0.005
    gamma: float = 0.99


if __name__ == "__main__":
    env = gym.make("FourRooms-v0")
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    args = OptionCriticArgs(
        num_options=4,
        num_env_steps=4e6,
        temp=1.0,
        eps_start=1.0,
        eps_min=0.1,
        eps_decay=20_000,
        max_history=10000,
        termination_reg=0.01,
        entropy_reg=0.01,
        update_freq=4,
        freeze_interval=200,
        tau=0.005,
        gamma=0.99
    )
    
    learning_rate = 5e-4
    batch_size = 32
    device = utils.get_device()

    option_critic = OptionCriticAgent(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.eps_start,
        eps_min=args.eps_min,
        eps_decay=args.eps_decay,
        device=device,
        testing=False
    )
    option_critic.to(device)
    option_critic.train()
    
    option_critic_prime = deepcopy(option_critic)
    replay_buffer = ReplayBuffer(args.max_history)
    
    optim = torch.optim.Adam(option_critic.parameters(), lr=learning_rate)
    
    env_steps = 0
    n_episodes = 0
    
    episode_len_hist = []
    
    try:
        while env_steps < args.num_env_steps:
            obs, info = env.reset()
            
            done = False
            episode_steps = 0
            episode_reward = 0
            curr_option_len = 0
            option = None
            option_termination = True
            n_option_switches = -1
            
            while not done:
                epsilon = option_critic.epsilon
                
                if option_termination:
                    option_greedy = option_critic.greedy_option(obs)
                    option_random = np.random.randint(0, args.num_options)
                    option = option_greedy if np.random.rand() > epsilon else option_random
                    
                    n_option_switches += 1
                    curr_option_len = 0
                
                assert option is not None
                action = option_critic.get_action(obs, option)
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                replay_buffer.push(obs, option, action, reward, next_obs, done)
                
                if len(replay_buffer) > batch_size:
                    data_batch = (obs, option, action, reward, next_obs, done)
                    loss = get_actor_loss(option_critic, option_critic_prime, data_batch, args=args)
                    
                    if episode_steps % args.update_freq == 0:    
                        data_batch = replay_buffer.sample(batch_size)
                        loss += get_critic_loss(option_critic, option_critic_prime, data_batch, args=args)
                    
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    
                    if episode_steps % args.freeze_interval == 0:
                        # Update target network
                        # soft_update(option_critic_prime, option_critic, args.tau)
                        option_critic_prime.load_state_dict(option_critic.state_dict())
                    
                    # soft_update(option_critic_prime, option_critic, args.tau)
                
                option_termination_prob = option_critic.get_option_termination(next_obs, option)
                option_termination = Bernoulli(option_termination_prob).sample().item()
                
                # Update global variables
                obs = next_obs
                env_steps += 1
                episode_steps += 1
                episode_reward += reward
                
                if not option_termination:
                    curr_option_len += 1
            
            n_episodes += 1
            print(f"Episode {n_episodes} finished in {episode_steps} with reward {episode_reward}; epsilon= {epsilon:.3f}; option_switches= {n_option_switches}; option length= {curr_option_len}")
            
            episode_len_hist.append(episode_steps)
    except KeyboardInterrupt:
        print("Training interrupted.")
        
    # Plotting the episode length histogram
    plt.figure(figsize=(10, 5))
    plt.plot(episode_len_hist)
    plt.title("Episode Length History")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.grid()
    plt.savefig(os.path.join(SCRIPT_DIR, "episode_length_history.png"))
                