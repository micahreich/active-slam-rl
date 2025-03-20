from asrl.meow.buffer import ReplayBuffer
from asrl.meow.multi_goal_env import MultiGoal
from asrl.meow.utils import get_device, torch_to_numpy
import gymnasium as gym
import numpy as np
import torch
import random
from collections import deque
from dataclasses import dataclass, field
from tqdm import tqdm
import os
from datetime import datetime
from typing import Optional


@dataclass
class MEOWTrainingConfig:
    replay_buffer_size: int = field(default=1000000)
    tau: float = field(default=0.0005)
    learning_rate: float = field(default=1e-4)
    batch_size: int = field(default=256)
    gym_seed: int = field(default=0)
    n_train_env_steps: int = field(default=5000)
    n_train_env_warmup_steps: int = field(default=1000)
    eval_every: int = field(default=1000)
    n_eval_envs: int = field(default=10)
    save: bool = field(default=False)
    model_name: Optional[str] = field(default=None)
    plot_every: Optional[int] = field(default=None)
    grad_clip: Optional[float] = field(default=None)
    
    def create_save_dirs(self):
        fpath = os.path.dirname(os.path.abspath(__file__))
        formatted_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        if self.model_name is None:
            self.model_name = f"MEOW_{formatted_time}"
        else:
            self.model_name = f"{self.model_name}_{formatted_time}"
        
        self.save_model_dir = os.path.join(fpath, self.model_name)
        self.save_figures_dir = os.path.join(self.save_model_dir, 'figures')
        
        os.makedirs(self.save_model_dir, exist_ok=True)
        os.makedirs(self.save_figures_dir, exist_ok=True)
        
    def __post_init__(self):
        assert self.replay_buffer_size > 0, "replay_buffer_size must be greater than 0"
        assert 1 > self.tau > 0, "tau must be greater than 0"
        assert self.learning_rate > 0, "learning_rate must be greater than 0"
        assert self.batch_size > 0, "batch_size must be greater than 0"
        assert self.gym_seed >= 0, "gym_seed must be non-negative"
        assert self.n_train_env_steps > 0, "n_train_env_steps must be greater than 0"
        assert self.n_train_env_warmup_steps >= 0, "n_train_env_warmup_steps must be non-negative"
        
        if self.plot_every is not None:
            assert self.plot_every % self.eval_every == 0, "plot_every must be a multiple of eval_every"

def evaluate(agent: "MEOWAgent", envs: gym.vector.VectorEnv, plot_save_fpath, deterministic=True):
    is_training = agent.policy.training
    agent.policy.eval()
    
    with torch.no_grad():
        num_envs = envs.unwrapped.num_envs
        rewards = np.zeros((num_envs,))
        dones = np.zeros((num_envs,)).astype(bool)
        s, _ = envs.reset(seed=range(num_envs))
        
        s_hist = [[] for _ in range(num_envs)]
        
        while not all(dones):
            a, _ = agent.get_action(s, deterministic=deterministic)
            a = torch_to_numpy(a)
            
            s_, r, terminated, truncated, _ = envs.step(a)
            done = terminated | truncated
            rewards += r * (1-dones)
            dones |= done
            s = s_
            
            for i in range(num_envs):
                if not dones[i]:
                    s_hist[i].append(s[i])
    
    envs.envs[0].unwrapped.render_rollouts(s_hist, file_name=plot_save_fpath)
    
    if is_training:
        agent.policy.train()
            
    return rewards.mean()


def training_loop(agent: "MEOWAgent", env: gym.Env, config: MEOWTrainingConfig,
                  eval_envs: gym.Env=None):
    if config.save:
        config.create_save_dirs()
    
    buffer = ReplayBuffer(config.replay_buffer_size, agent.nx, agent.nu)
    
    if config.eval_every > 0:
        assert eval_envs is not None, "eval_env must be provided if eval_every is set"
    
    best_test_return = -np.inf
    best_train_return = -np.inf
    episode_return = 0
    episode = 0
    obs, _ = env.reset(seed=config.gym_seed)
    
    pbar = tqdm(range(config.n_train_env_steps+1), postfix={"status": "Initializing"})
    
    for t in pbar:
        if t < config.n_train_env_warmup_steps:
            act = env.action_space.sample()
        else:
            agent.policy.eval()
            act, _ = agent.get_action(obs[None, :])
            act = torch_to_numpy(act.squeeze(0))
        
        next_obs, reward, terminated, truncated, info = env.step(act)
        buffer.store(
            s=obs,
            a=act,
            r=reward,
            s_=next_obs,
            d=terminated*1.0
        )
        episode_return += reward
        obs = next_obs
        
        if t >= config.n_train_env_warmup_steps:
            batch = buffer.sample(config.batch_size, agent.device)
            agent.update(batch)
            
            if t % config.eval_every == 0:
                agent.policy.eval()
                test_return = evaluate(agent, eval_envs,
                                       plot_save_fpath=os.path.join(config.save_figures_dir, f"{t}_eval"))
                
                pbar.set_postfix({
                    "test_return": f"{test_return:.3f}",
                })
                
                if test_return > best_test_return:
                    best_test_return = test_return
                    
                    if config.save:
                        torch.save(agent.policy, os.path.join(config.save_model_dir, f"best_policy.pt"))
                
            # if config.plot_every and t % config.plot_every == 0:
            #     env.render_rollouts()
                
            #     figdir = os.path.join(config.save_path, 'figures', str(t))
            #     _, _ = plot_traj_multigoal(agent.policy, figdir+'_traj')
            #     plot_value(agent.policy, figdir+'_value')
            #     torch.save(agent.policy, os.path.join(args.save_path, 'best.pt'))
        
        if terminated or truncated:
            best_train_return = max(best_train_return, episode_return)
            
            episode_return = 0
            episode += 1
    
    print("Training complete.")
    
    if config.save:
        torch.save(agent.policy, os.path.join(config.save_model_dir, f"final_policy.pt"))