import os
import time
import numpy as np
from asrl.sac_explore import Agent
from asrl.sac_explore import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import gymnasium as gym
import json

from asrl.sac_explore.logger import Logger
from asrl.sac_explore.replay_buffer import ReplayBuffer
from asrl.sac_explore.sac import SACAgent
from asrl.slam_sim.gym_env_exploration import GymExploreEnv
from matplotlib import pyplot as plt



class Workspace(object):
    def __init__(self, cfg, device):
        self.device = device
        self.work_dir = os.path.dirname(os.path.abspath(__file__))
        print(f'[Train] workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(self.work_dir)

        self.env = GymExploreEnv.from_dict(cfg['environment'])
        # self.env.render_mode = 'human'
        
        self.agent = SACAgent(cfg['sac'], self.device)
        print(f'[Train] agent # params: {self.agent.count_parameters()}')

        if isinstance(self.env.observation_space, gym.spaces.Dict):
            obs_shape = {
                key: self.env.observation_space[key].shape for key in self.env.observation_space
            }
        else:
            obs_shape = self.env.observation_space.shape
        
        self.replay_buffer = ReplayBuffer(obs_shape,
                                          self.env.action_space.shape,
                                          int(cfg['train']['replay_buffer_capacity']),
                                          self.device)
        
        self.step = 0
        self.episode_num = 0
        
    def evaluate(self):
        average_episode_reward = 0
        num_episodes = self.cfg['train']['num_eval_episodes']
        
        ending_maps = []
        traveled_poses = []
        ending_rewards = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            self.agent.reset()

            done = False
            episode_reward = 0
            poses = []
            
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = float(terminated or truncated)
                episode_reward += reward
                
                pose = obs['pose'] * np.array([self.env.simulator.og_map.width_m,
                                               self.env.simulator.og_map.height_m,
                                               2*np.pi])
                curr_map = obs['og_map']
                
                poses.append(pose)

            average_episode_reward += episode_reward
            
            ending_maps.append(curr_map)
            traveled_poses.append(np.array(poses))
            ending_rewards.append(episode_reward)
            
        average_episode_reward /= num_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        
        fig, ax = plt.subplots(int(np.ceil(num_episodes / 2)), 2, squeeze=False)
        
        for i in range(num_episodes):
            prob_map = ending_maps[i]
            poses = traveled_poses[i]
            
            row, col = divmod(i, 2)
            height = self.env.simulator.og_map.height_px
            width = self.env.simulator.og_map.width_px
            res = self.env.simulator.og_map.resolution
            extent = [0, width * res, 0, height * res]
            
            ax[row, col].imshow(prob_map[0], vmin=0, vmax=255, cmap='gray_r',
                                interpolation='nearest', origin='upper', extent=extent)
            ax[row, col].scatter(poses[:, 0], poses[:, 1], c='r', s=1)
            ax[row, col].set_title(f'Episode {i} - Reward: {ending_rewards[i]:.3f}')
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.logger.log_dir, f'eval_{self.step}.png'))
            
                
    def train(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        
        while self.step < self.cfg['train']['num_train_steps']:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    
                    print(f'[Train] step: {self.step}, episode: {episode}, reward: {episode_reward}')
                    self.episode_num += 1

                # # evaluate agent periodically
                # if self.episode_num % self.cfg['train']['eval_frequency'] == 0:
                #     self.logger.log('eval/episode', episode, self.step)
                #     self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)

                obs, _ = self.env.reset()
                self.agent.reset()
                
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
            
            # sample action for data collection
            if self.step < self.cfg['train']['num_seed_steps']:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg['train']['num_seed_steps']:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, terminations, truncations, infos = self.env.step(action)
            self.env.render()

            # allow infinite bootstrap
            done = float(terminations or truncations)
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_step += 1
            self.step += 1


if __name__ == '__main__':
    with open("/home/dev/workspace/asrl/sac_explore/config/params.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        print(json.dumps(cfg, indent=4))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    workspace = Workspace(cfg, device)
    workspace.train()