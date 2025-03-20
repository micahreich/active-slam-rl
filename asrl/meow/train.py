from asrl.meow.buffer import ReplayBuffer
from asrl.meow.meow_agent import MEOWAgent
from asrl.meow.multi_goal_env import MultiGoal
from asrl.meow.training_loop import MEOWTrainingConfig
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


def main(config: MEOWTrainingConfig):
    device = get_device()
    
    env = gym.make("MultiGoal-v0")
    
    nx = env.observation_space.shape[0]
    nu = env.action_space.shape[0]
    
    print(f"State space dimension: {nx}")
    print(f"Action space dimension: {nu}")
    print(env.action_space.low, env.action_space.high)
    
    agent = MEOWAgent(
        nx=nx,
        nu=nu,
        alpha=2.5,
        gamma=0.9,
        action_range=(env.action_space.low, env.action_space.high),
        device=device,
        config=config,
    )
    
    agent.train(env)
    

if __name__ == "__main__":
    main(config=MEOWTrainingConfig(
        eval_every=500,
        save=True,
        grad_clip=30.0
    ))