import os
import sys
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import argparse
from asrl.meow_og_refactored.agents.sac import SAC, Args
from asrl.meow_og_refactored.toy_envs.return_env import ReturnAfterFarExplorationEnv
from agents import *
from modules import flatten_cfg, outputdir_make_and_add
import yaml

if __name__ == "__main__":
    args = Args(
        env_id="ReturnToOrigin-v0",
        total_timesteps=20_000,
        learning_starts=5_000,
        num_envs=1,
        eval_freq=1_000,
        tau=0.005,
    )
    
    sac = SAC(
        args=args,
    )
    
    sac.train(run_name="test1")