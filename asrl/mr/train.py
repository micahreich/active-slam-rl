import os
import sys
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import argparse
import toy_envs
from agents import *
from modules import flatten_cfg, outputdir_make_and_add
import yaml


def main(cfg) -> None:
    # parse args
    cfg = flatten_cfg(cfg)  # flatten the nested Dict structure from hydra
    args = argparse.Namespace(**cfg)

    # logger init
    save_path = os.path.join('ckpts', args.env, args.algo, args.description)
    os.makedirs(save_path, exist_ok=True)
    outputdir = outputdir_make_and_add(outputdir=save_path,
                                       title=f'seed{args.seed}')
    args.save_path = outputdir
    figdir = os.path.join(args.save_path, 'figures')
    os.makedirs(figdir, exist_ok=True)

    # environment init
    env_gen = lambda: RescaleAction(gym.make(args.env), -1.0, 1.0)

    train_envs = gym.vector.SyncVectorEnv([env_gen])
    test_envs = gym.vector.SyncVectorEnv(
        [env_gen for _ in range(args.test_num)])
    args.state_size = train_envs.observation_space.shape[1]
    args.action_size = train_envs.action_space.shape[1]

    print("Args:", args)
    print("Observation space:", train_envs.observation_space)
    print("Action space:", train_envs.action_space)

    # model
    agent = MEOW(args)
    agent.train(train_envs, test_envs)


if __name__ == '__main__':
    path = "/home/dev/workspace/asrl/mr/conf/base.yaml"
    with open(path, "r") as file:
        cfg = yaml.safe_load(file)

    main(cfg)
