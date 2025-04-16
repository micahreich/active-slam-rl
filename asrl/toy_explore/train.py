import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import gymnasium as gym
import json

from asrl.sac_explore.logger import Logger
from matplotlib import pyplot as plt

from asrl.toy_explore.agent import Agent
from asrl.toy_explore.env import GridExploreEnv
import torch.optim as optim


gamma = 0.99
gae_lambda = 0.95
update_epochs = 5
norm_adv = True
clip_coef = 0.2
clip_vloss = True
ent_coef = 0.1
vf_coef = 0.5
learning_rate = 1e-3
num_minibatches = 4

num_envs = 10
num_steps = 128
total_timesteps = 1_000_000

batch_size = num_envs * num_steps
minibatch_size = batch_size // num_minibatches
num_iterations = total_timesteps // batch_size

grid_height = 8
grid_width = 8

def make_vector_obs(obs):
    visited = obs['visited']
    positions = obs['position']
    
    batched = visited.ndim > 1
    if not batched:
        visited = visited[None, :]
        positions = np.array([positions])[None, :]
    
    # xys = np.zeros_like(visited, dtype=np.float32)
    # xys[:, positions] = 1.0
    
    # row = positions // grid_width
    # col = positions % grid_width
    
    # x = (col + 0.5) / grid_width
    # y = ((grid_height - row - 1) + 0.5) / grid_height
    
    # xys = np.column_stack((x, y))
    # obs_out = np.concatenate((visited, xys), axis=-1).astype(np.float32)
    
    visited[:, positions] += 100.0
    obs_out = visited.astype(np.float32)
    
    if not batched:
        obs_out = obs_out[0]
    
    return obs_out


def train():
    work_dir = os.path.dirname(os.path.abspath(__file__))
    print(f'[Train] workspace: {work_dir}')
    logger = Logger(work_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def make_env(env_id):
        def thunk():
            return GridExploreEnv(
                grid_size=(grid_height, grid_width),
                max_steps=num_steps,
            )

        return thunk
    
    envs = gym.vector.SyncVectorEnv(
        [make_env(i) for i in range(num_envs)],
    )
    
    obs_dim = 2 * envs.get_attr('num_cells')[0]
    action_dim = envs.single_action_space.n
    
    agent = Agent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    print(f'[Train] obs_dim: {obs_dim}, action_dim: {action_dim}')
    
    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, num_envs) + (obs_dim,)).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = make_vector_obs(next_obs)
    
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)

    for iteration in range(1, num_iterations + 1):        
        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_obs = make_vector_obs(next_obs)
                        
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        logger.log('train/episodic_return', info['episode']['r'], global_step)
                        logger.log('train/episodic_length', info['episode']['l'], global_step)

        cum_reward = torch.sum(rewards, dim=0).mean()
        print(f'[Train] Iteration {iteration}/{num_iterations}, Cumulative Rewards: {cum_reward.item()}')
        
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + (obs_dim,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                logger.log('train/pg_loss', pg_loss.item(), global_step)
                logger.log('train/v_loss', v_loss.item(), global_step)
                logger.log('train/entropy_loss', entropy_loss.item(), global_step)
                
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
    
    # Save the model
    torch.save(agent.state_dict(), os.path.join(logger.log_dir, 'model.pth'))
    
    
def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = GridExploreEnv(
        grid_size=(grid_height, grid_width),
        max_steps=num_steps,
    )
    
    obs_dim = 2 + env.num_cells
    action_dim = env.action_space.n
        
    agent = Agent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    checkpoint_path = '/home/dev/workspace/asrl/toy_explore/runs/exp_2025-04-16_08-50-49/model.pth'
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()
        
    # Begin rollout
    obs_dict, info = env.reset()
    obs = make_vector_obs(obs_dict)
    
    done = False
    truncated = False
    i = 0

    while not done:
        i += 1
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action, log_prob, entropy, value = agent.get_action_and_value(obs_tensor)

        obs_dict, reward, done, truncated, info = env.step(action)
        obs = make_vector_obs(obs_dict)
        
        print(f'Step {i}, Action: {action.item()} (p={log_prob.exp().item()}), Reward: {reward}, Done: {done}')
        
        env.render()
        time.sleep(1/20)

    env.close()

if __name__ == "__main__":
    # train()
    evaluate()