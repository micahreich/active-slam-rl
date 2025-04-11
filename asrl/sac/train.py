import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import toy_envs
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm


def evaluate(
    model: BaseAlgorithm,
    num_episodes: int = 10,
    deterministic: bool = True,
) -> float:
    """
    Evaluate an RL agent for `num_episodes`.

    :param model: the RL Agent
    :param env: the gym Environment
    :param num_episodes: number of episodes to evaluate it
    :param deterministic: Whether to use deterministic or stochastic actions
    :return: Mean reward for the last `num_episodes`
    """
    # This function will only work for a single environment
    vec_env = model.get_env()
    obs = vec_env.reset()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        # Note: SB3 VecEnv resets automatically:
        # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
        # obs = vec_env.reset()
        
        positions = []
                
        while not done:
            # _states are only useful when using LSTM policies
            # `deterministic` is to use deterministic actions
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, _info = vec_env.step(action)
            positions.append(obs[0])
            
            episode_rewards.append(reward)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        goal_positions = np.array(
            (
                (5, 0),
                (-5, 0),
                (0, 5),
                (0, -5)
            ),
            dtype=np.float32)
        print("plotting")
        
        ax.scatter(goal_positions[:, 0], goal_positions[:, 1], c='red', label='Goals')
        positions = np.array(positions)
        ax.plot(positions[:-1, 0], positions[:-1, 1], marker='o', markersize=2, label='Agent Path')
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
        ax.set_title('Agent Path and Goals')
        ax.legend()
        plt.savefig(f"agent_path_{i}.png")
        
        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print(f"Mean reward: {mean_episode_reward:.2f} - Num episodes: {num_episodes}")

    return mean_episode_reward

env = gym.make("MultiGoal-v0")

model = SAC("MlpPolicy", env, verbose=1)

mean_reward_before_train = evaluate(model, num_episodes=5, deterministic=False)

model.learn(total_timesteps=5_000)

mean_reward_before_train = evaluate(model, num_episodes=5, deterministic=False)

# vec_env = model.get_env()
# obs = vec_env.reset()

# for i in range(10_000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     # vec_env.render()
#     # VecEnv resets automatically
#     # if done:
#     #   obs = env.reset()
    


env.close()