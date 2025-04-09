import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from asrl.slam_sim.environments.gym_env import GymSLAMEnv  
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define map and starting pose
og = np.array([
    [1,1,1,1,1],
    [1,1,0,0,1],
    [1,1,1,1,1],
    [1,0,0,1,1],
    [1,1,1,1,1],
])
start_pose = np.array([2.5, 2.5, 0.0])  # x, y, theta

# Initialize environment
env = GymSLAMEnv(occupancy_grid=og, start_pose=start_pose)

#check if the env is Gym-compatible
check_env(env, warn=True)

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save model
model.save("ppo_slam_agent")

# Test the trained policy
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
