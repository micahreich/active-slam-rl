import asrl.meow
import gymnasium as gym
from tqdm import tqdm
import time
import os


env = gym.make("MultiGoal-v0")

pbar = tqdm(range(100), postfix={"status": "Initializing"})
for i in pbar:
    if 30 <= i <= 50:
        pbar.set_postfix({"status": "Updating"})
    else:
        pbar.set_postfix({"status": "Running"})
    time.sleep(0.01)
    
print(os.path.dirname(os.path.abspath(__file__)))

print(env.spec.id)

# Create a vectorized environment with 4 instances of CartPole-v1
envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])

# Access the first environment in the vectorized environment
env_0 = envs.envs[0]

print(env_0.render())