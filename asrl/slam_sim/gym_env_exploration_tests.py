import numpy as np
import time
from asrl.slam_sim.gym_env_exploration import GymExploreEnv
import sys
import termios
import tty
import select
import time
import gymnasium as gym


def get_key(timeout=0.1):
    """Non-blocking keypress read (single character)"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)
        rlist, _, _ = select.select([fd], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    env = GymExploreEnv(
        episode_maxlen_s=60 * 5,
        percentage_of_map_to_explore=0.90,
        map_name="floorplan1",
        og_map_resolution=0.2,
        omega=1.0,
        v=1.0,
        dt=0.1,
        travel_cut_short_dist_m=0.1,
        og_map_shape=(1, 128, 128),
        render_mode="human",
    )
    
    # print(env.observation_space["og_map"].shape)
    # print(isinstance(env.observation_space, gym.spaces.Dict))
    
    # for x in env.observation_space:
    #     print(x)
        
    # print(env.action_space.sample())
    
    # Reset the environment
    obs, info = env.reset()
    env.render()
    
    d = 1.0
    key_to_action = {
        'k': np.array([np.pi/2, 0]),
        'l': np.array([-np.pi/2, 0]),
        'w': np.array([0, d]),
    }
    
    cumulative_reward = 0.0
    
    try:
        while True:
            key = input("")
            if key in key_to_action:
                action = key_to_action[key]
                obs, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += reward
                print(f"Reward: {reward}, Cumulative Reward: {cumulative_reward}")
                                
                if terminated or truncated:
                    obs, info = env.reset()
                    cumulative_reward = 0.0
                    
            env.render()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        env.close()