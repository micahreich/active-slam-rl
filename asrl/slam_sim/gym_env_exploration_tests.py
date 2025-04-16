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
        episode_maxlen_steps=150,
        percentage_of_map_to_explore=0.95,
        map_name="box2",
        og_map_resolution=0.2,
        omega=1.0,
        v=1.0,
        dt=0.2,
        travel_cut_short_dist_m=0.1,
        og_map_shape=(1, 128, 128),
        render_mode="human",
    )
    
    # Reset the environment
    obs, info = env.reset()
    env.render()
    
    d = 2.0
    key_to_action = {
        'w': np.array([np.pi/2, d]),
        'a': np.array([np.pi, d]),
        's': np.array([-np.pi/2, d]),
        'd': np.array([0.0, d]),
    }
    
    episode_reward = 0.0
    episode = 1
    
    try:
        while True:
            key = input("")
            if key in key_to_action:
                action = key_to_action[key]
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                print(f"{episode} - Reward: {reward}, Episode Reward: {episode_reward}")

                if terminated or truncated:
                    print(f"\tEpisode {episode} finished; terminated? {terminated}, truncated? {truncated}, envsteps: {env.simulator.envsteps_elapsed}")
                    obs, info = env.reset()
                    episode_reward = 0.0
                    episode += 1
                    
            env.render()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        env.close()