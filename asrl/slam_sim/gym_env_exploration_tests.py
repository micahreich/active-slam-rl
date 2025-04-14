import numpy as np
import time
from asrl.slam_sim.gym_env_exploration import GymExploreEnv
import sys
import termios
import tty
import select
import time


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
        render_mode="human",
    )
    
    # Reset the environment
    obs, info = env.reset()
    env.render()
    
    d = 1.0
    key_to_action = {
        'k': np.array([np.pi/2, 0]),
        'l': np.array([-np.pi/2, 0]),
        'w': np.array([0, d]),
    }
    
    try:
        while True:
            key = input("")
            if key in key_to_action:
                action = key_to_action[key]
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    obs, info = env.reset()
                    
            env.render()
            time.sleep(1.0 / 10.0)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        env.close()