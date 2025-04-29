import pprint
import numpy as np
import time
from asrl.slam_sim.gym_env_exploration import GymExploreEnv
import gymnasium as gym
import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = GymExploreEnv(
        max_steps=100,
        percentage_of_map_to_explore=0.95,
        map_name="box2",
        og_map_resolution=0.2,
        dt=0.2,
        k=10,
        og_map_shape=(1, 100, 100),
        render_mode="human",
    )
    
    # Reset the environment
    obs, info = env.reset()
    env.render()
    
    episode_reward = 0.0
    episode = 1
    input_buffer = ""

    def on_key(event):
        global episode_reward, episode, input_buffer

        if event.key == 'q':
            print("Quitting...")
            env.close()
            plt.close('all')
            return
        
        if event.key.isdigit():
            input_buffer += event.key
            print(f"Current input: {input_buffer}")
        
        elif event.key == 'enter':
            if input_buffer != "":
                try:
                    action = int(input_buffer)
                    assert 0 <= action < env.simulator.k, f"Invalid action {action}"

                    # Take a step with the selected action
                    obs, reward, done, truncated, info = env.step(action)
                    episode_reward += reward

                    print(f"Action {action} taken")
                    print(f"{episode} - Reward: {reward:.2f}, Episode Reward: {episode_reward:.2f}, Done? {done}, Truncated? {truncated}, Steps {env.simulator.timesteps_elapsed}")
                    # pprint.pprint(info)

                    if done or truncated:
                        obs, info = env.reset()
                        episode_reward = 0.0
                        episode += 1

                    env.render()

                except (ValueError, AssertionError) as e:
                    print(f"Invalid action: {e}")
                finally:
                    input_buffer = ""  # Clear input after action

    # Connect the keypress event
    env.fig.canvas.mpl_connect('key_press_event', on_key)

    # Run GUI event loop
    while env.fig is not None and plt.fignum_exists(env.fig.number):
        plt.pause(0.1)
