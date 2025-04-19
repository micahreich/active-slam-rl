import numpy as np
import time
from asrl.slam_sim.gym_env_exploration import GymExploreEnv
import time
import gymnasium as gym
import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = GymExploreEnv(
        episode_maxlen_steps=150,
        percentage_of_map_to_explore=0.95,
        map_name="box2",
        og_map_resolution=0.2,
        dt=0.2,
        og_map_shape=(1, 128, 128),
        render_mode="human",
    )
    
    # Reset the environment
    obs, info = env.reset()
    env.render()
    
    episode_reward = 0.0
    episode = 1
    
    def on_key(event):
        if event.key == 'q':
            print("Quitting...")
            env.close()
            plt.close('all')  # Close the figure window
    
    def on_click(event):
        global episode_reward, episode
        
        if event.inaxes:
            r_goal_n = 1.0 - event.ydata / env.simulator.og_map.height_m
            c_goal_n = event.xdata / env.simulator.og_map.width_m
            
            # print(f"Clicked at: ({r_goal_n:.2f}, {c_goal_n:.2f})")
            
            _, reward, done, truncated, info = env.step(np.array([r_goal_n, c_goal_n]))
            episode_reward += reward

            print(f"{episode} - Reward: {reward}, Episode Reward: {episode_reward}, Done? {done}, Truncated? {truncated}, Steps {env.simulator.timesteps_elapsed}")
            print(f"\texploration_reward: {info['exploration_reward']}, time_reward: {info['time_reward']}, pathlength_reward: {info['pathlength_reward']}")
            
            if done or truncated:
                obs, info = env.reset()
                
                episode_reward = 0.0
                episode += 1
            
            env.render()
    
    env.fig.canvas.mpl_connect('button_press_event', on_click)
    env.fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Run GUI event loop
    while env.fig is not None and plt.fignum_exists(env.fig.number):
        plt.pause(0.1)
        