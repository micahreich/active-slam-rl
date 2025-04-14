import os
from time import time
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np

from asrl.slam_sim.sim_env import SimulationEnvironment


MAP_SIZE_PX = (220, 220)  # Size of the occupancy grid map image in pixels


class GymExploreEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}
    
    def __init__(self,
                 episode_maxlen_s=60 * 5,
                 percentage_of_map_to_explore=0.90,
                 map_name="floorplan1",
                 render_mode=None):
        super().__init__()
        
        self.episode_maxlen_s = episode_maxlen_s
        self.percentage_of_map_to_explore = percentage_of_map_to_explore

        # Define observation space; positions are normalized to [0, 1] by dividing by the occupancy grid map size
        # and the angle is normalized to [0, 1] by dividing by 2 * pi
        self.observation_space = spaces.Dict({
            "pose": spaces.Box(
                low=0.0, high=1.0, shape=(3,), dtype=np.float32
            ),
            "grid": spaces.Box(
                low=0.0, high=1.0, shape=(MAP_SIZE_PX[0], MAP_SIZE_PX[1], 1), dtype=np.float32
            ),
        })

        # Define action space as (angle, distance); angle is normalized to [-pi, pi]rad and distance are
        # limited to [0, 100]m
        self.action_space = spaces.Box(
            low=np.array([-np.pi, 0.0], dtype=np.float32),
            high=np.array([np.pi, 1e2], dtype=np.float32),
            dtype=np.float32
        )
        
        self.simulator = SimulationEnvironment(
            map_name=map_name,
            og_map_resolution = 0.1,
            omega = 1.0,
            v = 1.0,
            dt = 0.1,
            travel_cut_short_dist_m = 0.1,
            map_image_size_px = MAP_SIZE_PX
        )
        
        self.prev_free_area = 0
        self.timesteps_elapsed = 0
        self.render_mode = render_mode
        
        self.fig = None
    
    def _to_gym_observation(self, obs):
        """
        Convert the observation from the simulator to the gym format.
        """
        pose, grid = obs
        return {
            "pose": pose,
            "grid": grid
        }
    
    def reset(self, seed=None, options={'pose': None}):
        super().reset(seed=seed)
        
        self.prev_free_area = 0
        self.timesteps_elapsed = 0
        
        obs = self.simulator.reset(options['pose'])
        info = {}
        
        return self._to_gym_observation(obs), info
    
    def step(self, action):
        obs, _ = self.simulator.step(action)
        free_area = self.simulator.og_map.free_area_m2
        
        delta_area = free_area - self.prev_free_area
        map_area = self.simulator.array_map.free_area_m2
        
        self.prev_free_area = free_area
        
        reward = delta_area / map_area
        done = free_area / map_area > self.percentage_of_map_to_explore
        truncated = self.simulator.time_elapsed >= self.episode_maxlen_s
        
        return self._to_gym_observation(obs), reward, done, truncated, {}
    
    def render(self):
        if self.render_mode != "human":
            return
        
        prob_map = self.simulator.og_map.to_prob_map()
        height = self.simulator.og_map.height_px
        width = self.simulator.og_map.width_px
        res = self.simulator.og_map.resolution
        extent = [0, width * res, 0, height * res]
        
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            
            self.im = self.ax.imshow(prob_map, vmin=0, vmax=1, cmap='gray_r',
                                 interpolation='nearest', origin='upper', extent=extent)

            self.pose_circle = plt.Circle((0, 0), radius=1.0, edgecolor='black', facecolor='none')
            self.pose_line, = self.ax.plot([], [], color='black')
            self.ax.add_patch(self.pose_circle)
            self.ax.set_aspect('equal')
            self.ax.set_title("Occupancy Grid")
        else:
            self.im.set_data(prob_map)
            
        print(f"{prob_map.shape=}, {prob_map.dtype=}")
        
        # Update pose drawing
        x, y, theta = self.simulator.pose
        scale = 1.0
        self.pose_circle.center = (x, y)
        self.pose_circle.radius = scale * 0.5
        self.pose_line.set_data(
            [x, x + scale * 0.5 * np.cos(theta)],
            [y, y + scale * 0.5 * np.sin(theta)]
        )

        self.ax.set_xlim(0, width * res)
        self.ax.set_ylim(0, height * res)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        if hasattr(self, 'fig') and self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.im = None
            self.pose_circle = None
            self.pose_line = None