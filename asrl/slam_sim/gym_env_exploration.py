import os
from time import time
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np

from asrl.slam_sim.sim_env import SimulationEnvironment


class GymExploreEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}
    
    def __init__(self,
                 episode_maxlen_steps,
                 percentage_of_map_to_explore,
                 map_name,
                 og_map_resolution,
                 omega,
                 v,
                 dt,
                 travel_cut_short_dist_m,
                 og_map_shape,
                 render_mode=None):
        super().__init__()
        
        self.episode_maxlen_steps = episode_maxlen_steps
        self.percentage_of_map_to_explore = percentage_of_map_to_explore

        # Define observation space; positions are normalized to [0, 1] by dividing by the occupancy grid map size
        # and the angle is normalized to [0, 1] by dividing by 2 * pi
        self.observation_space = spaces.Dict({
            "og_map": spaces.Box(
                low=0.0, high=1.0, shape=og_map_shape, dtype=np.float32
            ),
            "pose": spaces.Box(
                low=0.0, high=1.0, shape=(3,), dtype=np.float32
            ),
        })

        # Define action space as (angle, distance); angle is normalized to [-pi, pi]rad and distance are
        # limited to [0, max]m
        self.action_space = spaces.Box(
            low=np.array([-np.pi, 0.0], dtype=np.float32),
            high=np.array([np.pi, 10.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.simulator = SimulationEnvironment(
            map_name,
            og_map_resolution,
            omega,
            v,
            dt,
            travel_cut_short_dist_m,
            og_map_shape
        )
        
        self.prev_free_area = 0
        self.timesteps_elapsed = 0
        self.render_mode = render_mode
        
        self.fig = None
    
    def _to_gym_observation(self, obs):
        """
        Convert the observation from the simulator to the gym format.
        """
        grid, pose = obs
        return {
            "og_map": grid,
            "pose": pose
        }
    
    def reset(self, seed=None, options={'pose': None}):
        super().reset(seed=seed)
        obs = self.simulator.reset(options['pose'])
        info = {}
        
        self.prev_free_area = 0.0
        self.timesteps_elapsed = 0
        
        return self._to_gym_observation(obs), info
    
    def step(self, action):
        obs, timesteps_elapsed, dist_from_env = self.simulator.step(action)
        free_area = self.simulator.og_map.free_area_m2
        
        delta_area = free_area - self.prev_free_area
        map_area = self.simulator.array_map.free_area_m2
                
        coverage_reward = max(0.0, delta_area / map_area)
        fast_reward = -1 #* self.simulator._dt * timesteps_elapsed
        # safety_reward = (np.abs(dist_from_env) * dist_from_env) / max(self.simulator.array_map.height_m,
        #                                                               self.simulator.array_map.width_m)
    
        reward = 10 * coverage_reward + 0.1 * fast_reward
        
        done = free_area / map_area > self.percentage_of_map_to_explore
        truncated = self.simulator.envsteps_elapsed >= self.episode_maxlen_steps
        
        self.prev_free_area = free_area
        
        return self._to_gym_observation(obs), reward, done, truncated, {}
    
    def render(self):
        if self.render_mode != "human":
            return
        
        prob_map, normalized_pose = self.simulator.get_observation()
        pose = normalized_pose * np.array([self.simulator.og_map.width_m,
                                           self.simulator.og_map.height_m,
                                           2*np.pi])
        
        height = self.simulator.og_map.height_px
        width = self.simulator.og_map.width_px
        res = self.simulator.og_map.resolution
        extent = [0, width * res, 0, height * res]
        
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            
            self.im = self.ax.imshow(prob_map[0], vmin=0, vmax=255, cmap='gray_r',
                                 interpolation='nearest', origin='upper', extent=extent)

            self.pose_circle = plt.Circle((0, 0), radius=1.0, edgecolor='blue', facecolor='none')
            self.pose_line, = self.ax.plot([], [], color='blue')
            self.ax.add_patch(self.pose_circle)
            self.ax.set_aspect('equal')
            self.ax.set_title("Occupancy Grid")
        else:
            self.im.set_data(prob_map[0])
                    
        # Update pose drawing
        x, y, theta = pose
        scale = 1.0
        self.pose_circle.center = (x, y)
        self.pose_circle.radius = scale * 0.5
        self.pose_line.set_data(
            [x, x + scale * 0.5 * np.cos(theta)],
            [y, y + scale * 0.5 * np.sin(theta)]
        )

        self.ax.set_xlim(0, width * res)
        self.ax.set_ylim(0, height * res)
        self.ax.grid(True)
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
    
    @staticmethod
    def from_dict(cfg):
        return GymExploreEnv(
            episode_maxlen_steps=cfg['episode_maxlen_steps'],
            percentage_of_map_to_explore=cfg['percentage_of_map_to_explore'],
            map_name=cfg['map_name'],
            og_map_resolution=cfg['og_map_resolution'],
            omega=cfg['omega'],
            v=cfg['v'],
            dt=cfg['dt'],
            travel_cut_short_dist_m=cfg['travel_cut_short_dist_m'],
            og_map_shape=cfg['og_map_shape'],
            render_mode=cfg.get('render_mode', None)
        )