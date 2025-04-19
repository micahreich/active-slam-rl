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
                 max_steps,
                 percentage_of_map_to_explore,
                 map_name,
                 og_map_resolution,
                 dt,
                 og_map_shape,
                 render_mode=None):
        super().__init__()
        
        self.max_steps = max_steps
        print(self.max_steps)
        
        self.percentage_of_map_to_explore = percentage_of_map_to_explore
        
        self.simulator = SimulationEnvironment(
            map_name,
            og_map_resolution,
            dt,
            og_map_shape,
            self.np_random
        )
        
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2, self.simulator.og_map.height_px, self.simulator.og_map.width_px),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.fig, self.ax = None, None
    
    def _to_obs(self):
        prob_map = self.simulator.og_map.to_prob_map()
        
        agent_r, agent_c = self.simulator.og_map.indexer.xy_m_to_ij(self.simulator.pose[:2])
        agent_position_map = np.zeros_like(prob_map)
        agent_position_map[agent_r, agent_c] = 1.0
        
        return np.stack([prob_map, agent_position_map], axis=0)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.simulator.reset()
        
        return self._to_obs(), {}
    
    def step(self, action):
        entropy_before = self.simulator.og_map.entropy
        
        traversed_path = self.simulator.step(action)
        
        entropy_after = self.simulator.og_map.entropy
        
        if traversed_path is None:
            pathlength_reward = 0.0
            exploration_reward = -1.0
        else:
            pathlength_reward = -0.05 * len(traversed_path) * self.simulator.og_map.resolution
            exploration_reward = 200.0 * (entropy_before - entropy_after)
        
        # Check if the agent has explored enough of the map
        done = self.simulator.og_map.free_area_m2 / self.simulator.array_map.free_area_m2 > self.percentage_of_map_to_explore
        
        # Check if the episode has reached its maximum length
        truncated = self.simulator.timesteps_elapsed >= self.max_steps
                
        time_reward = -0.5
        reward = exploration_reward + time_reward #+ pathlength_reward
        
        info = {
            "traversed_path": traversed_path,
            "exploration_reward": exploration_reward,
            "time_reward": time_reward,
            "pathlength_reward": pathlength_reward,
        }
        
        return self._to_obs(), reward, done, truncated, info
    
    def render(self):
        if self.render_mode != "human":
            return
        
        prob_map = self.simulator.og_map.to_prob_map()
        extent = [0, self.simulator.og_map.width_m, 0, self.simulator.og_map.height_m]
        x, y, theta = self.simulator.pose
        
        R = 0.5
        
        # Set up figure and axes
        if self.fig is None:
            plt.ion()
            
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(
                prob_map,
                cmap='gray_r',
                interpolation='nearest',
                origin='upper',
                extent=extent,
                vmin=0, vmax=1  # explicitly set range
            )
            
            # Add colorbar only once
            self.colorbar = self.fig.colorbar(self.im, ax=self.ax)
            
            self.pose_circle = plt.Circle((x, y), radius=R, edgecolor='blue', facecolor='none')
            self.pose_line, = self.ax.plot(
                [x, x + R * np.cos(theta)],
                [y, y + R * np.sin(theta)],
                color='blue'
            )
            
            self.ax.add_patch(self.pose_circle)
        else:
            self.im.set_data(prob_map)
            
            self.pose_circle.center = (x, y)
            self.pose_circle.radius = R
            self.pose_line.set_data(
                [x, x + R * np.cos(theta)],
                [y, y + R * np.sin(theta)]
            )
        
        # Set ticks
        xticks = np.arange(0, self.simulator.og_map.width_m, 1.0)
        yticks = np.arange(0, self.simulator.og_map.height_m, 1.0)
        self.ax.set_xticks(xticks)
        self.ax.set_yticks(yticks)

        self.ax.set_aspect('equal')
        self.ax.set_title(f'Occupancy grid map (H={self.simulator.og_map.entropy:.4f})')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        print("huh")
        
        # prob_map, normalized_pose = self.simulator.get_observation()
        # pose = normalized_pose * np.array([self.simulator.og_map.width_m,
        #                                    self.simulator.og_map.height_m,
        #                                    2*np.pi])
        
        # height = self.simulator.og_map.height_px
        # width = self.simulator.og_map.width_px
        # res = self.simulator.og_map.resolution
        # extent = [0, width * res, 0, height * res]
        
        # if self.fig is None:
        #     plt.ion()
        #     self.fig, self.ax = plt.subplots()
            
        #     self.im = self.ax.imshow(prob_map[0], vmin=0, vmax=255, cmap='gray_r',
        #                          interpolation='nearest', origin='upper', extent=extent)

        #     self.pose_circle = plt.Circle((0, 0), radius=1.0, edgecolor='blue', facecolor='none')
        #     self.pose_line, = self.ax.plot([], [], color='blue')
        #     self.ax.add_patch(self.pose_circle)
        #     self.ax.set_aspect('equal')
        #     self.ax.set_title("Occupancy Grid")
        # else:
        #     self.im.set_data(prob_map[0])
                    
        # # Update pose drawing
        # x, y, theta = pose
        # scale = 1.0
        # self.pose_circle.center = (x, y)
        # self.pose_circle.radius = scale * 0.5
        # self.pose_line.set_data(
        #     [x, x + scale * 0.5 * np.cos(theta)],
        #     [y, y + scale * 0.5 * np.sin(theta)]
        # )

        # self.ax.set_xlim(0, width * res)
        # self.ax.set_ylim(0, height * res)
        # self.ax.grid(True)
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
    
    def close(self):
        if hasattr(self, 'fig') and self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            
            self.fig = None
            self.ax = None
            self.im = None