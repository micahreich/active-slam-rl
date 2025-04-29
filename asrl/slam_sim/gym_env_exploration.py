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
                 k,
                 og_map_shape,
                 render_mode=None):
        super().__init__()
        
        self.max_steps = max_steps        
        self.percentage_of_map_to_explore = percentage_of_map_to_explore
        
        self.simulator = SimulationEnvironment(
            map_name,
            og_map_resolution,
            dt,
            k,
            og_map_shape,
            self.np_random
        )
        
        self.action_space = spaces.Discrete(self.simulator.k)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2, self.simulator.og_map.height_px, self.simulator.og_map.width_px),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            "frontiers": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.simulator.k, 2),
                dtype=np.float32
            ),
            "image": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2, self.simulator.og_map.height_px, self.simulator.og_map.width_px),
                dtype=np.float32
            ),
        })
        
        self.render_mode = render_mode
        self.fig, self.ax = None, None
    
    def _to_obs(self):
        prob_map = self.simulator.og_map.to_prob_map()

        agent_r, agent_c = self.simulator.og_map.indexer.xy_m_to_ij(self.simulator.pose[:2])
        agent_position_map = np.zeros_like(prob_map)
        agent_position_map[agent_r, agent_c] = 1.0

        image = np.stack([prob_map, agent_position_map], axis=0)

        frontiers = np.zeros((self.simulator.k, 2), dtype=np.float32)
        n_frontiers = len(self.simulator.frontiers_xy_m)

        if n_frontiers > 0:
            frontiers[:n_frontiers] = self.simulator.og_map.indexer.xy_m_to_ij(self.simulator.frontiers_xy_m)

        frontiers[n_frontiers:] = np.array([agent_r, agent_c])

        # normalize (row, col) to [0,1]
        normalizer = 1.0 / (np.array([self.simulator.og_map.height_px, self.simulator.og_map.width_px]) - 1)
        frontiers = frontiers * normalizer

        return {
            "frontiers": frontiers,
            "image": image
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
                
        self.simulator.reset()
        self.og_map_entropy = self.simulator.og_map.entropy
        
        return self._to_obs(), {}
    
    def step(self, action):        
        traversed_path = self.simulator.step(action)
        new_entropy = self.simulator.og_map.entropy
        info_gain = self.og_map_entropy - new_entropy
        
        self.og_map_entropy = new_entropy
        
        exploration_done = self.simulator.og_map.free_area_m2 / self.simulator.array_map.free_area_m2 > self.percentage_of_map_to_explore
        frontiers_done = len(self.simulator.frontiers_xy_m) == 0
        
        # Rewards
        exploration_reward = 50.0 * info_gain
        time_reward = -0.4

        if traversed_path is not None:
            pathlength_reward = -0.02 * len(traversed_path) * self.simulator.og_map.resolution
        else:
            pathlength_reward = 0.0
        
        reward = exploration_reward + time_reward + pathlength_reward

        # Done / truncated 
        done = False #exploration_done or frontiers_done
        truncated = False #self.simulator.timesteps_elapsed >= self.max_steps
                        
        info = {
            "exploration_reward": exploration_reward,
            "time_reward": time_reward,
            "pathlength_reward": pathlength_reward,
            "traversed_path": traversed_path,
        }
        
        return self._to_obs(), reward, done, truncated, info
    
    def render(self):
        if self.render_mode != "human":
            return
        
        obs = self._to_obs()
        prob_map = obs["image"][0]  # the occupancy map
        agent_map = obs["image"][1]  # not used here
        frontiers = obs["frontiers"]  # normalized [0,1] xy (row, col) positions

        extent = [0, self.simulator.og_map.width_m, 0, self.simulator.og_map.height_m]
        x, y, theta = self.simulator.pose
        R = 0.5  # robot visualization size

        # Set up figure and axes
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()

            # Occupancy grid
            self.im = self.ax.imshow(prob_map, cmap='gray_r', origin='upper', vmin=0.0, vmax=1.0, extent=extent)

            # Frontiers mask
            self.frontiers_im = self.ax.imshow(np.zeros_like(prob_map), cmap='Reds', alpha=0.5, origin='upper', vmin=0.0, vmax=1.0, extent=extent)

            # Sampled frontiers (blue x's)
            self.sampled_frontiers = self.ax.scatter([], [], c='red', marker='s', s=30, label='Sampled Frontiers')

            # Frontier labels
            self.frontier_texts = []
            for _ in range(self.simulator.k):
                txt = self.ax.text(0, 0, '', color='red', fontsize=10, ha='center', va='center')
                txt.set_visible(False)
                self.frontier_texts.append(txt)

            # Agent pose
            self.pose_circle = plt.Circle((x, y), radius=R, edgecolor='blue', facecolor='none')
            self.pose_line, = self.ax.plot(
                [x, x + R * np.cos(theta)],
                [y, y + R * np.sin(theta)],
                color='blue'
            )
            self.ax.add_patch(self.pose_circle)
            self.ax.set_aspect('equal')
            
            self.fig.colorbar(self.im, ax=self.ax)

        else:
            self.im.set_data(prob_map)

            self.pose_circle.center = (x, y)
            self.pose_line.set_data(
                [x, x + R * np.cos(theta)],
                [y, y + R * np.sin(theta)]
            )

        # Update frontiers mask
        frontiers_mask = self.simulator.og_map.frontiers_mask()
        masked_frontiers_map = np.ma.masked_where(frontiers_mask == 0, frontiers_mask)
        self.frontiers_im.set_data(masked_frontiers_map)

        # Update sampled frontiers points
        denormalizer = np.array([self.simulator.og_map.height_px, self.simulator.og_map.width_px])
        frontiers_ij = frontiers * denormalizer

        # Convert (row, col) ij -> xy meters for plotting
        frontiers_xy_m = self.simulator.og_map.indexer.ij_to_xy_m(frontiers_ij)
        self.sampled_frontiers.set_offsets(frontiers_xy_m)

        # Update frontier labels
        for i, txt in enumerate(self.frontier_texts):
            x_f, y_f = frontiers_xy_m[i]
            label_offset = 0.45
            
            txt.set_position((x_f, y_f + label_offset))
            txt.set_text(str(i))  # Label frontier 0, ..., k-1
            txt.set_visible(True)

        # Update ticks
        self.ax.set_xticks(np.arange(0, self.simulator.og_map.width_m, 1.0))
        self.ax.set_yticks(np.arange(0, self.simulator.og_map.height_m, 1.0))

        self.ax.set_xlim(0, self.simulator.og_map.width_m)
        self.ax.set_ylim(0, self.simulator.og_map.height_m)
        
        self.ax.set_title(f'Occupancy Grid Map (Entropy={self.simulator.og_map.entropy:.4f})')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
            
    def close(self):
        if hasattr(self, 'fig') and self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            
            self.fig = None
            self.ax = None
            self.im = None