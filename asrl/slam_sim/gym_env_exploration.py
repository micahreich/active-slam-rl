import os
from time import time
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import label
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

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

        # Observation space: occupancy grid map and pose, both normalized
        self.observation_space = spaces.Dict({
            "og_map": spaces.Box(
                low=0.0, high=1.0, shape=og_map_shape, dtype=np.float32
            ),
            "pose": spaces.Box(
                low=0.0, high=1.0, shape=(3,), dtype=np.float32
            ),
        })

        # Action space: goal position (x, y) in meters
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([og_map_shape[-1] * og_map_resolution, 
                          og_map_shape[-2] * og_map_resolution], dtype=np.float32),
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
        """Convert simulator observation to gym format."""
        grid, pose = obs
        return {"og_map": grid, "pose": pose}
    
    def reset(self, seed=None, options={'pose': None}):
        super().reset(seed=seed)
        obs = self.simulator.reset(options['pose'])
        info = {}
        
        self.prev_free_area = 0.0
        self.timesteps_elapsed = 0
        
        return self._to_gym_observation(obs), info
    
    def _is_in_obstacle_space(self, position):
        """Check if a position is in obstacle space based on the ground truth map."""
        # Convert position (in meters) to grid indices using the ground truth map's indexer
        i, j = self.simulator.array_map._indexer.xy_m_to_ij(position[0], position[1])
        
        # Check if indices are within bounds
        if not (0 <= i < self.simulator.array_map.height_px and 0 <= j < self.simulator.array_map.width_px):
            return True  # Out of bounds is considered an obstacle
        
        # Check occupancy in the ground truth map (assuming _walls is a binary grid where 1 is obstacle)
        return self.simulator.array_map._walls[i, j] == 1
    
    def _plan_path(self, start, goal):
        """Use A* to find a path from start to goal."""
        prob_map = self.simulator.og_map.to_prob_map()
        # Convert to binary grid: 0 (free/unknown) or 1 (obstacle)
        grid_matrix = np.where(prob_map > 0.7, 1, 0)
        grid = Grid(matrix=1 - grid_matrix)  # Invert: 1 is walkable, 0 is not
        
        start_i, start_j = self.simulator.og_map._indexer.xy_m_to_ij(start[0], start[1])
        goal_i, goal_j = self.simulator.og_map._indexer.xy_m_to_ij(goal[0], goal[1])
        
        start_node = grid.node(start_j, start_i)
        end_node = grid.node(goal_j, goal_i)
        
        finder = AStarFinder()
        path, _ = finder.find_path(start_node, end_node, grid)
        
        if not path:
            return []
        
        # Convert path back to (x, y) coordinates in meters
        path_xy = [self.simulator.og_map._indexer.ij_to_xy_m([node.y, node.x]) for node in path]
        return path_xy
    
    def step(self, action):
        """Execute a step with goal position action."""
        goal_pos = action  # (x, y) in meters
        current_pos = self.simulator.pose[:2]
        
        # Check if goal is in obstacle space using ground truth map
        if self._is_in_obstacle_space(goal_pos):
            reward = -10.0  # Large negative reward
            obs = self.simulator.get_observation()
            self.timesteps_elapsed += 1
            self.simulator.envsteps_elapsed += 1
            done = False
            truncated = self.simulator.envsteps_elapsed >= self.episode_maxlen_steps
            return self._to_gym_observation(obs), reward, done, truncated, {}
        
        # Plan path using A*
        path = self._plan_path(current_pos, goal_pos)
        if not path:
            # No valid path; penalize slightly and stay put
            reward = -1.0
            obs = self.simulator.get_observation()
            self.timesteps_elapsed += 1
            self.simulator.envsteps_elapsed += 1
            done = False
            truncated = self.simulator.envsteps_elapsed >= self.episode_maxlen_steps
            return self._to_gym_observation(obs), reward, done, truncated, {}
        
        # Simulate movement along the path
        total_timesteps = 0
        for waypoint in path[1:]:  # Skip starting position
            # Compute relative angle and distance to waypoint
            dx = waypoint[0] - self.simulator.pose[0]
            dy = waypoint[1] - self.simulator.pose[1]
            distance = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx) - self.simulator.pose[2]
            angle = np.arctan2(np.sin(angle), np.cos(angle))  # Normalize to [-pi, pi]
            
            obs, timesteps, _ = self.simulator.step(np.array([angle, distance]))
            total_timesteps += timesteps
        
        free_area = self.simulator.og_map.free_area_m2
        delta_area = free_area - self.prev_free_area
        map_area = self.simulator.array_map.free_area_m2
        
        # Rewards
        coverage_reward = max(0.0, delta_area / map_area)
        fast_reward = -1 * total_timesteps * self.simulator._dt
        reward = 10 * coverage_reward + 0.1 * fast_reward
        
        done = free_area / map_area > self.percentage_of_map_to_explore
        truncated = self.simulator.envsteps_elapsed >= self.episode_maxlen_steps
        
        self.prev_free_area = free_area
        self.timesteps_elapsed += total_timesteps
        
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