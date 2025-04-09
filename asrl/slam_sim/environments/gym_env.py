import gymnasium as gym
from gymnasium import spaces
import numpy as np
from asrl.slam_sim.environments.sim_env import SimEnv

class GymSLAMEnv(gym.Env):
    def __init__(self, occupancy_grid, start_pose, max_steps=100):
        super().__init__()

        self.sim = SimEnv(
            occupancy_grid=occupancy_grid,
            start_pose=start_pose,
            update_occupancy=True
        )

        self.actions = [
            np.array([0.0, 0.4]),
            np.array([0.0, -0.4]),
            np.array([-0.4, 0.0]),
            np.array([0.4, 0.0]),
            np.array([-0.3, 0.3]),
            np.array([0.3, 0.3]),
            np.array([-0.3, -0.3]),
            np.array([0.3, -0.3]),
        ]

        self.action_space = spaces.Discrete(len(self.actions))
        self.max_steps = max_steps
        self.step_count = 0

        H, W = self.sim.mapper.grid.shape
        self.grid_shape = (H, W)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 + H * W,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(self.sim.agent.pose)
        self.step_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action_idx: int):
        action = self.actions[action_idx]
        obs_dict, reward = self.sim.step(action)

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {}

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        pose = self.sim.agent.pose.astype(np.float32)
        grid = self.sim.mapper.log_odds_map_to_prob_map().flatten().astype(np.float32)
        return np.concatenate([pose, grid])
