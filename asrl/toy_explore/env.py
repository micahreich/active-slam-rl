import time
from matplotlib import patches, pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridExploreEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=(20, 20), obstacle_size=(6, 6), max_steps=1000):
        super().__init__()
        self.grid_size = grid_size
        self.nrows, self.ncols = grid_size
        self.obstacle_size = obstacle_size
        self.max_steps = max_steps
        
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.nrows, self.ncols), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        obstacle_row = np.random.randint(0, self.nrows - self.obstacle_size[0] + 1)
        obstacle_col = np.random.randint(0, self.ncols - self.obstacle_size[1] + 1)
        self.obstacle = np.zeros((self.nrows, self.ncols), dtype=np.int32)
        self.obstacle[obstacle_row:obstacle_row + self.obstacle_size[0], 
                      obstacle_col:obstacle_col + self.obstacle_size[1]] = 1
        
        while True:
            self.agent_row = np.random.randint(self.nrows)
            self.agent_col = np.random.randint(self.ncols)
            if self.obstacle[self.agent_row, self.agent_col] == 0:
                break
        
        self.visited = np.zeros((self.nrows, self.ncols), dtype=np.int32)
        self.visited[self.agent_row, self.agent_col] = 1
        self.steps = 0

        return self._get_obs(), {}

    def step(self, action):
        # Calculate new position
        new_row, new_col = self.agent_row, self.agent_col
        if action == 0 and self.agent_row > 0:         # up
            new_row -= 1
        elif action == 1 and self.agent_row < self.nrows - 1:  # down
            new_row += 1
        elif action == 2 and self.agent_col > 0:        # left
            new_col -= 1
        elif action == 3 and self.agent_col < self.ncols - 1:  # right
            new_col += 1
        
        if self.obstacle[new_row, new_col] == 1:
            reward = -1.0  
        else:
            self.agent_row, self.agent_col = new_row, new_col
            if not self.visited[self.agent_row, self.agent_col]:
                reward = 5.0
                self.visited[self.agent_row, self.agent_col] = 1
            else:
                reward = -0.1
        
        done = np.all(self.visited)
        if done:
            reward += 50.0
        
        truncated = self.steps >= self.max_steps
        self.steps += 1

        return self._get_obs(), reward / 20.0, done, truncated, {}

    def _get_obs(self):
        visited_channel = self.visited.astype(np.float32)
        agent_position_channel = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        agent_position_channel[self.agent_row, self.agent_col] = 1.0
        obstacle_channel = self.obstacle.astype(np.float32)
        return np.stack([visited_channel, agent_position_channel, obstacle_channel], axis=0)

    def render(self, mode="human"):
        if not hasattr(self, "_fig"):
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(10, 10))  
            self._ax.set_xlim(0, self.ncols)
            self._ax.set_ylim(0, self.nrows)
            self._ax.set_xticks(np.arange(0, self.ncols + 1))
            self._ax.set_yticks(np.arange(0, self.nrows + 1))
            self._ax.grid(True)
            self._ax.set_aspect('equal')
            
            self._agent_patch = patches.Circle((0.5, 0.5), 0.3, color='blue', zorder=3)
            self._ax.add_patch(self._agent_patch)
            self._visited_patches = []
            self._obstacle_patches = []

        for patch in self._visited_patches + self._obstacle_patches:
            patch.remove()
        self._visited_patches.clear()
        self._obstacle_patches.clear()

        for r in range(self.nrows):
            for c in range(self.ncols):
                if self.obstacle[r, c]:
                    patch = patches.Rectangle((c, self.nrows - r - 1), 1, 1, color='red')
                    self._obstacle_patches.append(self._ax.add_patch(patch))
                elif self.visited[r, c]:
                    patch = patches.Rectangle((c, self.nrows - r - 1), 1, 1, color='gray')
                    self._visited_patches.append(self._ax.add_patch(patch))

        self._agent_patch.center = (
            self.agent_col + 0.5,
            self.nrows - self.agent_row - 0.5,
        )

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def close(self):
        pass