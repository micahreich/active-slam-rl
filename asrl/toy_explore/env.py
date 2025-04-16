from matplotlib import patches, pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridExploreEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=(8, 8), max_steps=200):
        super().__init__()
        self.grid_size = grid_size
        self.nrows, self.ncols = grid_size
        self.num_cells = self.nrows * self.ncols
        self.max_steps = max_steps
        
        # 4 actions: up, down, left, right
        self.action_space = spaces.Discrete(4)

        # Observation space: (visited vector, position index)
        self.observation_space = spaces.Dict({
            "visited": spaces.MultiBinary(self.num_cells),
            "position": spaces.Discrete(self.num_cells)
        })

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_row = np.random.randint(self.nrows)
        self.agent_col = np.random.randint(self.ncols)
        self.visited = np.zeros((self.nrows, self.ncols), dtype=np.int32)
        self.visited[self.agent_row, self.agent_col] = 1
        self.steps = 0

        return self._get_obs(), {}

    def step(self, action):
        if action == 0 and self.agent_row > 0:         # up
            self.agent_row -= 1
        elif action == 1 and self.agent_row < self.nrows - 1:  # down
            self.agent_row += 1
        elif action == 2 and self.agent_col > 0:        # left
            self.agent_col -= 1
        elif action == 3 and self.agent_col < self.ncols - 1:  # right
            self.agent_col += 1

        done = np.all(self.visited)
        truncated = self.steps >= self.max_steps
        
        reward = 5.0 if not self.visited[self.agent_row, self.agent_col] else -0.1
        if done:
            reward += 50.0
    
        self.visited[self.agent_row, self.agent_col] = 1
        self.steps += 1

        return self._get_obs(), reward / 20.0, done, False, {}

    def _get_obs(self):
        flat_visited = self.visited.flatten()
        pos_index = self.agent_row * self.ncols + self.agent_col
        
        return {
            "visited": flat_visited.copy(),
            "position": pos_index
        }

    def render(self, mode="human"):
        if not hasattr(self, "_fig"):
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            self._ax.set_xlim(0, self.ncols)
            self._ax.set_ylim(0, self.nrows)
            self._ax.set_xticks(np.arange(0, self.ncols+1))
            self._ax.set_yticks(np.arange(0, self.nrows+1))
            self._ax.grid(True)
            self._ax.set_aspect('equal')
            
            self._agent_patch = patches.Circle((0.5, 0.5), 0.3, color='blue', zorder=3)
            self._ax.add_patch(self._agent_patch)
            self._visited_patches = []

        # Remove old visited patches
        for patch in self._visited_patches:
            patch.remove()
        self._visited_patches.clear()

        # Add updated visited cells
        for r in range(self.nrows):
            for c in range(self.ncols):
                if self.visited[r, c]:
                    patch = patches.Rectangle((c, self.nrows - r - 1), 1, 1, color='gray')
                    self._visited_patches.append(self._ax.add_patch(patch))

        # Move the agent
        self._agent_patch.center = (
            self.agent_col + 0.5,
            self.nrows - self.agent_row - 0.5,
        )

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def close(self):
        pass

if __name__ == "__main__":
    env = GridExploreEnv()
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        env.render()
    env.close()