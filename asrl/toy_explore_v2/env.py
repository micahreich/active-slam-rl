import time
from matplotlib import patches, pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial import KDTree
from skimage.graph import route_through_array


class GridExploreEnvTeleport(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=(8, 8), max_steps=200):
        super().__init__()
        self.grid_size = grid_size
        self.nrows, self.ncols = grid_size
        self.num_cells = self.nrows * self.ncols
        self.max_steps = max_steps
        
        self.coords = np.indices((self.nrows, self.ncols)).reshape(2, -1).T
        self.costmap = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        
        # 4 actions: up, down, left, right
        # self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_cells)
        
        # Observation space: (visited vector, position index)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, self.nrows, self.ncols), dtype=np.float32)

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
        denormalized_action = np.floor(action * np.array([self.nrows, self.ncols]))
        clipped_action = np.clip(denormalized_action, 0, np.array([self.ncols - 1, self.nrows - 1])).astype(np.int32)
        # row = int(action // self.ncols)
        # col = int(action % self.ncols)

        self.agent_row, self.agent_col = row, col
        
        reward = 1.0 if not self.visited[self.agent_row, self.agent_col] else -1
        
        self.visited[self.agent_row, self.agent_col] = 1.0
        
        done = np.all(self.visited)
        # truncated = self.steps >= self.max_steps
        
        if done:
            reward += 64.0
        
        self.steps += 1

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        visited_channel = self.visited.astype(np.float32)
        agent_position_channel = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        agent_position_channel[self.agent_row, self.agent_col] = 1.0
        
        return np.stack([visited_channel, agent_position_channel], axis=0)

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
    env = GridExploreEnvTeleport()
    obs, info = env.reset()
    _ = env.render()
    
    plt.ion()
    
    done = False
    should_step = False
    action = None
    
    def on_click(event):
        global should_step, action
        
        if event.inaxes:  # Only respond if inside axes
            normalized_j = event.xdata / env.ncols
            normalized_i = 1.0 - event.ydata / env.nrows
            
            should_step = True
            action = np.array([normalized_i, normalized_j])
            
            print("Action:", action)
            
    cid = env._fig.canvas.mpl_connect('button_press_event', on_click)
    
    while not done:
        plt.pause(0.05)  # Yield to GUI thread and check for click events
        
        if should_step:
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            should_step = False
        
    env.close()