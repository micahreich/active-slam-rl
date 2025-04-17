import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter


class GridExploreEnvTeleport(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=(64, 64), max_steps=200, gaussian_sigma=5.0, map_value_max=1.0):
        super().__init__()
        self.nrows, self.ncols = grid_size
        self.grid_size = grid_size
        self.ncells = self.nrows * self.ncols
        self.max_steps = max_steps
        self.gaussian_sigma = gaussian_sigma
        self.map_value_max = map_value_max

        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=map_value_max, shape=(1, self.nrows, self.ncols), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.map = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        self.steps = 0
        return self.map.copy().reshape((1, self.nrows, self.ncols)), {}

    def step(self, action):
        self.steps += 1
        action = np.clip(action, 0, 1)

        # Convert normalized action to float (row, col) in grid coordinates
        r = action[0] * (self.nrows - 1)
        c = action[1] * (self.ncols - 1)
        
        # Create Gaussian centered at (r, c)
        stamp = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        stamp[int(r), int(c)] = 1.0
        stamp = self.gaussian_sigma * 50.0 * gaussian_filter(stamp, sigma=self.gaussian_sigma)

        # Track old total for reward
        total_before = np.sum(self.map)

        # Add and clip the map
        self.map = np.clip(self.map + stamp, 0, self.map_value_max)

        # Compute reward as increase in total value
        total_after = np.sum(self.map)
        reward = 1e2 * (total_after - total_before) / self.ncells - 0.75

        done = total_after / self.ncells >= self.map_value_max * 0.85
                
        return self.map.copy().reshape((1, self.nrows, self.ncols)), reward, done, False, {}

    def render(self, mode="human"):
        if not hasattr(self, "_fig"):
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        self._ax.clear()
        self._ax.imshow(self.map, cmap='viridis', vmin=0, vmax=self.map_value_max,
                        extent=[0, self.ncols, 0, self.nrows])
        self._ax.set_title("Map Value Heatmap")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def close(self):
        plt.close(self._fig)
        

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
            
            print(f"{event.xdata=:.2f}, {event.ydata=:.2f} -> {normalized_i=:.2f}, {normalized_j=:.2f}")
            
    cid = env._fig.canvas.mpl_connect('button_press_event', on_click)
    
    while not done:
        plt.pause(0.05)  # Yield to GUI thread and check for click events
        
        if should_step:
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            print(f"Step: {env.steps}, Reward: {reward:.2f}, Done: {done}")
            should_step = False
        
    env.close()