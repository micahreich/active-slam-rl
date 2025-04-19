import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter


class GridExploreEnvTeleport(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=(64, 64), max_steps=200, gaussian_sigma=10.0, map_value_max=1.0):
        super().__init__()

        self.nrows, self.ncols = grid_size
        self.grid_size = grid_size
        self.ncells = self.nrows * self.ncols
        self.max_steps = max_steps
        self.gaussian_sigma = gaussian_sigma
        self.map_value_max = map_value_max

        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=map_value_max, shape=(2, self.nrows, self.ncols), dtype=np.float32)
        
        self.reset()
    
    # def binary_entropy(self, p: np.ndarray) -> np.ndarray:
    #     eps = 1e-10  # avoid log(0)
    #     return -p * np.log2(p + eps) - (1 - p) * np.log2(1 - p + eps)

    def map_entropy(self) -> float:
        eps = 1e-10
        entropies = -self.map * np.log2(self.map + eps) - (1 - self.map) * np.log2(1 - self.map + eps)

        return np.mean(entropies)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.obstacles = np.zeros((self.nrows, self.ncols), dtype=np.float32)        
        s = 10
        self.obstacles[s:-s, s:-s] = 1.0
        self.free_space = 1.0 - self.obstacles
        self.ncells_free = self.free_space.sum()
        
        self.map = self.map_value_max / 2 * np.ones((self.nrows, self.ncols), dtype=np.float32)
        
        
        indices = np.argwhere(self.free_space == 1)
        agent_r, agent_c = indices[self.np_random.choice(len(indices))]
        
        self.agent_r = agent_r * 1.0
        self.agent_c = agent_c * 1.0
                    
        # self.free_space = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        self.steps = 0
        
        self._stamp_map(self.agent_r, self.agent_c)
        
        return self._to_obs(), {}

    def _to_obs(self):
        map_img = self.map.copy()
        
        curr_pos = np.zeros_like(map_img)
        curr_pos[int(self.agent_r), int(self.agent_c)] = 1.0
        
        return np.stack([map_img, curr_pos], axis=0).astype(np.float32)
    
    def _stamp_map(self, r, c):
        # Gaussian stamp centered at (r, c)
        stamp = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        stamp[int(r), int(c)] = 1.0
        stamp = gaussian_filter(stamp, sigma=self.gaussian_sigma)

        # Normalize stamp so its values are in [0, 1]
        stamp = stamp / np.max(stamp)

        # Shift free-space cells *toward 0* (known free), obstacles *toward 1* (known occupied)
        self.map = (
            self.map * (1 - stamp) +               # retain old value where stamp is small
            stamp * (self.obstacles * 0.0 + self.free_space * 1.0)  # move toward 1 or 0
        )

        # Clip for safety
        self.map = np.clip(self.map, 0.0, self.map_value_max)
    
    def step(self, action):
        self.steps += 1
        action = np.clip(action, 0, 1)

        # Convert normalized action to float (row, col) in grid coordinates
        r = action[0] * (self.nrows - 1)
        c = action[1] * (self.ncols - 1)
        
        # Track old total for reward
        map_entropy_before = self.map_entropy()
        
        self._stamp_map(r, c)

        # Compute reward as increase in total value
        map_entropy_after = self.map_entropy()
        
        entropy_reward = 50.0 * (map_entropy_before - map_entropy_after)
        time_reward = -0.5
        closeness_reward = 0.5 * -np.linalg.norm(
            np.array([self.agent_r, self.agent_c]) - np.array([r, c])
        ) / np.sqrt(self.nrows**2 + self.ncols**2)
        
        if self.obstacles[int(r), int(c)] == 1.0:
            obstacle_penalty = -6.0
        else:
            obstacle_penalty = 0.0

        total_explored = np.sum(self.map * self.free_space / self.map_value_max)
        done = total_explored / self.ncells_free >= self.map_value_max * 0.90
        
        reward = entropy_reward + time_reward + closeness_reward + obstacle_penalty
        
        self.agent_r = r
        self.agent_c = c

        return self._to_obs(), reward, done, False, {}

    def render(self, mode="human"):
        if not hasattr(self, "_fig"):
            plt.ion()
            
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            self._im = self._ax.imshow(
                self.map, cmap='inferno', vmin=0, vmax=self.map_value_max,
                extent=[0, self.ncols, 0, self.nrows]
            )
            self._cbar = self._fig.colorbar(self._im, ax=self._ax)
            self._ax.set_title("Map Value Heatmap")
        else:
            self._im.set_data(self.map)
                
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
    episode_reward = 0.0
    
    while not done:
        plt.pause(0.05)  # Yield to GUI thread and check for click events
        
        if should_step:
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            print(f"Step: {env.steps}, Reward: {reward:.2f}, Done: {done}, Entropy: {env.map_entropy():.2f}")
            episode_reward += reward
            should_step = False
    
    print(f"Episode finished. Total reward: {episode_reward:.2f}")
    env.close()