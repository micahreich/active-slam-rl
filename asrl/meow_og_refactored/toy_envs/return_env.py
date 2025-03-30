import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class ReturnAfterFarExplorationEnv(gym.Env):
    """
    A Gym environment where the agent is a 2D point mass that controls its velocity.
    
    Goals:
      - Start at the origin (0,0).
      - Move away until the maximum distance reached is at least `min_distance`.
      - Once that distance is reached, the agent is rewarded for returning close to the origin.
      - The agent is further rewarded for having reached a farther maximum distance before returning.
      
    Observation:
      A concatenation of:
        [current x, current y, history of the last N distances from the origin]
        
    Action:
      A 2D velocity vector (bounded in [-1,1] for each dimension).
    """
    def __init__(self,
                 min_distance=4.0,
                 return_radius=1.0,
                 max_steps=200,
                 history_length=10,
                 return_bonus=1000.0,
                 return_factor=1.0,
                 time_penalty=-0.01,
                 dt=1.0):
        super(ReturnAfterFarExplorationEnv, self).__init__()
        
        # Parameters for the task
        self.min_distance = min_distance      # minimum distance to achieve before bonus
        self.return_radius = return_radius    # how close to origin counts as "returning"
        self.max_steps = max_steps            # episode length
        self.history_length = history_length  # number of past distances to include
        self.return_bonus = return_bonus      # bonus for returning to origin after exploration
        self.return_factor = return_factor    # bonus factor scaled by (max_distance - min_distance)
        self.time_penalty = time_penalty      # step penalty to encourage efficiency
        self.dt = dt                          # time step for dynamics

        # Dynamics: a simple point mass starting at origin
        self.position = np.zeros(2, dtype=np.float32)
        self.max_distance = 0.0  # track the maximum distance reached
        
        # A buffer to store the last N distances from the origin
        self.distance_history = np.zeros(self.history_length, dtype=np.float32)
        
        # Observation: current position (2 dims) + history of distances (history_length dims)
        bounds_max = 10.0
        r_max = np.sqrt(2) * bounds_max
        
        obs_low = np.concatenate([
            np.array([-bounds_max, -bounds_max], dtype=np.float32),
            np.zeros(self.history_length, dtype=np.float32)
        ])
        obs_high = np.concatenate([
            np.array([bounds_max, bounds_max], dtype=np.float32),
            r_max * np.ones(self.history_length, dtype=np.float32)
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Action space: 2D velocity in each dimension bounded by [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self.current_step = 0
        
        # Rendering: set up placeholders for matplotlib
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.position = np.zeros(2, dtype=np.float32) + np.random.randn(2) * 0.1  # small random noise
        self.max_distance = 0.0
        self.distance_history = np.zeros(self.history_length, dtype=np.float32)
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Observation includes current position and the history of distances
        return np.concatenate([self.position, self.distance_history])

    def step(self, action):
        # Clip action to ensure it's within the valid range
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
        
        # Update position using simple Euler integration
        new_position = np.clip(self.position + action * self.dt,
                                self.observation_space.low[:2],
                                self.observation_space.high[:2])
        delta_pos = new_position - self.position
        
        self.position = new_position
        
        # Compute current distance from origin
        current_distance = np.linalg.norm(self.position)
        self.max_distance = max(self.max_distance, current_distance)
        
        # # Initialize reward with a time penalty
        # reward = self.time_penalty
        reward = 0.0
        
        # # Reward progress: if the current distance is a new maximum, reward the improvement.
        # if current_distance > self.max_distance:
        #     # reward += (current_distance - self.max_distance)
        #     self.max_distance = current_distance
        
        if self.max_distance < self.min_distance:
            reward += np.linalg.norm(delta_pos)  # reward for moving away from origin
        
        # Once the agent has gone far enough, reward returning close to the origin.
        if self.max_distance >= self.min_distance and current_distance < self.return_radius:
            reward += self.return_bonus + self.return_factor * (self.max_distance - self.min_distance)
        
        # Update the history buffer: shift the array and append the new distance.
        self.distance_history = np.roll(self.distance_history, -1)
        self.distance_history[-1] = current_distance
        
        self.current_step += 1
        done = self.current_step >= self.max_steps or (self.max_distance >= self.min_distance and current_distance < self.return_radius)
        
        return self._get_obs(), reward, done, False, {}

    def render(self, mode='human', close=False):
        if close:
            if self.fig is not None:
                plt.close(self.fig)
            return
        
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
        
        self.ax.clear()
        self.ax.set_title("Return After Far Exploration")
        
        # Set limits a bit beyond the maximum distance reached
        lim = max(self.max_distance + 1, self.return_radius + 1)
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        
        # Plot the origin (start point)
        self.ax.plot(0, 0, 'ro', label='Origin')
        # Plot the agent's current position
        self.ax.plot(self.position[0], self.position[1], 'bo', label='Agent')
        # Draw the return zone as a circle around the origin
        circle = plt.Circle((0, 0), self.return_radius, color='g', fill=False, linestyle='--', label='Return Zone')
        self.ax.add_patch(circle)
        self.ax.legend()
        plt.pause(0.001)
        plt.show()

# Example usage:
if __name__ == "__main__":
    env = ReturnAfterFarExplorationEnv(min_distance=5.0, return_radius=1.0, max_steps=200, history_length=5, dt=0.1)
    obs, _ = env.reset(seed=42)
    print("Initial observation:", obs)
    
    # Simulate a few random steps
    for i in range(200):
        if i < 100:
            action = np.array([0.0, 1.0], dtype=np.float32)  # Move down
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)  # Move up
            
        obs, reward, done, _, _ = env.step(action)
        print(f"Obs: {obs}, Reward: {reward:.2f}")
        env.render()
        
        if done:
            break