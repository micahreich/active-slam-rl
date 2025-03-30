import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces


class MultiGoal(gym.Env):
    """
    Move a 2D point mass to one of the goal positions. The cost is the distance to
    the closest goal plus action penalties.
    
    State: position.
    Action: velocity.
    """
    def __init__(self,
                 goal_reward=10,
                 actuation_cost_coeff=30.0,
                 distance_cost_coeff=1.0,
                 init_sigma=0.1):
        super(MultiGoal, self).__init__()
        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.zeros(2, dtype=np.float32)
        self.init_sigma = init_sigma
        self.goal_positions = np.array(
            [
                [5, 0],
                [-5, 0],
                [0, 5],
                [0, -5]
            ],
            dtype=np.float32)
        self.goal_threshold = 1.0
        self.goal_reward = goal_reward
        self.action_cost_coeff = actuation_cost_coeff
        self.distance_cost_coeff = distance_cost_coeff
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.0

        self.observation_space = spaces.Box(
            low=np.array([self.xlim[0], self.ylim[0]], dtype=np.float32),
            high=np.array([self.xlim[1], self.ylim[1]], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(2,),
            dtype=np.float32
        )
        self.observation = None

        # For visualization
        self._ax = None
        self._env_lines = []
        self.dynamic_plots = []
        self.timestep = 0

        # Ensure initial reset to set state
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        # Sample initial state in float32
        unclipped_observation = self.init_mu + self.init_sigma * np.random.normal(size=self.dynamics.s_dim).astype(np.float32)
        self.observation = np.clip(unclipped_observation, 
                                   self.observation_space.low, 
                                   self.observation_space.high).astype(np.float32)
        self.timestep = 0
        return self.observation, {'pos': self.observation}

    def get_current_obs(self):
        return np.copy(self.observation)

    def step(self, action):
        action = np.clip(action.ravel(), 
                         self.action_space.low, 
                         self.action_space.high).astype(np.float32)
        # Compute new observation and ensure it's float32
        observation = self.dynamics.forward(self.observation, action)
        observation = np.clip(observation, 
                              self.observation_space.low, 
                              self.observation_space.high).astype(np.float32)

        reward = self.compute_reward(observation, action)
        # Check if any goal is reached
        dist_to_goal = np.min([np.linalg.norm(observation - goal_pos) for goal_pos in self.goal_positions])
        done = dist_to_goal < self.goal_threshold
        if done:
            reward += self.goal_reward

        self.timestep += 1
        truncate = (self.timestep >= 1000)
        self.observation = np.copy(observation)

        return observation, reward, done, truncate, {'pos': observation}

    def _init_plot(self):
        fig_env = plt.figure(figsize=(7, 7))
        self._ax = fig_env.add_subplot(111)
        self._ax.axis('equal')
        self._ax.set_xlim(self.xlim[0]*1.1, self.xlim[1]*1.1)
        self._ax.set_ylim(self.ylim[0]*1.1, self.ylim[1]*1.1)
        self._ax.set_title('Multi-Goal Environment')
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')
        # Plot cost contours and goals once
        self._plot_position_cost(self._ax)

    def render_rollouts(self, paths=(), file_name="rollout"):
        """
        Render past rollouts. Each element in paths should be an array of positions.
        The rendered plot is saved to a PNG file.
        """
        if self._ax is None:
            self._init_plot()

        # Clear previous dynamic lines
        for line in self._env_lines:
            line.remove()
        self._env_lines = []

        for path in paths:
            positions = np.stack(path)
            xx = positions[:, 0]
            yy = positions[:, 1]
            self._env_lines += self._ax.plot(xx, yy, 'b')

        plt.savefig(file_name + '.png')

    def render(self, mode='human', close=False):
        """
        Render the current state of the environment.
        Call this method in interactive mode to see the current position.
        """
        if close:
            plt.close()
            return

        if self._ax is None:
            self._init_plot()

        # Remove previous dynamic plots (current position)
        for line in self.dynamic_plots:
            line.remove()
        self.dynamic_plots = []

        # Plot current position
        pos_plot, = self._ax.plot(self.observation[0], self.observation[1], 'bo', markersize=8)
        self.dynamic_plots.append(pos_plot)

        plt.pause(0.001)
        if mode == 'human':
            plt.show()
        return self._ax

    def compute_reward(self, observation, action):
        # Penalize the L2 norm of the action
        action_cost = np.sum(action ** 2) * self.action_cost_coeff
        # Penalize squared distance to the nearest goal
        cur_position = observation
        goal_cost = self.distance_cost_coeff * np.min([
            np.sum((cur_position - goal_pos) ** 2)
            for goal_pos in self.goal_positions
        ])
        return -1.0 * (action_cost + goal_cost)

    def _plot_position_cost(self, ax):
        delta = 0.01
        x_min, x_max = (1.1 * np.array(self.xlim))
        y_min, y_max = (1.1 * np.array(self.ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )
        # Compute the squared distance to the nearest goal at each grid point
        goal_costs = np.min([
            (X - goal_x) ** 2 + (Y - goal_y) ** 2
            for goal_x, goal_y in self.goal_positions
        ], axis=0)
        contours = ax.contour(X, Y, goal_costs, 20)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f')
        ax.plot(self.goal_positions[:, 0], self.goal_positions[:, 1], 'ro')
        return contours


class PointDynamics:
    """
    2D point dynamics.
    State: position.
    Action: velocity.
    """
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action):
        mu_next = state + action
        noise = (self.sigma * np.random.normal(size=self.s_dim)).astype(np.float32)
        state_next = mu_next + noise
        return state_next.astype(np.float32)


# Example usage:
if __name__ == "__main__":
    env = MultiGoal()
    obs, info = env.reset(seed=42)
    print("Initial observation:", obs)

    # Take a random step
    action = env.action_space.sample()
    obs, reward, done, truncate, info = env.step(action)
    print("Observation after one step:", obs)

    # Render the current state (this will open a matplotlib window)
    env.render()
    plt.show()  # Ensure the window stays open
