import logging
import math
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class FourRooms(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50
    }

    def __init__(self):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in layout.splitlines()])

        # From any state the agent can perform one of four actions: up, down, left, or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0., high=1., shape=(np.sum(self.occupancy == 0),), dtype=np.float32)

        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.rng = np.random.default_rng(1234)  # Updated to use numpy's default_rng

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.tocell = {v: k for k, v in self.tostate.items()}

        self.goal = self.tostate[(7, 9)]
        self.init_states = list(range(self.observation_space.shape[0]))
        self.init_states.remove(self.goal)
        self.ep_steps = 0

        self.currentcell = None  # Initialize current cell

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed]

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Ensure compatibility with gymnasium's reset signature
        self.seed(seed)
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        self.ep_steps = 0
        return self.get_state(state), {}

    def switch_goal(self):
        prev_goal = self.goal
        self.goal = self.rng.choice(self.init_states)
        self.init_states.append(prev_goal)
        self.init_states.remove(self.goal)
        assert prev_goal in self.init_states
        assert self.goal not in self.init_states

    def get_state(self, state):
        s = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        s[state] = 1
        return s

    def render(self, mode='human', show_goal=True):
        current_grid = np.array(self.occupancy)
        current_grid[self.currentcell[0], self.currentcell[1]] = -1
        if show_goal:
            goal_cell = self.tocell[self.goal]
            current_grid[goal_cell[0], goal_cell[1]] = -2
        if mode == 'human':
            print(current_grid)
        elif mode == 'rgb_array':
            return current_grid
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported.")

    def step(self, action):
        """
        The agent can perform one of four actions:
        up, down, left, or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall, then the agent remains in the
        same cell.
        Rewards are zero on all state transitions except reaching the goal.
        """
        self.ep_steps += 1

        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            # if self.rng.uniform() <= 0.02:
            #     empty_cells = self.empty_around(self.currentcell)
            #     self.currentcell = empty_cells[self.rng.integers(len(empty_cells))]
            # else:
            self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        done = state == self.goal
        if done:
            reward = 20.0
        else:
            reward = -1.0

        if not done and self.ep_steps >= 1000:
            truncated = True
            reward = -1.0
        else:
            truncated = False

        return self.get_state(state), reward, done, truncated, {'pos': self.currentcell}

if __name__ == "__main__":
    # env = FourRooms()
    # env.seed(3)
    
    # env.reset()
    # env.render('human', show_goal=True)
    import asrl.option_critic.envs
    
    env = gym.make("FourRooms-v0")