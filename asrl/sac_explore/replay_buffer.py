import numpy as np
import torch


class ReplayBuffer:
    """Buffer to store environment transitions with support for composite observations."""

    def __init__(self, obs_shapes, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # Normalize obs_shapes to list of shapes
        if isinstance(obs_shapes, dict):
            self.multi_obs = True
            self.obs_shapes = obs_shapes
        else:
            self.multi_obs = False
            self.obs_shapes = {'obs': obs_shapes}

        # Create observation buffers
        self.obses = {}
        self.next_obses = {}
        
        for k in self.obs_shapes:
            shape = self.obs_shapes[k]
            dtype = np.float32 if len(shape) == 1 else np.uint8
            
            self.obses[k] = np.empty((capacity, *shape), dtype=dtype)
            self.next_obses[k] = np.empty((capacity, *shape), dtype=dtype)
        
        # Other buffers
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        if not self.multi_obs:
            obs = {'obs': obs}
            next_obs = {'obs': next_obs}

        for i, k in enumerate(self.obs_shapes):
            np.copyto(self.obses[k][self.idx], obs[k])
            np.copyto(self.next_obses[k][self.idx], next_obs[k])

        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        max_idx = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_idx, size=batch_size)

        # Sample observations
        sampled_obs = {
            k: torch.as_tensor(self.obses[k][idxs], device=self.device).float()
            for k in self.obses
        }
        
        sampled_next_obs = {
            k: torch.as_tensor(self.next_obses[k][idxs], device=self.device).float()
            for k in self.next_obses
        }

        if not self.multi_obs:
            sampled_obs = sampled_obs['obs']
            sampled_next_obs = sampled_next_obs['obs']

        actions = torch.as_tensor(self.actions[idxs], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device).float()

        return sampled_obs, actions, rewards, sampled_next_obs, not_dones
