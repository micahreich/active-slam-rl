import torch
import numpy as np
import gymnasium as gym


def get_device():
    """Get the device to use for PyTorch tensors."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs

def to_numpy(tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy array."""
    return tensor.detach().cpu().numpy()

def make_env(env_name):
    env = gym.make(env_name)
    # is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    # if is_atari:
    #     env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=True)
    #     env = TransformReward(env, lambda r: np.clip(r, -1, 1))
    #     env = FrameStack(env, 4)
    return env, False
