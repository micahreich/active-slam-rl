from gymnasium.envs.registration import register
from .multi_goal import MultiGoal
from .return_env import ReturnAfterFarExplorationEnv

register(
    id='MultiGoal-v0',
    entry_point='toy_envs:MultiGoal',
)

register(
    id='ReturnToOrigin-v0',
    entry_point='toy_envs:ReturnAfterFarExplorationEnv',
)