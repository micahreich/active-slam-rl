from gymnasium.envs.registration import register

register(
    id="MultiGoal-v0",  # Unique ID for your environment
    entry_point="asrl.meow.multi_goal_env:MultiGoal",  # Module path
)