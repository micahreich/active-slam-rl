from gymnasium.envs.registration import register
from .four_rooms import FourRooms

register(
    id='FourRooms-v0',
    entry_point='envs:FourRooms',
)