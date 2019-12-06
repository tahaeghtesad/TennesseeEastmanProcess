import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='TEP-v0',
    entry_point='gym_control.envs:TennesseeEastmanProcess'
)

register(
    id='BRP-v0',
    entry_point='gym_control.envs:BioReactor'
)