import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='TEP-v0',
    entry_point='gym_control.envs:TennesseeEastmanProcess'
)

register(
    id='BRP-v0',
    entry_point='gym_control.envs:BioReactor',
    max_episode_steps=200,
    reward_threshold=50.0,
)

register(
    id='BRPAtt-v0',
    entry_point='gym_control.envs:BioReactorAttacker',
    max_episode_steps=200,
    reward_threshold=90.0,
)