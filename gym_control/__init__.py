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
    reward_threshold=70.0,
)

register(
    id='BRPAtt-v0',
    entry_point='gym_control.envs:BioReactorAttacker',
    max_episode_steps=200,
    reward_threshold=20,
)

register(
    id='BRPDef-v0',
    entry_point='gym_control.envs:BioReactorDefender',
    max_episode_steps=200,
    reward_threshold=60.0,
)

register(
    id='TT-v0',
    entry_point='gym_control.envs:ThreeTank',
    max_episode_steps=200,
    reward_threshold=70.0,
)

register(
    id='TTAtt-v0',
    entry_point='gym_control.envs:ThreeTankAttacker',
    max_episode_steps=200,
    reward_threshold=20,
)

register(
    id='TTDef-v0',
    entry_point='gym_control.envs:ThreeTankDefender',
    max_episode_steps=200,
    reward_threshold=60.0,
)