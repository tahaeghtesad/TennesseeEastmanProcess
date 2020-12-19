import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='TEP-v0',
    entry_point='envs.control.envs:TennesseeEastmanProcess'
)

register(
    id='BRP-v0',
    entry_point='envs.control.envs:BioReactor',
    reward_threshold=-0.01,
)

register(
    id='BRPAtt-v0',
    entry_point='envs.control.envs:BioReactorAttacker',
    reward_threshold=10,
)

register(
    id='BRPDef-v0',
    entry_point='envs.control.envs:BioReactorDefender',
    reward_threshold=-0.01
)

register(
    id='TT-v0',
    entry_point='envs.control.envs:ThreeTank',
    reward_threshold=-0.01
)

register(
    id='TTAtt-v0',
    entry_point='envs.control.envs:ThreeTankAttacker',
    reward_threshold=10
)

register(
    id='TTDef-v0',
    entry_point='envs.control.envs:ThreeTankDefender',
    reward_threshold=-0.01
)
