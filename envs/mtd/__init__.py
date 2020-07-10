import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='MTD-v0',
    entry_point='envs.mtd.envs:MovingTargetDefenceEnv',
    max_episode_steps=1000
)

register(
    id='MTDAtt-v0',
    entry_point='envs.mtd.envs:MTDAttackerEnv',
    max_episode_steps=1000
)

register(
    id='MTDDef-v0',
    entry_point='envs.mtd.envs:MTDDefenderEnv',
    max_episode_steps=1000
)
