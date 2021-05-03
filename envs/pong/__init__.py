import logging
from gym.envs.registration import register

register(
    id='Memory-v0',
    entry_point='envs.pong.memory:MemoryEnv'
)

register(
    id='Processing-v0',
    entry_point='envs.pong.processing:ProcessingEnv'
)