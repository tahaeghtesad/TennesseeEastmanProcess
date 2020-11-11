from gym.envs.registration import register

import envs.control
import envs.mtd

register(
    id='Historitized-v0',
    entry_point='envs.env_helpers:Historitized'
)

register(
    id='LimitedHistoritized-v0',
    entry_point='envs.env_helpers:LimitedHistoritized'
)