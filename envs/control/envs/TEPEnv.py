import gym
from typing import *


class TennesseeEastmanProcess(gym.Env):

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError()

    def step(self, action) -> Tuple[Any, float, bool, Dict]: # Obs, Reward, Done, Info
        raise NotImplementedError()

    def reset(self) -> Any:
        raise NotImplementedError()

    def render(self, mode='human') -> None:
        raise NotImplementedError()
