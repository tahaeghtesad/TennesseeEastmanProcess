import gym
import numpy as np
from enum import Enum
from typing import *


class BioReactor(gym.Env):

    def __init__(self) -> None:
        super().__init__()
        self.x = np.array([0., 0.])

        self.action_space = gym.spaces.Box(low=-np.array([10.0, 10.0]), high=np.array([10.0, 10.0]))
        self.observation_space = gym.spaces.Box(low=-np.array([10., 10.]), high=np.array([10., 10.]))

        self.episode_count = 0
        self.step_count = 0

        self.highest_reward = -np.inf

        self.goal = np.array([0.99510292, 1.5122427]) # Unstable

        self.win_count = 0

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:  # Obs, Reward, Done, Info
        self.step_count += 1

        u = action

        dx = np.array([
            (mu(self.x[1]) - u[0]) * self.x[0],
            u[0] * (u[1] - self.x[1]) - mu(self.x[1]) / 0.4 * self.x[0]
        ])

        self.x = self.x * 1. + dx
        self.x = np.clip(self.x, self.observation_space.low, self.observation_space.high)

        win = False
        reward = -np.linalg.norm(self.x - self.goal)
        if np.linalg.norm(self.x - self.goal) < 0.01:
            reward += 100
            win = True

        if win:
            self.win_count += 1

        return self.x * (1. + np.random.normal(loc=np.zeros(2,), scale=np.array([0.03, 0.07]))), reward, win, {
            'u': u,
            'x': self.x,
            'dx': dx
        }

    def reset(self) -> Any:
        self.highest_reward = -np.inf

        self.episode_count += 1
        self.step_count = 0
        self.x = self.observation_space.sample()
        return self.x

    def render(self, mode='human') -> None:
        raise NotImplementedError()


def mu(x2: float, mu_max: float = 0.53, km: float = 0.12, k1: float = 0.4545) -> float:
    return mu_max * (x2 / (km + x2 + k1 * x2 * x2))


class AttackerMode(Enum):
    Zero = 0,
    Observation = 1,
    Actuation = 2,
    Both = 3


class BioReactorAttacker(BioReactor):  # This is a noise generator attacker.

    def __init__(self, defender, mode: AttackerMode, range: np.float) -> None:
        super().__init__()
        self.defender = defender

        # This is the amount of noise. Two first for the observation, two last for action

        if mode == AttackerMode.Zero:
            self.action_space = gym.spaces.Box(low=-range * np.array([0.0, 0.0, 0.0, 0.0]),
                                               high=range * np.array([0.0, 0.0, 0.0, 0.0]))
        if mode == AttackerMode.Observation:
            self.action_space = gym.spaces.Box(low=-range * np.array([1.0, 1.0, 0.0, 0.0]),
                                               high=range * np.array([1.0, 1.0, 0.0, 0.0]))
        if mode == AttackerMode.Actuation:
            self.action_space = gym.spaces.Box(low=-range * np.array([0.0, 0.0, 1.0, 1.0]),
                                               high=range * np.array([0.0, 0.0, 1.0, 1.0]))
        if mode == AttackerMode.Both:
            self.action_space = gym.spaces.Box(low=-range * np.array([1.0, 1.0, 1.0, 1.0]),
                                               high=range * np.array([1.0, 1.0, 1.0, 1.0]))

        self.observation_space = gym.spaces.Box(low=-np.array([10., 10.]), high=np.array([10., 10.]))

        self.defender_obs = np.zeros((2,))

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:

        defender_action = self.defender.predict(self.defender_obs)
        action_and_noise = defender_action * (1. + action[2:])

        obs, reward, done, info = super().step(action_and_noise)
        self.defender_obs = obs * (1. + action[:2])

        info['a'] = action[2:]
        info['o'] = action[:2]

        return obs, -reward, done, info

    def reset(self) -> Any:
        self.defender_obs = super().reset()
        return self.defender_obs

    def render(self, mode='human') -> None:
        raise NotImplementedError()


class BioReactorDefender(BioReactor):

    def __init__(self, attacker) -> None:
        super().__init__()
        self.attacker = attacker
        self.attacker_obs = np.zeros((2,))

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:
        attacker_action = self.attacker.predict(self.attacker_obs)
        action_and_noise = action * (1. + attacker_action[2:])

        obs, reward, done, info = super().step(action_and_noise)
        self.attacker_obs = obs

        info['a'] = attacker_action[2:]
        info['o'] = attacker_action[:2]

        return obs * (1. + attacker_action[:2]), reward, done, info

    def reset(self) -> Any:
        self.attacker_obs = super().reset()
        return self.attacker_obs

    def render(self, mode='human') -> None:
        raise NotImplementedError

