import gym
import numpy as np
import logging
from enum import Enum
from typing import *


class BioReactor(gym.Env):

    def __init__(self, noise=True) -> None:
        super().__init__()
        self.logger = logging.getLogger(__class__.__name__)
        self.x = np.array([0., 0.])

        self.action_space = gym.spaces.Box(low=-np.array([10.0, 10.0]), high=np.array([10.0, 10.0]))
        self.observation_space = gym.spaces.Box(low=-np.array([10., 10.]), high=np.array([10., 10.]))

        self.episode_count = 0
        self.step_count = 0

        self.highest_reward = -np.inf

        self.goal = np.array([0.99510292, 1.5122427])  # Unstable
        self.noise = noise

        self.win_count = 0

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:  # Obs, Reward, Done, Info
        self.step_count += 1

        u = action * (1. + np.random.normal(loc=np.zeros(2,), scale=np.array([0.00, 0.07]))) if self.noise else action

        dx = np.array([
            (mu(self.x[1]) - u[0]) * self.x[0],
            u[0] * (u[1] - self.x[1]) - mu(self.x[1]) / 0.4 * self.x[0]
        ])

        self.x = self.x * 1. + dx
        self.x = np.clip(self.x, self.observation_space.low[:2], self.observation_space.high[:2])

        win = False
        reward = -np.linalg.norm(self.x - self.goal)
        if np.linalg.norm(self.x - self.goal) < 0.01:
            reward += 100
            win = True

        if win:
            self.win_count += 1

        return self.x * (1. + np.random.normal(loc=np.zeros(2,), scale=np.array([0.00, 0.07]))) if self.noise else self.x, reward, win, {
            'u': u,
            'x': self.x,
            'dx': dx
        }

    def reset(self) -> Any:
        self.highest_reward = -np.inf

        self.episode_count += 1
        self.step_count = 0
        self.x = self.observation_space.sample()[:2]
        self.logger.debug(f'Reset... Starting Point: {self.x}')
        return self.x

    def render(self, mode='human') -> None:
        raise NotImplementedError()


def mu(x2: float, mu_max: float = 0.53, km: float = 0.12, k1: float = 0.4545) -> float:
    return mu_max * (x2 / (km + x2 + k1 * x2 * x2))


class AdversarialBioReactor(gym.Env):
    def __init__(self, compromise_actuation_prob: float, compromise_observation_prob: float) -> None:
        super().__init__()
        self.logger = logging.getLogger(__class__.__name__)
        self.compromise_actuation_prob = compromise_actuation_prob
        self.compromise_observation_prob = compromise_observation_prob

        self.compromise_actuation = False
        self.compromise_observation = False

        self.env = gym.make('BRP-v0')

    def reset(self) -> Any:
        obs = self.env.reset()
        self.compromise_observation = np.random.random() < self.compromise_observation_prob
        self.compromise_actuation = np.random.random() < self.compromise_actuation_prob
        self.logger.debug(f'Observation Compromised: {self.compromise_observation} - Actuation Compromised: {self.compromise_actuation}')

        return np.append(obs, np.array([np.float(self.compromise_observation), np.float(self.compromise_actuation)]))


class BioReactorAttacker(AdversarialBioReactor):  # This is a noise generator attacker.

    def __init__(self, defender, compromise_actuation_prob: float, compromise_observation_prob: float, power: float = 0.3) -> None:
        super().__init__(compromise_actuation_prob, compromise_observation_prob)
        self.logger = logging.getLogger(__class__.__name__)
        self.defender = defender

        self.observation_space = gym.spaces.Box(low=np.array([-10., -10., 0.0, 0.0]), high=np.array([10., 10., 1.0, 1.0]))
        self.action_space = gym.spaces.Box(low=-power * np.array([1.0, 1.0, 1.0, 1.0]),
                                           high=power * np.array([1.0, 1.0, 1.0, 1.0]))

        self.defender_obs = np.zeros((4,))

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:

        defender_action = self.defender.predict(self.defender_obs)
        action_and_noise = defender_action * (1. + action[2:]) if self.compromise_actuation else defender_action

        obs, reward, done, info = self.env.step(action_and_noise)
        self.defender_obs = np.append(obs[:2] * (1. + action[:2]) if self.compromise_observation else obs[:2], np.array([np.float(self.compromise_actuation), np.float(self.compromise_observation)]))

        info['a'] = action[2:]
        info['o'] = action[:2]

        ret = np.append(obs[:2], np.array([np.float(self.compromise_actuation), np.float(self.compromise_observation)])), -reward, done, info
        assert ret[0].shape == (4,)
        return ret

    def reset(self) -> Any:
        self.defender_obs = super().reset()
        assert self.defender_obs.shape == (4,)
        return self.defender_obs

    def render(self, mode='human') -> None:
        raise NotImplementedError()


class BioReactorDefender(AdversarialBioReactor):

    def __init__(self, attacker, compromise_actuation_prob: float, compromise_observation_prob: float) -> None:
        super().__init__(compromise_actuation_prob, compromise_observation_prob)
        self.logger = logging.getLogger(__class__.__name__)
        self.attacker = attacker
        self.attacker_obs = np.zeros((4,))

        self.observation_space = gym.spaces.Box(low=np.array([-10., -10., 0., 0.]),
                                                high=np.array([10., 10., 1., 1.]))
        self.action_space = gym.spaces.Box(low=-np.array([10., 10.]),
                                           high=np.array([10., 10.]))

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:
        attacker_action = self.attacker.predict(self.attacker_obs)
        action_and_noise = action * (1. + attacker_action[2:]) if self.compromise_actuation else action

        obs, reward, done, info = self.env.step(action_and_noise)
        self.attacker_obs = np.append(obs[:2], np.array([np.float(self.compromise_actuation), np.float(self.compromise_observation)]))

        info['a'] = attacker_action[2:]
        info['o'] = attacker_action[:2]

        defender_obs = obs[:2] * (1. + attacker_action[:2]) if self.compromise_observation else obs[:2]

        return np.append(defender_obs[:2], np.array([np.float(self.compromise_actuation), np.float(self.compromise_observation)])), reward, done, info

    def reset(self) -> Any:
        self.attacker_obs = super().reset()
        return self.attacker_obs

    def render(self, mode='human') -> None:
        raise NotImplementedError

