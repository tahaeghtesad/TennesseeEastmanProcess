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

        self.action_dim = 2
        self.observation_dim = 2

        self.episode_count = 0
        self.step_count = 0

        self.highest_reward = -np.inf

        self.goal = np.array([0.99510292, 1.5122427])  # Unstable
        self.noise = noise

        self.win_count = 0

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:  # Obs, Reward, Done, Info
        self.step_count += 1

        u = action * (1. + np.random.normal(loc=np.zeros(self.action_dim,), scale=np.array([0.00, 0.07]))) if self.noise else action

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

        return self.x * (1. + np.random.normal(loc=np.zeros(self.observation_dim,), scale=np.array([0.00, 0.07]))) if self.noise else self.x, reward, win, {
            'u': u,
            'x': self.x,
            'dx': dx
        }

    def reset(self) -> Any:
        self.highest_reward = -np.inf

        self.episode_count += 1
        self.step_count = 0
        self.x = self.observation_space.sample()
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

        self.compromise = None

        self.env = gym.make('BRP-v0')

    def reset(self) -> Any:
        obs = self.env.reset()

        self.compromise = np.concatenate(
            (np.random.rand(self.env.observation_dim) < self.compromise_observation_prob,
             np.random.rand(self.env.action_dim) < self.compromise_actuation_prob)
            , axis=0).astype(np.float)

        return np.concatenate((obs, self.compromise), axis=0)


class BioReactorAttacker(AdversarialBioReactor):  # This is a noise generator attacker.

    def __init__(self, defender, compromise_actuation_prob: float, compromise_observation_prob: float, power: float = 0.3) -> None:
        super().__init__(compromise_actuation_prob, compromise_observation_prob)
        self.logger = logging.getLogger(__class__.__name__)
        self.defender = defender

        self.observation_space = gym.spaces.Box(low=np.array([-10., -10.] + [0.] * (self.env.action_dim + self.env.observation_dim)),
                                                high=np.array([10., 10.]  + [0.] * (self.env.action_dim + self.env.observation_dim)))
        self.action_space = gym.spaces.Box(low=-power * np.array([1.] * (self.env.action_dim + self.env.observation_dim)),
                                           high=power * np.array([1.] * (self.env.action_dim + self.env.observation_dim)))

        self.defender_obs = np.zeros((4,))

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:

        defender_action = self.defender.predict(self.defender_obs)
        action_and_noise = defender_action * (1. + action[self.env.observation_dim:] * self.compromise[self.env.observation_dim:])

        obs, reward, done, info = self.env.step(action_and_noise)
        self.defender_obs = np.concatenate((obs * (1. + action[:self.env.observation_dim] * self.compromise[:self.env.observation_dim]), self.compromise), axis=0)

        info['a'] = action[self.env.observation_dim:]
        info['o'] = action[:self.env.observation_dim]

        return np.concatenate((obs, self.compromise), axis=0), -reward, done, info

    def reset(self) -> Any:
        self.defender_obs = super().reset()
        return self.defender_obs

    def render(self, mode='human') -> None:
        raise NotImplementedError()


class BioReactorDefender(AdversarialBioReactor):

    def __init__(self, attacker, compromise_actuation_prob: float, compromise_observation_prob: float, power: float = 0.3) -> None:
        super().__init__(compromise_actuation_prob, compromise_observation_prob)
        self.logger = logging.getLogger(__class__.__name__)
        self.attacker = attacker
        self.attacker_obs = None

        self.observation_space = gym.spaces.Box(low=np.array([-10., -10.] + [0.] * (self.env.action_dim + self.env.observation_dim)),
                                                high=np.array([10., 10.] + [1.] * (self.env.action_dim + self.env.observation_dim)))
        self.action_space = gym.spaces.Box(low=-np.array([10., 10.]),
                                           high=np.array([10., 10.]))

        self.attacker_power = power

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:
        attacker_action = self.attacker.predict(self.attacker_obs)
        assert (-self.attacker_power <= attacker_action).all() and (attacker_action <= self.attacker_power).all()
        action_and_noise = action * (1. + attacker_action[self.env.observation_dim:] * self.compromise[self.env.observation_dim:])

        obs, reward, done, info = self.env.step(action_and_noise)
        self.attacker_obs = np.concatenate((obs, self.compromise), axis=0)

        info['a'] = attacker_action[self.env.observation_dim:]
        info['o'] = attacker_action[:self.env.observation_dim]

        defender_obs = obs * (1. + attacker_action[:self.env.observation_dim] * self.compromise[:self.env.observation_dim])

        return np.concatenate((defender_obs, self.compromise), axis=0), reward, done, info

    def reset(self) -> Any:
        self.attacker_obs = super().reset()
        return self.attacker_obs

    def render(self, mode='human') -> None:
        raise NotImplementedError

