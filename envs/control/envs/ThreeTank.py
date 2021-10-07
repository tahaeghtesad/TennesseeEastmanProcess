import logging
from typing import *

import gym
import numpy as np

from agents.RLAgents import Agent, ConstantAgent
from envs.control.adversarial_control import AdversarialControlEnv
from envs.control.control_env import ControlEnv


class ThreeTank(ControlEnv):

    def __init__(self, test_env=False, noise_sigma=0.05, t_epoch=200) -> None:
        super().__init__(test_env, noise_sigma, t_epoch)
        self.logger = logging.getLogger(__class__.__name__)
        self.x = np.array([0., 0., 0.])

        self.action_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.5e-4, 1.5e-4]))
        self.observation_space = gym.spaces.Box(low=-np.array([0, 0]), high=np.array([1., 1.]))

        self.tank_size = gym.spaces.Box(low=np.array([0., 0., 0]), high=np.array([.62, .62, .62]))

        self.episode_count = 0
        self.step_count = 0

        self.highest_reward = -np.inf

        self.goal = np.array([.4, .2, .3])

        self.win_count = 0

    def q13(self, mu13=0.5, sn=0.00005):
        return np.sign(self.x[0] - self.x[2]) * np.sqrt(2 * 9.8 * np.abs(self.x[0] - self.x[2])) * mu13 * sn

    def q32(self, mu32=0.5, sn=0.00005):
        return np.sign(self.x[2] - self.x[1]) * np.sqrt(2 * 9.8 * np.abs(self.x[2] - self.x[1])) * mu32 * sn

    def q20(self, mu20=0.6, sn=0.00005):
        return np.sqrt(2 * 9.8 * (self.x[1] if self.x[1] >= 0 else 0)) * mu20 * sn

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:  # Obs, Reward, Done, Info
        self.step_count += 1

        u = action * (1. + np.random.normal(loc=np.zeros(self.action_dim,), scale=np.repeat(self.noise_sigma, self.action_dim)))

        dx = np.array([
            (u[0] - self.q13()) / 0.0154,
            (u[1] + self.q32() - self.q20()) / 0.0154,
            (self.q13() - self.q32()) / 0.0154
        ])

        self.x = self.x + dx * 10.0
        self.x = np.clip(self.x, self.tank_size.low, self.tank_size.high)

        win = self.t_epoch == self.step_count
        reward = -np.linalg.norm(self.x - self.goal)
        # if np.linalg.norm(self.x - self.goal) < self.proximity:
        #     reward += 100
        #     win = True

        if win:
            self.win_count += 1

        obs = self.x[:2]

        return obs * (1. + np.random.normal(loc=np.zeros(self.action_dim,), scale=np.repeat(self.noise_sigma, self.observation_dim))), reward, win, {
            'u': u,
            'x': self.x,
            'dx': dx
        }

    def reset(self) -> Any:
        self.highest_reward = -np.inf

        self.episode_count += 1
        self.step_count = 0

        if self.test_env:
            self.x = self.goal.copy()
        else:
            self.x = self.goal * (1. + np.random.normal(loc=np.zeros(3, ), scale=np.array([.3, .3, .3])))

        self.logger.debug(f'Reset... Starting Point: {self.x}')
        return self.x[:2]

    def render(self, mode='human') -> None:
        raise NotImplementedError()


class ThreeTankAttacker(gym.Env):

    def __init__(self, defender: Agent, compromise_actuation_prob: float, compromise_observation_prob: float,
                 power: float = 0.3, noise_sigma=0.07,
                 history_length=12, include_compromise=True, test_env=False, t_epoch=50) -> None:
        self.include_compromise = include_compromise
        self.logger = logging.getLogger(__class__.__name__)

        self.adversarial_control_env = AdversarialControlEnv('TT-v0', None, defender, compromise_actuation_prob,
                                                             compromise_observation_prob, history_length,
                                                             include_compromise, noise_sigma, t_epoch, power, test_env)

        self.observation_space = gym.spaces.Box(
            low=np.array([0., 0.] * history_length + [0.] * (self.adversarial_control_env.env.action_dim + self.adversarial_control_env.env.observation_dim)) if self.include_compromise else np.array([0., 0.] * history_length),
            high=np.array([1., 1.] * history_length + [0.] * (self.adversarial_control_env.env.action_dim + self.adversarial_control_env.env.observation_dim)) if self.include_compromise else np.array([1., 1.] * history_length))

        self.action_space = gym.spaces.Box(
            low=-power * np.array([1.] * (self.adversarial_control_env.env.action_dim + self.adversarial_control_env.env.observation_dim)),
            high=power * np.array([1.] * (self.adversarial_control_env.env.action_dim + self.adversarial_control_env.env.observation_dim)))

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:
        self.adversarial_control_env.set_attacker(ConstantAgent(action))
        (obs, _), (reward, _), done, info = self.adversarial_control_env.step()

        return obs, reward, done, info

    def reset(self) -> Any:
        return self.adversarial_control_env.reset()[0]


class ThreeTankDefender(gym.Env):

    def __init__(self, attacker: Agent, compromise_actuation_prob: float, compromise_observation_prob: float,
                 power: float = 0.3, noise_sigma=0.07, test_env=False,
                 history_length=12, include_compromise=True, t_epoch=50) -> None:
        self.include_compromise = include_compromise
        self.logger = logging.getLogger(__class__.__name__)

        self.adversarial_control_env = AdversarialControlEnv('TT-v0', attacker, None, compromise_actuation_prob,
                                                             compromise_observation_prob, history_length,
                                                             include_compromise, noise_sigma, t_epoch, power, test_env)

        self.observation_space = gym.spaces.Box(
            low=np.tile(self.adversarial_control_env.env.observation_space.low, history_length),
            high=np.tile(self.adversarial_control_env.env.observation_space.high, history_length))
        if include_compromise:
            self.observation_space = gym.spaces.Box(
                low=np.append(self.observation_space.low, np.zeros(
                    self.adversarial_control_env.env.action_dim + self.adversarial_control_env.env.observation_dim)),
                high=np.append(self.observation_space.high, np.ones(
                    self.adversarial_control_env.env.action_dim + self.adversarial_control_env.env.observation_dim)))

        self.action_space = gym.spaces.Box(low=self.adversarial_control_env.env.action_space.low,
                                           high=self.adversarial_control_env.env.action_space.high)

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:
        self.adversarial_control_env.set_defender(ConstantAgent(action))
        (_, obs), (_, reward), done, info = self.adversarial_control_env.step()

        return obs, reward, done, info

    def reset(self) -> Any:
        return self.adversarial_control_env.reset()[1]

    def render(self, mode='human') -> None:
        raise NotImplementedError