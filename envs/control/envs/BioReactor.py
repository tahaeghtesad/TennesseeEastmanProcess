import gym
import numpy as np
import logging
from typing import *

from agents.RLAgents import Agent, ConstantAgent
from envs.control.adversarial_control import AdversarialControlEnv
from envs.control.control_env import ControlEnv


class BioReactor(ControlEnv):

    def __init__(self, test_env=False, noise_sigma=0.05, t_epoch=50) -> None:
        super().__init__(test_env, noise_sigma)
        self.logger = logging.getLogger(__class__.__name__)
        self.x = np.array([0., 0.])
        self.t_epoch = t_epoch

        self.action_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([3.0, 8.0]))
        self.observation_space = gym.spaces.Box(low=np.array([0.00, 0.00]), high=np.array([10., 10.]))

        # self.observation_space = gym.spaces.Box(low=np.array([0.6, 1] * history_length + [0.] * (self.adversarial_control_env.env.action_dim + self.adversarial_control_env.env.observation_dim)) if self.include_compromise else np.array([-5., -5.] * history_length),
        #                                         high=np.array([1.2, 2.] * history_length + [1.] * (self.adversarial_control_env.env.action_dim + self.adversarial_control_env.env.observation_dim)) if self.include_compromise else np.array([5., 5.] * history_length))
        # self.action_space = gym.spaces.Box(low=-np.array([0.0, 3.0]),
        #                                    high=np.array([1.0, 6.0]))

        self.action_dim = 2
        self.observation_dim = 2

        self.episode_count = 0
        self.step_count = 0

        self.highest_reward = -np.inf

        self.goal = np.array([0.99510292, 1.5122427])  # Unstable

        self.win_count = 0

    def clip(self, state):
        return np.clip(state, self.observation_space.low, self.observation_space.high)

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:  # Obs, Reward, Done, Info
        self.step_count += 1

        u = action * (1. + np.random.normal(loc=np.zeros(self.action_dim,), scale=np.repeat(self.noise_sigma, self.action_dim)))

        dx = np.array([
            (mu(self.x[1]) - u[0]) * self.x[0],
            u[0] * (u[1] - self.x[1]) - mu(self.x[1]) / 0.4 * self.x[0]
        ])

        self.x = self.x * 0.1 + dx
        self.x = self.clip(self.x)

        win = self.t_epoch == self.step_count
        reward = -np.linalg.norm(self.x - self.goal)
        # if np.linalg.norm(self.x - self.goal) < 0.01:
        #     reward += 5.
        #     win = True

        if win:
            self.win_count += 1

        return self.x * (1. + np.random.normal(loc=np.zeros(self.action_dim,), scale=np.repeat(self.noise_sigma, self.action_dim))), reward, win, {
            'u': u,
            'x': self.x,
            'dx': dx
        }

    def reset(self) -> Any:
        self.highest_reward = -np.inf

        self.episode_count += 1
        self.step_count = 0
        # self.x = self.goal.copy()
        if self.test_env:
            self.x = self.goal.copy()
        else:
            self.x = self.goal * (1. + np.random.normal(loc=np.zeros(2, ), scale=np.array([.3, .3])))
        self.logger.debug(f'Reset... Starting Point: {self.x}')
        return self.x

    def render(self, mode='human') -> None:
        raise NotImplementedError()


def mu(x2: float, mu_max: float = 0.53, km: float = 0.12, k1: float = 0.4545) -> float:
    return mu_max * (x2 / (km + x2 + k1 * x2 * x2))


class BioReactorAttacker(gym.Env):  # This is a noise generator attacker.

    def __init__(self, defender: Agent, compromise_actuation_prob: float, compromise_observation_prob: float,
                 power: float = 0.3, noise_sigma=0.07,
                 history_length=12, include_compromise=True, test_env=False, t_epoch=50) -> None:
        self.include_compromise = include_compromise
        self.logger = logging.getLogger(__class__.__name__)

        self.adversarial_control_env = AdversarialControlEnv('BRP-v0', None, defender, compromise_actuation_prob,
                                                             compromise_observation_prob, history_length,
                                                             include_compromise, noise_sigma, t_epoch, power, test_env)

        self.observation_space = gym.spaces.Box(low=np.array([-5., -5.] * history_length + [0.] * (
                    self.adversarial_control_env.env.action_dim + self.adversarial_control_env.env.observation_dim)) if self.include_compromise else np.array(
            [-5., -5.] * history_length),
                                                high=np.array([5., 5.] * history_length + [0.] * (
                                                            self.adversarial_control_env.env.action_dim + self.adversarial_control_env.env.observation_dim)) if self.include_compromise else np.array(
                                                    [-5., -5.] * history_length))
        self.action_space = gym.spaces.Box(low=-power * np.array(
            [1.] * (self.adversarial_control_env.env.action_dim + self.adversarial_control_env.env.observation_dim)),
                                           high=power * np.array([1.] * (
                                                       self.adversarial_control_env.env.action_dim + self.adversarial_control_env.env.observation_dim)))

        self.defender_obs = np.zeros((4,))

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:
        self.adversarial_control_env.set_attacker(ConstantAgent(action))
        (obs, _), (reward, _), done, info = self.adversarial_control_env.step()

        return obs, reward, done, info

    def reset(self) -> Any:
        self.adversarial_control_env.defender.reset()
        return self.adversarial_control_env.reset()[0]

    def render(self, mode='human') -> None:
        raise NotImplementedError()


class BioReactorDefender(gym.Env):

    def __init__(self, attacker: Agent, compromise_actuation_prob: float, compromise_observation_prob: float,
                 power: float = 0.3, noise_sigma=0.07, test_env=False,
                 history_length=12, include_compromise=True, t_epoch=50) -> None:
        self.include_compromise = include_compromise
        self.logger = logging.getLogger(__class__.__name__)

        self.adversarial_control_env = AdversarialControlEnv('BRP-v0', attacker, None, compromise_actuation_prob,
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
        self.adversarial_control_env.attacker.reset()
        return self.adversarial_control_env.reset()[1]

    def render(self, mode='human') -> None:
        raise NotImplementedError
