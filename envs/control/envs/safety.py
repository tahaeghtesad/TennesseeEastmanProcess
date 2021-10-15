import logging
from typing import Tuple, Any, Dict

import gym
import numpy as np

from agents.RLAgents import Agent, ConstantAgent
from envs.control.threat.safety_theat import SafetyThreatModel


class SafetyEnvAttacker(gym.Env):

    def __init__(self, env, defender: Agent) -> None:
        self.logger = logging.getLogger(__class__.__name__)

        self.adversarial_control_env = SafetyThreatModel(env, attacker=None, defender=defender)
        # Adversary receives Lidar (16,) and defender's action
        self.observation_space = gym.spaces.Box(low=self.adversarial_control_env.env.action_space.low + [0] * 16,
                                                high=self.adversarial_control_env.env.action_space.high + [1] * 16)
        self.action_space = self.adversarial_control_env.env.action_space

        self.defender_obs = np.zeros((4,))

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:
        self.adversarial_control_env.set_attacker(ConstantAgent(action))
        (obs, _), (reward, _), done, info = self.adversarial_control_env.step()

        return obs, reward, done, info

    def reset(self) -> Any:
        return self.adversarial_control_env.reset()[0]

    def render(self, mode='human') -> None:
        raise NotImplementedError()


class SafetyEnvDefender(gym.Env):

    def __init__(self, env, attacker: Agent) -> None:
        self.logger = logging.getLogger(__class__.__name__)

        self.adversarial_control_env = SafetyThreatModel(env, attacker=attacker, defender=None)
        self.action_space = self.adversarial_control_env.env.action_space
        self.observation_space = self.adversarial_control_env.env.observation_space

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:
        self.adversarial_control_env.set_defender(ConstantAgent(action))
        (_, obs), (_, reward), done, info = self.adversarial_control_env.step()

        return obs, reward, done, info

    def reset(self) -> Any:
        return self.adversarial_control_env.reset()[1]

    def render(self, mode='human') -> None:
        raise NotImplementedError
