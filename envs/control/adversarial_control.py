import logging
from typing import Tuple, Any

import numpy as np
import gym

from agents.RLAgents import Agent
from envs.control.control_env import ControlEnv

class AdversarialControlEnv:
    def __init__(self,
                 env,
                 attacker: Agent,
                 defender: Agent,
                 compromise_actuation_prob: float,
                 compromise_observation_prob: float,
                 history_length=12,
                 include_compromise=True,
                 noise_sigma=0.07,
                 t_epoch=50,
                 power=.3,
                 test_env=False) -> None:

        super().__init__()
        self.attacker = attacker
        self.defender = defender
        self.logger = logging.getLogger(__class__.__name__)
        self.compromise_actuation_prob = compromise_actuation_prob
        self.compromise_observation_prob = compromise_observation_prob
        self.history_length = history_length
        self.include_compromise = include_compromise
        self.power = power

        self.compromise = None
        self.attacker_history = None
        self.defender_history = None

        if isinstance(env, str):
            self.env = gym.make(f'{env}', noise_sigma=noise_sigma, test_env=test_env, t_epoch=t_epoch)
        elif isinstance(env, ControlEnv):
            self.env = env
        else:
            raise Exception('Invalid Environment.')

    def set_attacker(self, attacker):
        self.attacker = attacker

    def set_defender(self, defender):
        self.defender = defender

    def step(self) -> Any:

        attacker_action = self.attacker.predict(np.concatenate((np.array(self.attacker_history).flatten(), self.compromise), axis=0) if self.include_compromise else np.array(self.attacker_history).flatten())
        defender_action = self.defender.predict(np.concatenate((np.array(self.defender_history).flatten(), self.compromise), axis=0) if self.include_compromise else np.array(self.defender_history).flatten())

        action_and_noise = defender_action * (
                    1. + attacker_action[self.env.observation_dim:] * self.compromise[self.env.observation_dim:])

        obs, reward, done, info = self.env.step(action_and_noise)
        defender_obs = obs * (
                    1. + attacker_action[:self.env.observation_dim] * self.compromise[:self.env.observation_dim])

        self.attacker_history += [obs]
        self.defender_history += [defender_obs]

        if len(self.attacker_history) > self.history_length:
            del self.attacker_history[0]
        if len(self.defender_history) > self.history_length:
            del self.defender_history[0]

        info['a'] = attacker_action[self.env.observation_dim:]
        info['o'] = attacker_action[:self.env.observation_dim]
        info['d'] = defender_action
        info['c'] = self.compromise

        return (np.concatenate((np.array(self.attacker_history).flatten(), self.compromise), axis=0) if self.include_compromise else np.array(self.attacker_history).flatten(),
                np.concatenate((np.array(self.defender_history).flatten(), self.compromise), axis=0) if self.include_compromise else np.array(self.defender_history).flatten()),\
               (-reward, reward), done, info

    def reset(self, compromise=None):
        obs = self.env.reset()

        self.compromise = np.concatenate(
            (np.random.rand(self.env.observation_dim) < self.compromise_observation_prob,
             np.random.rand(self.env.action_dim) < self.compromise_actuation_prob)
            , axis=0).astype(np.float) if compromise is None else compromise

        self.defender_history = [obs] * self.history_length
        self.attacker_history = [obs] * self.history_length

        return np.concatenate((np.array(self.attacker_history).flatten(), self.compromise), axis=0) if self.include_compromise else np.array(self.attacker_history).flatten(),\
               np.concatenate((np.array(self.defender_history).flatten(), self.compromise), axis=0) if self.include_compromise else np.array(self.defender_history).flatten()

    def get_compromise(self):
        return self.compromise

    def set_compromise(self, compromise):
        self.compromise = compromise