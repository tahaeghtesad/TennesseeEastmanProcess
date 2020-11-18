import logging

import numpy as np

from agents.RLAgents import Agent


class Trainer:
    def __init__(self,
                 prefix,
                 env_params=None,
                 rl_params=None,
                 policy_params=None,
                 training_params=None) -> None:
        self.policy_params = policy_params
        self.prefix = prefix
        self.env_params = env_params
        self.rl_params = rl_params
        self.training_params = training_params

        # Attacker is the row player
        # Defender is the column player
        self.attacker_payoff_table = np.array([[]])
        self.defender_payoff_table = np.array([[]])

        self.logger = logging.getLogger(__name__)

    @staticmethod
    def callback(locals_, globals_):
        raise NotImplementedError()

    def get_policy_class(self, policy_params):
        raise NotImplementedError()

    def train_attacker(self, defender, iteration):
        raise NotImplementedError()

    def train_defender(self, attacker, iteration):
        raise NotImplementedError()

    def get_payoff(self, attacker: Agent, defender: Agent):
        raise NotImplementedError()

    def update_defender_payoff_table(self, attacker_util, defender_util):
        self.attacker_payoff_table = np.hstack([self.attacker_payoff_table, attacker_util.reshape((attacker_util.size, 1))])
        self.defender_payoff_table = np.hstack([self.defender_payoff_table, defender_util.reshape((defender_util.size, 1))])
        self.save_tables()

    def update_attacker_payoff_table(self, attacker_util, defender_util):
        self.attacker_payoff_table = np.vstack([self.attacker_payoff_table, attacker_util.reshape((1, attacker_util.size))])
        self.defender_payoff_table = np.vstack([self.defender_payoff_table, defender_util.reshape((1, defender_util.size))])
        self.save_tables()

    def save_tables(self):
        np.save(f'{self.prefix}/attacker_payoff', self.attacker_payoff_table)
        np.save(f'{self.prefix}/defender_payoff', self.defender_payoff_table)

    def load_tables(self):
        self.attacker_payoff_table = np.load(f'{self.prefix}/attacker_payoff.npy')
        self.defender_payoff_table = np.load(f'{self.prefix}/defender_payoff.npy')

    def get_tables(self):
        return self.attacker_payoff_table, self.defender_payoff_table

    def initialize_strategies(self):
        raise NotImplementedError()
