from typing import List

import numpy as np
from stable_baselines.common import BaseRLModel


class Agent:
    def predict(self, observation, state=None, mask=None, deterministic=True):
        raise NotImplementedError()


class MixedStrategyAgent(Agent):

    def __init__(self):
        self.probabilities = None
        self.policies = []

    def update_probabilities(self, probabilities):
        self.probabilities = probabilities

    def add_policy(self, policy: Agent):
        self.policies.append(policy)

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return np.random.choice(self.policies, p=self.probabilities).predict(observation, state, mask, deterministic)


class SimpleWrapperAgent(Agent):

    def __init__(self, agent: [Agent, BaseRLModel]) -> None:
        super().__init__()
        self.agent = agent

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return self.agent.predict(observation, state, mask, deterministic)[0]

    def save(self, save_path, cloudpickle=False):
        return self.agent.save(save_path, cloudpickle)


class HistoryAgent(Agent):

    def __init__(self, agent: [Agent, BaseRLModel], observation_dim=2, history_length=12) -> None:
        super().__init__(agent)
        self.history_length = history_length
        self.observation_dim = observation_dim
        self.agent = agent
        self.history = []

    def predict(self, observation, state=None, mask=None, deterministic=True):
        self.history.append(observation)
        if len(self.history) == 1:
            self.history = [observation] * self.history_length
        if len(self.history) > self.history_length:
            del self.history[0]
        try:
            return self.agent.predict(np.array(self.history).flatten(), state, mask, deterministic)
        except ValueError:
            return \
                self.agent.predict(np.array(self.history)[:, :self.observation_dim].flatten(), state, mask,
                                   deterministic)[0]
