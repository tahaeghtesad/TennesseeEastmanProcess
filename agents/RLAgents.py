import logging
from typing import List

import numpy as np


class Agent:

    def __init__(self) -> None:
        super().__init__()

    def predict(self, observation, state=None, mask=None, deterministic=True):
        raise NotImplementedError()

    def reset(self):
        pass


class MixedStrategyAgent(Agent):

    def __init__(self):
        super().__init__()
        self.probabilities = None
        self.policies = []

    def update_probabilities(self, probabilities):
        self.probabilities = probabilities / probabilities.sum()

    def add_policy(self, policy: Agent):
        self.policies.append(policy)

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return np.random.choice(self.policies, p=self.probabilities).predict(observation, state, mask, deterministic)


class SinglePolicyMixedStrategyAgent(MixedStrategyAgent):

    def __init__(self):
        super().__init__()
        self.current_policy = None

    def reset(self):
        self.current_policy = np.random.choice(self.policies, p=self.probabilities)

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return self.current_policy.predict(observation, state, mask, deterministic)


class SimpleWrapperAgent(Agent):

    def __init__(self, agent: [Agent]) -> None:
        super().__init__()
        self.agent = agent

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return self.agent.predict(observation, state, mask, deterministic)[0]

    def save(self, save_path, cloudpickle=False):
        return self.agent.save(save_path, cloudpickle)


class HistoryAgent(SimpleWrapperAgent):

    def __init__(self, agent: [Agent], observation_dim=2, history_length=12) -> None:
        super().__init__(agent)
        self.history_length = history_length
        self.observation_dim = observation_dim
        self.agent = agent
        self.history = []
        self.compromise = None

    def add_history(self, observation):
        self.history.append(observation)
        if len(self.history) == 1:
            self.history = [observation] * self.history_length
        if len(self.history) > self.history_length:
            del self.history[0]

    def reset(self):
        self.history = []

    def predict(self, observation, state=None, mask=None, deterministic=True):
        self.add_history(observation)
        return self.agent.predict(np.array(self.history).flatten(), state, mask, deterministic)[0]


class LimitedHistoryAgent(HistoryAgent):

    def __init__(self, agent: [Agent], observation_dim=2, history_length=12, select=None) -> None:
        super().__init__(agent, observation_dim, history_length)
        self.select = [0, 4, 11] if select is None else select

    def predict(self, observation, state=None, mask=None, deterministic=True):
        self.add_history(observation)
        return self.agent.predict(np.array(self.history)[self.select].flatten(), state, mask, deterministic)[0]


class NoOpAgent(Agent):

    def __init__(self, action_dim) -> None:
        super().__init__()
        self.action_dim = action_dim

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return np.zeros(self.action_dim)
