import logging
from typing import List

import numpy as np


class Agent:

    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    def predict(self, observation, state=None, mask=None, deterministic=True):
        raise NotImplementedError()

    def __repr__(self):
        return self.name

    def reset(self):
        pass


class MixedStrategyAgent(Agent):

    def __init__(self, name):
        super().__init__(name)
        self.probabilities = None
        self.policies = []

    def update_probabilities(self, probabilities):
        self.probabilities = probabilities / probabilities.sum()

    def add_policy(self, policy: Agent):
        self.policies.append(policy)

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return np.random.choice(self.policies, p=self.probabilities).predict(observation, state, mask, deterministic)


class SinglePolicyMixedStrategyAgent(MixedStrategyAgent):

    def __init__(self, name):
        super().__init__(name)
        self.current_policy = None

    def reset(self):
        self.current_policy = np.random.choice(self.policies, p=self.probabilities)

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return self.current_policy.predict(observation, state, mask, deterministic)


class SimpleWrapperAgent(Agent):

    def __init__(self, name, agent: [Agent]) -> None:
        super().__init__(name)
        self.agent = agent

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return self.agent.predict(observation, state, mask, deterministic)[0]

    def save(self, save_path, cloudpickle=False):
        return self.agent.save(save_path, cloudpickle)


class ConstantAgent(Agent):

    def __init__(self, name, action) -> None:
        super().__init__(name)
        self.action = action

    def __repr__(self):
        return f'{self.name}-{self.action}'

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return self.action


class ZeroAgent(ConstantAgent):

    def __init__(self, name, action_dim):
        super().__init__(name, np.zeros(action_dim))
