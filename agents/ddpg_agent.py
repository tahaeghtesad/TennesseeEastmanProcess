from stable_baselines import DDPG
import numpy as np
import sys
import os
from tqdm import tqdm


class DDPGWrapper(DDPG):
    def predict(self, observation, state=None, mask=None, deterministic=True):
        try:
            return super().predict(observation, state, mask, deterministic)[0]
        except ValueError:
            return super().predict(observation[:2], state, mask, deterministic)[0]


class DDPGWrapperHistory(DDPG):

    @classmethod
    def load(cls, load_path, observation_dim=2, history_length=12, env=None, custom_objects=None, **kwargs):
        class_ = super().load(load_path, env, custom_objects, **kwargs)
        class_.observation_dim = observation_dim
        class_.history_length = history_length
        class_.history = []
        return class_

    def predict(self, observation, state=None, mask=None, deterministic=True):
        self.history.append(observation)
        if len(self.history) == 1:
            self.history = [observation] * self.history_length
        if len(self.history) > self.history_length:
            del self.history[0]
        try:
            return super().predict(np.array(self.history).flatten(), state, mask, deterministic)[0]
        except ValueError:
            return super().predict(np.array(self.history)[:, :self.observation_dim].flatten(), state, mask, deterministic)[0]


class MixedStrategyDDPG:

    def __init__(self, path, count, probabilities, history=True):
        assert len(probabilities) == count
        self.probabilities = probabilities / probabilities.sum()
        self.count = count
        self.policies = []

        for i in tqdm(range(count)):
            # print(f'Loading Policy {path}-{i}')
            sys.stdout = open(os.devnull, "w")
            if history:
                self.policies.append(DDPGWrapperHistory.load(f'{path}-{i}'))
            else:
                self.policies.append(DDPGWrapper.load(f'{path}-{i}'))
            sys.stdout = sys.__stdout__


    def predict(self, observation, state=None, mask=None, deterministic=True):
        return np.random.choice(self.policies, p=self.probabilities).predict(observation, state, mask, deterministic)