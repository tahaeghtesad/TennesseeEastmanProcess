from agents.RLAgents import Agent
import numpy as np


class AlternatingAttacker(Agent):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.action = 0.3 * np.ones(dim)

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return self.action

    def reset(self):
        self.action = -self.action


class RandomMoveAttacker(AlternatingAttacker):

    def __init__(self, dim) -> None:
        super().__init__(dim)

    def reset(self):
        self.action = 0.3 * np.random.choice(np.array([1., -1.]), self.dim)
