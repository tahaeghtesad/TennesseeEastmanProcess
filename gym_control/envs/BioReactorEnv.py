import gym
import numpy as np
from typing import *


class BioReactor(gym.Env):

    reward_range = (-float(1.0), float(1.0))
    action_space = gym.spaces.Box(low=-float(5), high=float(5), shape=(2,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(2,), dtype=np.float32)

    def __init__(self,
                 ) -> None:
        super().__init__()

        self.x = np.zeros((2, 1))
        self.win_counter = 0

        self.target = np.array([[0.95103], [1.512243]])

    def step(self, input: np.ndarray) -> Tuple[Any, float, bool, Dict]:  # Obs, Reward, Done, Info

        u = input * np.array([0.3, 4.])
        # u += np.array([.5, 2.5])

        dx = np.array([
            [(BioReactor.mu(self.x[1][0]) - u[0]) * self.x[0][0]],
            [u[0] * (u[1] - self.x[1][0]) - BioReactor.mu(self.x[1][0]) / 0.4 * self.x[0][0]]
        ])

        self.x += .001 * dx

        if np.linalg.norm(self.x - self.target) < 0.01:
            self.win_counter += 1
        else:
            self.win_counter = 0

        y = self.x  # * np.random.rand(2, 1)
        win = self.win_counter > 8
        lose = np.isnan(self.x).any() or (np.abs(self.x) > 10 ** 3).any()  # or self.x[0][0] < 0 or self.x[1][0] <= 0.
        # reward = np.tanh(u[0]) if win or not lose else -1
        # reward = np.exp(-np.linalg.norm(self.dx)) if win or not lose else -.1
        # reward = np.exp(-np.linalg.norm(self.x - self.target))
        reward = -np.linalg.norm(self.x - self.target)

        if win or lose:
            print(f'x0={self.x[0][0]:.3f}-x1={self.x[1][0]:.4f} \t\t dx={dx[0][0]:.4f}:{dx[1][0]:.4f} \t r={reward:.4f} \t d={input[0]}-x2f={input[1]} \t win={win} \t lose={lose}')

        return np.transpose(y)[0], reward, win or lose, {
            'x': self.x,
            'dx': dx,
            'u': u,
            'win': win,
            'lose': lose
        }

    def reset(self) -> Any:  # Obs
        self.win_counter = 0
        self.x = np.random.rand(2, 1) * np.array([[1.5], [5]])
        # self.x = np.array(self.target)
        y = self.x
        return np.transpose(y)[0]

    def render(self, mode='human') -> None:
        raise NotImplementedError()

    @staticmethod
    def mu(x2: float, mu_max: float=0.53, km: float=0.12, k1=0.4545) -> float:
        return mu_max * (x2/ (km + x2 + k1 * x2 * x2))

