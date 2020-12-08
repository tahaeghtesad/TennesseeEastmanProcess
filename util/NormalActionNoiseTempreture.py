from stable_baselines.common.noise import NormalActionNoise
import numpy as np


class NormalActionNoiseTemperature(NormalActionNoise):
    """
    A Gaussian action noise

    :param mean: (float) the mean value of the noise
    :param sigma: (float) the scale of the noise (std here)
    :param temperature: (float) temperature
    """

    def __init__(self, mean, sigma, temperature):
        super().__init__(mean, sigma)
        self._mu = mean
        self._sigma = sigma
        self._tau = temperature

        self.step = 1

    def reset(self):
        self.step = 1

    def __call__(self) -> np.ndarray:
        self.step += 1
        return np.random.normal(self._mu, self._sigma) * self._tau ** self.step

    def __repr__(self) -> str:
        return f'NormalActionNoiseTemperature(mu={self._mu}, sigma={self._sigma}, tau={self._tau})'
