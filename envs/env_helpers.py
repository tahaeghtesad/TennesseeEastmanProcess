import gym
import envs
import numpy as np


class Historitized(gym.Env):
    def __init__(self, env, history_size=12, **kwargs) -> None:
        self.history_size = history_size
        self.env = gym.make(env, **kwargs)

        self.history = []

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=np.tile(self.env.observation_space.low, self.history_size),
                                                high=np.tile(self.env.observation_space.high, self.history_size))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.history.append(obs)
        if len(self.history) > self.history_size:
            del self.history[0]
        return np.array(self.history).flatten(), reward, done, info

    def reset(self):
        initial_obs = self.env.reset()
        self.history = [initial_obs] * self.history_size
        return np.array(self.history).flatten()


class LimitedHistoritized(Historitized):

    def __init__(self, env, history_size=12, select=None, **kwargs) -> None:
        super().__init__(env, history_size, **kwargs)
        self.select = [0, 4, 11] if select is None else select
        self.observation_space = gym.spaces.Box(low=np.tile(self.env.observation_space.low, len(self.select)),
                                                high=np.tile(self.env.observation_space.high, len(self.select)))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.history.append(obs)
        if len(self.history) > self.history_size:
            del self.history[0]
        return np.array(self.history)[self.select].flatten(), reward, done, info

    def reset(self):
        initial_obs = self.env.reset()
        self.history = [initial_obs] * self.history_size
        return np.array(self.history)[self.select].flatten()
