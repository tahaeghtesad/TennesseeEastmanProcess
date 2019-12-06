from stable_baselines import DDPG
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from gym_control.envs.BioReactorEnv import BioReactor
import numpy as np
import matplotlib.pyplot as plt

env = BioReactor()

x0 = []
x1 = []

obs = env.reset()
for _ in range(1, 100):
    done = False
    while not done:
        obs, reward, done, _ = env.step(np.array([1., 1.]))
        x0.append(obs[0])
        x1.append(obs[1])
        # print(obs, reward)
    env.reset()

plt.plot(x0)
plt.plot(x1)
plt.show()