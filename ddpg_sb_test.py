from stable_baselines import DDPG
from stable_baselines.common import make_vec_env
import gym
import envs
import uuid


def get_env():
    # env = Environment.create(environment='gym',
    #                          level='TT-v0')
    env = gym.make('TT-v0')
    return env


concurrent_runs = 8
id = f'{uuid.uuid4().__str__().split("-")[0]}'
print(f'ID: {id}')

model = DDPG(
    'LnMlpPolicy',
    make_vec_env('TT-v0', n_envs=concurrent_runs)
)

model.learn(500_000)