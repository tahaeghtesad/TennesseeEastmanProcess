from stable_baselines import DDPG
import numpy as np
import gym
import gym_control
from tqdm import tqdm
from gym_control.envs.BioReactor import AttackerMode
from agents.ddpg_agent import DDPGWrapper
from os import listdir
from os.path import isfile, join


def evaluate(attacker, defender, episodes=50):
    env = gym.make('BRPAtt-v0',
                   defender=DDPGWrapper.load(f'params/{defender}'),
                   mode=AttackerMode.Observation,
                   range=np.float(0.3))
    attacker_model = DDPGWrapper.load(f'params/{attacker}')

    episode_reward = 0

    for i in tqdm(range(episodes)):
        obs = env.reset()
        done = False

        while not done:
            action = attacker_model.predict(obs)
            obs, reward, done, action = env.step(action)

            episode_reward += reward

    average_reward = episode_reward / episodes

    return average_reward, -average_reward


def get_attacker_payoff(attacker):
    defenders = [f.split('/')[-1].split('.')[0] for f in listdir('params') if isfile(join('params', f)) and 'defender' in f]

    attacker_utilities = list()
    defender_utilities = list()

    for defender in defenders:
        a_u, d_u = evaluate(attacker, defender)
        attacker_utilities.append(a_u)
        defender_utilities.append(d_u)

    return np.array(attacker_utilities), np.array(defender_utilities)


def get_defender_payoff(defender):
    attackers = [f.split('/')[-1].split('.')[0] for f in listdir('params') if isfile(join('params', f)) and 'attacker' in f]

    attacker_utilities = list()
    defender_utilities = list()

    for attacker in attackers:
        a_u, d_u = evaluate(attacker, defender)
        attacker_utilities.append(a_u)
        defender_utilities.append(d_u)

    return np.array(attacker_utilities), np.array(defender_utilities)
