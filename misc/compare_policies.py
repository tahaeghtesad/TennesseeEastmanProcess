import copy
import csv
import json

import gym
import gym.envs
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines import DDPG
from tqdm import tqdm

from agents.RLAgents import MixedStrategyAgent, HistoryAgent, SimpleWrapperAgent
from trainer import RCTrainer
from util.nash_helpers import find_zero_sum_mixed_ne_gambit

prefix = './runs/101'

# no_attack_x = [[], [], []]
# no_defense_x = [[], [], []]
# defense_x = [[], [], []]
#
# desired_x = [[], [], []]

compromise_actuation_prob = 0
compromise_observation_prob = 1

with open(f'{prefix}/info.json') as info_fp:
    params = json.load(info_fp)


def main(writer):
    attacker_payoff = np.load(f'{prefix}/attacker_payoff.npy')
    defender_payoff = np.load(f'{prefix}/defender_payoff.npy')
    sa, sd = find_zero_sum_mixed_ne_gambit(attacker_payoff, defender_payoff)
    print(f'sa = {sa} / sd = {sd}')

    no_op_attacker = RCTrainer.NoOpAgent(4)
    defender_0 = HistoryAgent(DDPG.load(f'{prefix}/defender-0.zip'))
    attacker_1 = HistoryAgent(DDPG.load(f'{prefix}/attacker-1.zip'))

    defender_msne = MixedStrategyAgent()
    print('Loading defender strategies.')
    for i in tqdm(range(0, len(sd))):
        defender_msne.add_policy(HistoryAgent(DDPG.load(f'{prefix}/defender-{i}.zip')))
    defender_msne.update_probabilities(sd)

    attacker_msne = MixedStrategyAgent()
    print('Loading attacker strategies.')
    attacker_msne.add_policy(no_op_attacker)
    for i in tqdm(range(1, len(sa))):
        attacker_msne.add_policy(HistoryAgent(DDPG.load(f'{prefix}/attacker-{i}.zip')))
    attacker_msne.update_probabilities(sa)

    adversarial_env = gym.make('BRPAtt-v0',
                               defender=None,
                               **params['env_params']
                               )
    initial_obs = adversarial_env.reset()
    print(f'compromise: {adversarial_env.get_compromise()}')

    no_attack_env = copy.deepcopy(adversarial_env)
    no_attack_env.set_defender(defender_0)
    # no_attack_env.defender = defender_0

    no_defense_env = copy.deepcopy(adversarial_env)
    no_defense_env.set_defender(defender_0)
    # no_defense_env.defender = defender_0

    defense_env = copy.deepcopy(adversarial_env)
    defense_env.set_defender(defender_msne)
    # defense_env.defender = defender_msne

    no_defense_attacker = attacker_1

    no_attack_obs = initial_obs
    no_defense_obs = initial_obs
    defense_obs = initial_obs

    for i in tqdm(range(210)):

        no_attacker_action = no_op_attacker.predict(no_attack_obs)
        no_attack_obs, no_attack_reward, no_attack_done, no_attack_info = no_attack_env.step(no_attacker_action)
        no_defense_attacker_action = no_defense_attacker.predict(no_defense_obs)
        no_defense_obs, no_defense_reward, no_defense_done, no_defense_info = no_defense_env.step(no_defense_attacker_action)
        msne_attacker_action = attacker_msne.predict(defense_obs)
        defense_obs, defense_reward, defense_done, defense_info = defense_env.step(msne_attacker_action)

        # if i > -1:
        # if random.random() < .4:
        writer.writerow([
            i,
            no_attack_info['x'][0],
            no_attack_info['x'][1],
            no_attack_info['a'][0],
            no_attack_info['a'][1],
            no_attack_info['o'][0],
            no_attack_info['o'][1],
            no_attack_info['u'][0] / (1 + no_attacker_action[0]),
            no_attack_info['u'][1] / (1 + no_attacker_action[1]),
            -no_attack_reward,

            # no_attack_info['x'][2],

            no_defense_info['x'][0],
            no_defense_info['x'][1],
            no_defense_info['a'][0],
            no_defense_info['a'][1],
            no_defense_info['o'][0],
            no_defense_info['o'][1],
            no_defense_info['u'][0] / (1 + no_defense_attacker_action[0]),
            no_defense_info['u'][1] / (1 + no_defense_attacker_action[1]),
            -no_defense_reward,
            # no_defense_info['x'][2],

            defense_info['x'][0],
            defense_info['x'][1],
            defense_info['a'][0],
            defense_info['a'][1],
            defense_info['o'][0],
            defense_info['o'][1],
            defense_info['u'][0] / (1 + msne_attacker_action[0]),
            defense_info['u'][1] / (1 + msne_attacker_action[1]),
            -defense_reward
            # defense_info['x'][2],
        ])

    #     for i in range(2):
    #         no_attack_x[i].append(no_attack_info['x'][i])
    #         no_defense_x[i].append(no_defense_info['x'][i])
    #         defense_x[i].append(defense_info['x'][i])
    #     desired_x[0].append(0.99510292)
    #     desired_x[1].append(1.5122427)
    #     # desired_x[2].append(.3)
    #
    # for i in range(2):
    #     plt.title(f'X{i + 1}')
    #     plt.plot(no_attack_x[i], label='no attack', alpha=.8)
    #     plt.plot(no_defense_x[i], label='no defense', alpha=.8)
    #     plt.plot(defense_x[i], label='with defense', alpha=.8)
    #     plt.plot(desired_x[i], label='desired', alpha=.6)
    #     plt.legend()
    #     plt.show()


if __name__ == '__main__':
    with open(f'{prefix}/compare.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['step',
                         'no_attack_state_0',
                         'no_attack_state_1',
                         'no_attack_attacker_action_a_0',
                         'no_attack_attacker_action_a_1',
                         'no_attack_attacker_action_o_0',
                         'no_attack_attacker_action_o_1',
                         'no_attack_defender_action_0',
                         'no_attack_defender_action_1',
                         'no_attack_reward',
                         # 'no_attack_2',
                         'no_defense_state_0',
                         'no_defense_state_1',
                         'no_defense_attacker_action_a_0',
                         'no_defense_attacker_action_a_1',
                         'no_defense_attacker_action_o_0',
                         'no_defense_attacker_action_o_1',
                         'no_defense_defender_action_0',
                         'no_defense_defender_action_1',
                         'no_defense_reward',
                         # 'no_defense_2',
                         'defense_state_0',
                         'defense_state_1',
                         'defense_attacker_action_a_0',
                         'defense_attacker_action_a_1',
                         'defense_attacker_action_o_0',
                         'defense_attacker_action_o_1',
                         'defense_defender_action_0',
                         'defense_defender_action_1',
                         'defense_reward'
                         # 'defense_2'
                         ])
        main(writer)
