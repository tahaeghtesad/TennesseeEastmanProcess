import copy
import csv
import json
import multiprocessing
from time import time

import gym
import gym.envs
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines import DDPG
from tqdm import tqdm

from agents.RLAgents import MixedStrategyAgent, HistoryAgent, SimpleWrapperAgent, ConstantAgent, ZeroAgent
from envs.control.adversarial_control import AdversarialControlEnv
from trainer import RCTrainer
from util.nash_helpers import find_zero_sum_mixed_ne_gambit

# no_attack_x = [[], [], []]
# no_defense_x = [[], [], []]
# defense_x = [[], [], []]
#
# desired_x = [[], [], []]


def extract_info(prefix):
    start = time()
    with open(f'{prefix}/info.json') as info_fp:
        params = json.load(info_fp)

    with open(f'./compares/{prefix.split("/")[-1]}.csv', 'w') as file:
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

        attacker_payoff = np.load(f'{prefix}/attacker_payoff.npy')
        defender_payoff = np.load(f'{prefix}/defender_payoff.npy')
        sa, sd = find_zero_sum_mixed_ne_gambit(attacker_payoff, defender_payoff)
        # print(f'sa = {sa} / sd = {sd}')

        no_op_attacker = ZeroAgent(4)
        # compromise = np.zeros(4)
        compromise = np.concatenate(
            (np.random.rand(2) < params['env_params']['compromise_observation_prob'],
             np.random.rand(2) < params['env_params']['compromise_actuation_prob'])
            , axis=0).astype(np.float)

        np.save(f'./compares/{prefix.split("/")[-1]}', compromise)

        defender_0 = SimpleWrapperAgent(DDPG.load(f'{prefix}/params/defender-0.zip', verbose=0))
        attacker_msne = SimpleWrapperAgent(DDPG.load(f'{prefix}/params/attacker-{np.argmax(sa)}.zip', verbose=0)) if np.argmax(sa) != 0 else no_op_attacker

        no_attack_env = AdversarialControlEnv(f'{params["env_id"]}-v0',
                                              no_op_attacker,
                                              defender_0,
                                              **params['env_params'],
                                              test_env=True)
        no_defense_env = AdversarialControlEnv(f'{params["env_id"]}-v0',
                                               attacker_msne,
                                               defender_0,
                                               **params['env_params'],
                                               test_env=True)
        defense_env = AdversarialControlEnv(f'{params["env_id"]}-v0',
                                            attacker_msne,
                                            SimpleWrapperAgent(DDPG.load(f'{prefix}/params/defender-{np.argmax(sd)}.zip', verbose=0)),
                                            **params['env_params'],
                                            test_env=True)

        no_attack_env.reset(compromise)
        no_defense_env.reset(compromise)
        defense_env.reset(compromise)

        for i in range(100):
            (_, no_attack_obs), (no_attack_reward, _), no_attack_done, no_attack_info = no_attack_env.step()
            (_, no_defense_obs), (no_defense_reward, _), no_defense_done, no_defense_info = no_defense_env.step()
            (_, defense_obs), (defense_reward, _), defense_done, defense_info = defense_env.step()

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
                no_attack_info['d'][0],
                no_attack_info['d'][1],
                -no_attack_reward,

                # no_attack_info['x'][2],

                no_defense_info['x'][0],
                no_defense_info['x'][1],
                no_defense_info['a'][0],
                no_defense_info['a'][1],
                no_defense_info['o'][0],
                no_defense_info['o'][1],
                no_defense_info['d'][0],
                no_defense_info['d'][1],
                -no_defense_reward,
                # no_defense_info['x'][2],

                defense_info['x'][0],
                defense_info['x'][1],
                defense_info['a'][0],
                defense_info['a'][1],
                defense_info['o'][0],
                defense_info['o'][1],
                defense_info['d'][0],
                defense_info['d'][1],
                -defense_reward
                # defense_info['x'][2],
            ])
    plot(prefix.split('/')[-1])
    print(f'Done with {prefix}, took {time() - start:.3f}(s)')
    return 0


def plot(index):

    compromise = np.load(f'compares/{index}.npy')
    print(compromise)

    fig, axs = plt.subplots(4, 2, figsize=(16, 12))

    names = ['no_attack', 'defense', 'no_defense']
    specs = ['state', 'attacker_action_a', 'attacker_action_o', 'defender_action']

    with open(f'compares/{index}.csv') as fd:
        reader = csv.DictReader(fd)
        data = list(reader)
        total_steps = 100
        for i, spec in enumerate(specs):
            for j in range(2):

                axs[i, j].set_title(f'{spec}_{j}')

                for name in names:
                    axs[i, j].plot([float(row[f'{name}_{spec}_{j}']) for row in data], label=f'{name}')

                    if i == 0:
                        # [0.99510292, 1.5122427]
                        axs[i, 0].plot([0.99510292 for i in range(total_steps)], label='desired')
                        axs[i, 1].plot([1.5122427  for i in range(total_steps)], label='desired')

        fig.suptitle(f'Compromise: {compromise}', fontsize=16)
        plt.legend()
        plt.savefig(f'compares/{index}.jpg')
        # plt.show()

    # 'no_attack_state_0',
    # 'no_attack_state_1',
    # 'no_attack_attacker_action_a_0',
    # 'no_attack_attacker_action_a_1',
    # 'no_attack_attacker_action_o_0',
    # 'no_attack_attacker_action_o_1',
    # 'no_attack_defender_action_0',
    # 'no_attack_defender_action_1',
    # 'no_attack_reward',


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
    # extract_info('../runs/back4/3000')
    # plot(3000)
    with multiprocessing.Pool(8) as pool:
        pool.map(extract_info, [f'../runs/back4/{i}' for i in range(3000, 3938)])
