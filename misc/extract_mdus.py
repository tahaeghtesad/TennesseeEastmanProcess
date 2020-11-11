import csv
import json

import numpy as np
from matplotlib import colors as mcolors

from util.nash_helpers import find_general_sum_mixed_ne


def extract_mdu(path):
    defender_payoff_table = np.load(path + 'defender_payoff.npy')
    attacker_payoff_table = np.load(path + 'attacker_payoff.npy')

    rewards = [[], []]
    for i in range(2, defender_payoff_table.shape[0]):
        for j in [1, 0]:
            ap = attacker_payoff_table[:i-j, :i]
            dp = defender_payoff_table[:i-j, :i]
            attacker_strategy,\
            defender_strategy = find_general_sum_mixed_ne(ap, dp)
            probabilities = np.outer(attacker_strategy, defender_strategy)
            reward_a = ap * probabilities
            reward_d = dp * probabilities
            # print(sum(sum(reward_a)), sum(sum(reward_d)))
            rewards[0].append(sum(sum(reward_d)))
            rewards[1].append(sum(sum(reward_a)))

    with open(path + 'info.json') as fp:
        info = json.load(fp)

    return info, rewards


def main():
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
    # print(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS))

    with open('data.csv', 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(['$M$', 'utenv', 'setting', '$C_A$', '$\\Delta$', '$\\alpha$', '$\\nu$'] + ['attacker', 'defender'])
        for i in range(1, 48):
            if i in [43, 44]:
                continue
            info, rewards = extract_mdu(f'runs/{i}/')
            writer.writerow([str(v) for v in info['env_params'].values()] + [f'{rewards[1][-1]:.2f}', f'{rewards[0][-1]:.2f}'])

    # with open('data.csv', 'w') as fd:
    #     writer = csv.writer(fd)
    #     writer.writerow(['attacker-0', 'defender-0', 'attacker-1', 'defender-1', 'attacker-2', 'defender-2'])
    #     payoff_eqs = [[], [], [], [], [], []]
    #     for i, index in enumerate([21, 22, 27]):
    #         info, (payoff_eqs[2 * i], payoff_eqs[2 * i + 1]) = extract_mdu(f'runs/{index}/')
    #     for i in range(12):
    #         row = []
    #         for j in range(6):
    #             row.append(payoff_eqs[j][i])
    #         writer.writerow(row)
    #     #     plt.plot(rewards[1], '--', color=f'{colors[0]}', label='-'.join([str(v) for v in info['env_params'].values()]))
    #     #     plt.plot(rewards[0], '-.', color=f'{colors[0]}')
    #             # plt.plot(rewards[1])

    # for i in range (1, 21):
    #     info, rewards = extract_mdu(f'runs/{i}/')
    #     print(f'{i}\t{info["env_params"].values()}')


    # print(info['env_params'].keys())



    # plt.legend()
    # plt.show()


def extract_msne_vs_pure_strategies_payoff():
    defender_payoff_table = np.load('runs/21/defender_payoff.npy')
    attacker_payoff_table = np.load('runs/21/attacker_payoff.npy')

    attacker_strategy, defender_strategy = find_general_sum_mixed_ne(attacker_payoff_table, defender_payoff_table)

    for i in range(4):
        att_strat = np.zeros((attacker_payoff_table.shape[0]))
        att_strat[i] = 1.

        probabilities = np.outer(att_strat, defender_strategy)
        reward_a = attacker_payoff_table * probabilities
        reward_d = defender_payoff_table * probabilities

        print(i, sum(sum(reward_a)), sum(sum(reward_d)))

    for j in range(5):
        def_strat = np.zeros((attacker_payoff_table.shape[1]))
        def_strat[j] = 1.

        probabilities = np.outer(attacker_strategy, def_strat)
        reward_a = attacker_payoff_table * probabilities
        reward_d = defender_payoff_table * probabilities

        print(j, sum(sum(reward_a)), sum(sum(reward_d)))

if __name__ == '__main__':
    main()
