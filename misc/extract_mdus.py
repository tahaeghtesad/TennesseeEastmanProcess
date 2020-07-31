import numpy as np
import json
from tqdm import tqdm
from util.nash_helpers import find_general_sum_mixed_ne
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import csv


def extract_mdu(path):
    defender_payoff_table = np.load(path + 'defender_payoff.npy')
    attacker_payoff_table = np.load(path + 'attacker_payoff.npy')

    rewards = [[], []]
    for i in range(5, defender_payoff_table.shape[0]):
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
    print(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS))

    with open('data.csv', 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(['attacker', 'defender'])
        info, rewards = extract_mdu(f'runs/{4}/')
        for a, d in zip(rewards[0], rewards[1]):
            writer.writerow([a, d])
        plt.plot(rewards[1], '--', color=f'{colors[0]}', label='-'.join([str(v) for v in info['env_params'].values()]))
        plt.plot(rewards[0], '-.', color=f'{colors[0]}')
            # plt.plot(rewards[1])


    print(info['env_params'].keys())

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
