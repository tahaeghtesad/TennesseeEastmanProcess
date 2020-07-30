import numpy as np
import json
from tqdm import tqdm
from util.nash_helpers import find_general_sum_mixed_ne
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors


def extract_mdu(path):
    defender_payoff_table = np.load(path + 'defender_payoff.npy')
    attacker_payoff_table = np.load(path + 'attacker_payoff.npy')

    rewards = [[], []]
    for i in range(5, defender_payoff_table.shape[0]):
        ap = attacker_payoff_table[:i, :i-1]
        dp = defender_payoff_table[:i, :i-1]
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
    for i in tqdm(range(1, 21)):
        info, rewards = extract_mdu(f'runs/{i}/')
        plt.plot(rewards[1], '--', color=f'{colors[i-1]}', label='-'.join([str(v) for v in info['env_params'].values()]))
        plt.plot(rewards[0], '-.', color=f'{colors[i-1]}')
        # plt.plot(rewards[1])

    print(info['env_params'].keys())

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
