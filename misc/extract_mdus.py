import numpy as np
from trainer import Trainer
from matplotlib import pyplot as plt


def main():
    trainer = Trainer(None, None)
    defender_payoff_table = np.load('./defender_payoff.npy')
    attacker_payoff_table = np.load('./attacker_payoff.npy')

    rewards = [[], []]

    # print('it\tlp_payoff\tgambit_payoff')
    for i in range(2, defender_payoff_table.shape[0]):
        for j in range(2, 0, -1):
            pt = defender_payoff_table[:i, :i-j+1]
            attacker_strategy_lp = trainer.find_zero_sum_mixed_ne(pt)
            defender_strategy_lp = trainer.find_zero_sum_mixed_ne(-pt.transpose())

            probabilities_lp = np.outer(defender_strategy_lp, attacker_strategy_lp)
            reward_d_lp = pt * probabilities_lp
            print(f'{i:2d}-{j-1}\t{sum(sum(reward_d_lp))}')
            rewards[j-1].append(sum(sum(reward_d_lp)))

    plt.plot(rewards[0], label='Before Attacker')
    plt.plot(rewards[1], label='After Attacker')
    plt.legend()
    plt.savefig('payoff.png')

        # apt = attacker_payoff_table[:i, :i]
        # dpt = defender_payoff_table[:i, :i]
        #
        # attacker_strategy, defender_strategy = trainer.find_general_sum_mixed_ne(apt, dpt)
        # # print(f'{attacker_strategy}\n{defender_strategy}')
        # probabilities = np.outer(defender_strategy, attacker_strategy)
        # reward_d = dpt * probabilities
        # print(f'{i}\t{sum(sum(reward_d_lp)):.4f}\t{sum(sum(reward_d)):.4f}')
        # # print(f'lp\tattacker:{attacker_strategy_lp}\tdefender:{defender_strategy_lp}')
        # # print(f'gambit\tattacker:{attacker_strategy}\tdefender:{defender_strategy}')




if __name__ == '__main__':
    main()
