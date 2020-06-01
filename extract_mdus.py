import numpy as np
from trainer import Trainer


def main():
    trainer = Trainer(None, None)
    defender_payoff_table = np.load('defender_payoff.npy')

    for i in range(1, defender_payoff_table.shape[0]):
        pt = defender_payoff_table[:i, :i]
        attacker_strategy = trainer.find_zero_sum_mixed_ne(pt)
        defender_strategy = trainer.find_zero_sum_mixed_ne(-pt.transpose())

        probabilities = np.outer(defender_strategy, attacker_strategy)
        reward_d = pt * probabilities
        print(sum(sum(reward_d)))


if __name__ == '__main__':
    main()
