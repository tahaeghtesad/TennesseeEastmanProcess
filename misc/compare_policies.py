from agents.ddpg_agent import DDPGWrapper, DDPGWrapperHistory, MixedStrategyDDPG
import gym
import gym.envs
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from trainer import Trainer
import random

prefix = '..'

no_attack_x = [[], [], []]
no_defense_x = [[], [], []]
defense_x = [[], [], []]

desired_x = [[], [], []]

compromise_actuation_prob = .5
compromise_observation_prob = .5


def main(writer):

    trainer = Trainer(None, None)
    defender_payoff = np.load(f'{prefix}/defender_payoff.npy')
    sa = trainer.find_zero_sum_mixed_ne(-defender_payoff.transpose())
    print(f'sa = {sa}')
    sd = trainer.find_zero_sum_mixed_ne(defender_payoff)
    print(f'sd = {sd}')

    no_attack_env = gym.make('Historitized-v0', env='BRPAtt-v0',
                             defender=DDPGWrapperHistory.load(f'{prefix}/params/defender-0.zip'),
                             compromise_actuation_prob=compromise_actuation_prob,
                             compromise_observation_prob=compromise_observation_prob,
                             power=.3
                             )


    no_defense_env = gym.make('Historitized-v0', env='BRPAtt-v0',
                              defender=DDPGWrapperHistory.load(f'{prefix}/params/defender-0.zip'),
                              compromise_actuation_prob=compromise_actuation_prob,
                              compromise_observation_prob=compromise_observation_prob,
                              power=.3
                              )

    defense_env = gym.make('Historitized-v0', env='BRPAtt-v0',
                           defender=MixedStrategyDDPG(f'{prefix}/params/defender', len(sd), sd),
                           compromise_actuation_prob=compromise_actuation_prob,
                           compromise_observation_prob=compromise_observation_prob,
                           power=.3
                           )

    attacker = MixedStrategyDDPG(f'{prefix}/params/attacker', len(sa), sa, False)
    # no_defense_attacker = DDPGWrapper.load(f'params/attacker-{np.argmax(defender_payoff[0, :])}.zip')
    no_defense_attacker = DDPGWrapper.load(f'{prefix}/params/attacker-0')

    no_attack_obs = no_attack_env.reset()
    no_defense_obs = no_defense_env.reset()
    defense_obs = defense_env.reset()

    starting_point = no_attack_env.observation_space.sample()

    no_attack_env.x = starting_point
    no_defense_env.x = starting_point
    defense_env.x = starting_point

    for i in tqdm(range(1000)):

        no_attack_obs, _, no_attack_done, no_attack_info = no_attack_env.step(np.zeros(no_attack_env.action_space.low.shape))
        no_defense_obs, _, no_defense_done, no_defense_info = no_defense_env.step(no_defense_attacker.predict(no_defense_obs))
        defense_obs, _, defense_done, defense_info = defense_env.step(attacker.predict(defense_obs))

        if no_attack_done:
            no_attack_obs = no_attack_env.reset()
        if no_defense_done:
            no_defense_obs = no_defense_env.reset()
        if defense_done:
            defense_obs = defense_env.reset()

        if i > -1:
            if random.random() < .4:
                writer.writerow([
                    i,
                    no_attack_info['x'][0],
                    no_attack_info['x'][1],
                    # no_attack_info['x'][2],
                    no_defense_info['x'][0],
                    no_defense_info['x'][1],
                    # no_defense_info['x'][2],
                    defense_info['x'][0],
                    defense_info['x'][1],
                    # defense_info['x'][2],
                ])

                for i in range(2):
                    no_attack_x[i].append(no_attack_info['x'][i])
                    no_defense_x[i].append(no_defense_info['x'][i])
                    defense_x[i].append(defense_info['x'][i])
                desired_x[0].append(0.99510292)
                desired_x[1].append(1.5122427)
                # desired_x[2].append(.3)

    for i in range(2):
        plt.title(f'X{i+1}')
        plt.plot(no_attack_x[i], label='no attack', alpha=.8)
        plt.plot(no_defense_x[i], label='no defense', alpha=.8)
        plt.plot(defense_x[i], label='with defense', alpha=.8)
        plt.plot(desired_x[i], label='desired', alpha=.6)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    with open(f'{prefix}/compare.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['step',
                         'no_attack_0',
                         'no_attack_1',
                         # 'no_attack_2',
                         'no_defense_0',
                         'no_defense_1',
                         # 'no_defense_2',
                         'defense_0',
                         'defense_1',
                         # 'defense_2'
                         ])
        main(writer)