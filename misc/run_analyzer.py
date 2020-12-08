import multiprocessing as mp
import threading
from time import time

import gym
from stable_baselines import DDPG

import envs
import json
import csv
from tqdm import tqdm

from agents.RLAgents import LimitedHistoryAgent, HistoryAgent, SimpleWrapperAgent, ZeroAgent
from envs.control.adversarial_control import AdversarialControlEnv


class Analyzer:

    def __init__(self, path) -> None:
        super().__init__()
        self.base_path = path
        with open(f'{self.base_path}/info.json') as param_file:
            self.params = json.load(param_file)

    def payoff(self, attacker, defender, repeat=50):
        env = AdversarialControlEnv('BRP-v0', attacker, defender, **self.params['env_params'])

        ra = 0
        rd = 0

        for _ in range(repeat):
            env.reset()
            done = False
            iter = 0
            for step in range(50):
                (att_obs, def_obs), (reward_a, reward_d), done, info = env.step()

                ra += reward_a * self.params['rl_params']['gamma'] ** iter
                rd += reward_d * self.params['rl_params']['gamma'] ** iter
                iter += 1

        return ra / repeat, rd / repeat

    def load_agent(self):
        dqn_agent = DDPG.load(f'{self.base_path}/params/defender-0.zip', verbose=0)
        return SimpleWrapperAgent(dqn_agent)

    def params_to_list(self):
        ret = []
        for k1 in self.params:
            v1 = self.params[k1]
            if type(v1) is dict:
                for k2 in v1:
                    v2 = v1[k2]
                    ret.append(v2)
            else:
                ret.append(v1)
        return ret

    def param_names_to_list(self):
        ret = []
        for k1 in self.params:
            v1 = self.params[k1]
            if type(v1) is dict:
                for k2 in v1:
                    ret.append(f'{k1}_{k2}')
            else:
                ret.append(f'{k1}')
        return ret


if __name__ == '__main__':
    with open('data.csv', 'w') as csv_file:

        writer = csv.writer(csv_file)

        info_row = Analyzer(f'../runs/back2/{2000}').param_names_to_list()
        info_row += ['defender_payoff']
        info_row = ['id'] + info_row
        writer.writerow(info_row)

        def extract(i):
            start = time()
            analyzer = Analyzer(f'../runs/back2/{i}')
            defender = analyzer.load_agent()
            attacker = ZeroAgent(4)
            _, du = analyzer.payoff(attacker, defender)
            conf_row = analyzer.params_to_list()
            conf_row += [du]
            conf_row = [i] + conf_row
            # conf_row = ['kir']
            print(f'Done with agent {i}, took {time() - start:.3f}s')
            return conf_row

        # for i in tqdm(range(1500, 1659)):
        #     extract(i)
        with mp.Pool(10) as p:
            for conf_row in p.map(extract, [i for i in range(2000, 2320)]):
                writer.writerow(conf_row)
        # pool.close()