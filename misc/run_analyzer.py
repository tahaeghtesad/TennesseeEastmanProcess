import csv
import json

import gym
from stable_baselines import DDPG
from tqdm import tqdm

from agents.RLAgents import LimitedHistoryAgent, HistoryAgent, SimpleWrapperAgent, NoOpAgent


class Analyzer:

    def __init__(self, path) -> None:
        super().__init__()
        self.base_path = path
        with open(f'{self.base_path}/info.json') as param_file:
            self.params = json.load(param_file)

    def payoff(self, attacker, defender, repeat=20):
        env = gym.make(f'{self.params["env_id"]}Def-v0',
                       **self.params['env_params'],
                       attacker=attacker
                       )

        r = 0

        for _ in range(repeat):
            def_obs = env.reset()
            done = False
            iter = 0
            while not done:
                action = defender.predict(def_obs)
                def_obs, reward, done, info = env.step(action)

                r += reward * self.params['rl_params']['gamma'] ** iter
                iter += 1

        return -r / repeat, r / repeat

    def load_agent(self):
        dqn_agent = DDPG.load(f'{self.base_path}/defender-0.zip')
        if self.params['training_params']['defender_history']:
            if self.params['training_params']['defender_limited_history']:
                return LimitedHistoryAgent(dqn_agent)
            else:
                return HistoryAgent(dqn_agent)
        else:
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

        for i in tqdm(range(101, 146)):
            analyzer = Analyzer(f'../runs/{i}')
            defender = analyzer.load_agent()
            attacker = NoOpAgent(4)
            if i == 101:
                info_row = analyzer.param_names_to_list()
                info_row += ['defender_payoff']
                writer.writerow(info_row)
            _, du = analyzer.payoff(attacker, defender)
            conf_row = analyzer.params_to_list()
            conf_row += [du]
            writer.writerow(conf_row)