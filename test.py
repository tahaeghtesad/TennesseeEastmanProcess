from agents.RLAgents import Agent, SimpleWrapperAgent, MixedStrategyAgent
from envs.mtd.processors import AttackerProcessor, DefenderProcessor
from stable_baselines import DQN
import gym
import envs.mtd
from envs.mtd.heuristics import *
import numpy as np
from util.nash_helpers import find_general_sum_mixed_ne
from tqdm import tqdm

attackers = [BaseAttacker, MaxProbeAttacker, UniformAttacker, ControlThresholdAttacker]
defenders = [BaseDefender, ControlThresholdDefender, PCPDefender, UniformDefender, MaxProbeDefender]

def get_payoff(env_params, attacker: Agent, defender: Agent, repeat=5):
        env = gym.make('MTD-v0', **env_params)
        ap = AttackerProcessor(m=env_params['m'])
        dp = DefenderProcessor(m=env_params['m'])

        ar, dr = 0, 0

        for i in range(repeat):

            initial_obs = env.reset()
            done = False
            iter = 0

            a_obs = ap.process_observation(initial_obs)
            d_obs = dp.process_observation(initial_obs)

            while not done:
                obs, reward, done, info = env.step((
                    ap.process_action(attacker.predict(a_obs)),
                    dp.process_action(defender.predict(d_obs))
                ))

                a_obs, a_r, a_d, a_i = ap.process_step(obs, reward, done, info)
                d_obs, d_r, d_d, d_i = dp.process_step(obs, reward, done, info)

                ar += a_r * 0.99 ** iter
                dr += d_r * 0.99 ** iter

                iter += 1

        return ar / repeat, dr / repeat

def get_policy(num):
    attacker_payoff_table = np.load(f'runs/{num}/attacker_payoff.npy')
    defender_payoff_table = np.load(f'runs/{num}/defender_payoff.npy')

    attacker_strategy, defender_strategy = find_general_sum_mixed_ne(attacker_payoff_table, defender_payoff_table)

    apc, dpc = attacker_payoff_table.shape

    attacker_policy = MixedStrategyAgent()
    defender_policy = MixedStrategyAgent()

    for a in tqdm(range(apc)):
        if a < len(attackers):
            attacker_policy.add_policy(attackers[a]())
        else:
            attacker_policy.add_policy(SimpleWrapperAgent(DQN.load(f'runs/{num}/attacker-{a}.zip')))

    for d in tqdm(range(dpc)):
        if d < len(defenders):
            defender_policy.add_policy(defenders[d]())
        else:
            defender_policy.add_policy(SimpleWrapperAgent(DQN.load(f'runs/{num}/defender-{d}.zip')))

    attacker_policy.update_probabilities(attacker_strategy)
    defender_policy.update_probabilities(defender_strategy)

    return attacker_policy, defender_policy

if __name__ == '__main__':
    a25, d25 = get_policy(25)
    a21, d21 = get_policy(21)
    a34, d34 = get_policy(34)

    env_params = {
        'm': 10,
        'utenv': 2,
        'setting': 0,
        'ca': 0.05,
        'downtime': 7,
        'alpha': 0.05,
        'probe_detection': 0.0
    }

    ap, dp = get_payoff(env_params, a25, d21)
    print(f'4 ~> {ap:.4f}:{dp:.4f}')
    env_params['ca'] = 0.2
    ap, dp = get_payoff(env_params, a21, a25)
    print(f'3 ~> {ap:.4f}:{dp:.4f}')

    env_params['ca'] = 0.02
    env_params['utenv'] = 2
    ap, dp = get_payoff(env_params, a21, d34)
    print(f'2 ~> {ap:.4f}:{dp:.4f}')
    env_params['utenv'] = 0
    ap, dp = get_payoff(env_params, a34, d21)
    print(f'1 ~> {ap:.4f}:{dp:.4f}')