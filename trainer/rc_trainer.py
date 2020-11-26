import gym
from stable_baselines import DDPG
from stable_baselines.common.noise import NormalActionNoise

import envs
from stable_baselines.ddpg import LnMlpPolicy

from agents.RLAgents import Agent, SimpleWrapperAgent, MixedStrategyAgent, HistoryAgent, LimitedHistoryAgent, \
    SinglePolicyMixedStrategyAgent, ZeroAgent
from envs.control.adversarial_control import AdversarialControlEnv
from envs.control.envs import BioReactorDefender, BioReactorAttacker
from trainer.trainer import Trainer
import tensorflow as tf
import numpy as np
import logging


class RCTrainer(Trainer):

    def __init__(self, prefix, env_id, env_params=None, rl_params=None, policy_params=None,
                 training_params=None) -> None:
        super().__init__(prefix, env_params, rl_params, policy_params, training_params)
        self.logger = logging.getLogger(__name__)
        self.env_id = env_id

    @staticmethod
    def callback(locals_, globals_):
        self_ = locals_['self']

        variables = ['u', 'x', 'dx', 'a', 'o']

        if 'info' in locals_ and 'writer' in locals_ and locals_['writer'] is not None:
            for var in variables:
                if var in locals_['info']:
                    for i in range(len(locals_['info'][var])):
                        summary = tf.Summary(
                            value=[tf.Summary.Value(tag=f'env/{var}{i}', simple_value=locals_['info'][var][i])])
                        locals_['writer'].add_summary(summary, self_.num_timesteps)
        return True

    def get_policy_class(self, policy_params):
        # policy params must have 'act_fun' and 'layers'
        class CustomPolicy(LnMlpPolicy):
            def __init__(self, *args, **_kwargs):
                super().__init__(*args, **_kwargs, **policy_params)

        return CustomPolicy

    def train_attacker(self, defender, iteration):
        # env params must have 'compromise_actuation_prob', 'compromise_observation_prob', 'history_length', and 'power'
        # rl params must have 'gamma', 'random_exploration'
        self.logger.info(f'Starting attacker training for {self.training_params["training_steps"]} steps.')
        if self.env_id == 'BRP':
            env = BioReactorAttacker(defender, **self.env_params)
        else:
            raise Exception('Invalid Environment')

        attacker_model = DDPG(
            policy=self.get_policy_class(self.policy_params),
            env=env,
            nb_train_steps=30,
            nb_rollout_steps=100,
            batch_size=128,
            buffer_size=5_000,
            action_noise=NormalActionNoise(0, self.training_params['action_noise_sigma']),
            **self.rl_params,
            verbose=2,
            tensorboard_log=f'{self.prefix}/tb_logs' if self.training_params['tb_logging'] else None,
            full_tensorboard_log=self.training_params['tb_logging']
        )

        attacker_model.learn(
            total_timesteps=self.training_params['training_steps'],
            callback=self.callback,
            tb_log_name=f'attacker_{iteration}'
        )

        attacker_model.save(f'{self.prefix}/params/attacker-{iteration}')

        return SimpleWrapperAgent(attacker_model)

    def train_defender(self, attacker, iteration):
        self.logger.info(f'Starting defender training for {self.training_params["training_steps"]} steps.')
        if self.env_id == 'BRP':
            env = BioReactorDefender(attacker, **self.env_params)
        else:
            raise Exception('Invalid Environment')

        defender_model = DDPG(
            policy=self.get_policy_class(self.policy_params),
            env=env,
            nb_train_steps=30,
            nb_rollout_steps=100,
            batch_size=128,
            buffer_size=5_000,
            action_noise=NormalActionNoise(0, self.training_params['action_noise_sigma']),
            **self.rl_params,
            verbose=2,
            tensorboard_log=f'{self.prefix}/tb_logs' if self.training_params['tb_logging'] else None,
            full_tensorboard_log=self.training_params['tb_logging']
        )

        defender_model.learn(
            total_timesteps=self.training_params['training_steps'],
            callback=self.callback,
            tb_log_name=f'defender_{iteration}'
        )

        defender_model.save(f'{self.prefix}/params/defender-{iteration}')

        return SimpleWrapperAgent(defender_model)

    def get_payoff(self, attacker: Agent, defender: Agent, repeat=20):
        if self.env_id == 'BRP':
            env = AdversarialControlEnv('BRP-v0', attacker, defender, **self.env_params)
        else:
            raise Exception('Invalid Environment')

        ra = 0
        rd = 0

        for _ in range(repeat):
            env.reset()
            done = False
            iter = 0
            while not done:
                (att_obs, def_obs), (reward_a, reward_d), done, info = env.step()

                ra += reward_a * self.rl_params['gamma'] ** iter
                rd += reward_d * self.rl_params['gamma'] ** iter
                iter += 1

        return ra / repeat, rd / repeat

    def initialize_strategies(self):
        attacker = ZeroAgent(4)  # TODO this should not be a constant 4
        self.logger.info('Initializing a defender against NoOp attacker...')
        defender = self.train_defender(attacker, 0)

        au, du = self.get_payoff(attacker, defender)

        self.defender_payoff_table = np.zeros((1, 1))
        self.attacker_payoff_table = np.zeros((1, 1))

        attacker_ms = SinglePolicyMixedStrategyAgent()
        defender_ms = SinglePolicyMixedStrategyAgent()

        self.defender_payoff_table[0, 0] = du
        self.attacker_payoff_table[0, 0] = au

        attacker_ms.add_policy(attacker)
        defender_ms.add_policy(defender)

        attacker_ms.update_probabilities(np.ones(1))
        defender_ms.update_probabilities(np.ones(1))

        return attacker_ms, defender_ms
