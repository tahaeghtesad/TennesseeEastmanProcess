import gym
from stable_baselines import DDPG

import envs
from stable_baselines.ddpg import LnMlpPolicy

from agents.RLAgents import Agent, SimpleWrapperAgent, MixedStrategyAgent, HistoryAgent, LimitedHistoryAgent, \
    SinglePolicyMixedStrategyAgent
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

    class NoOpAgent(Agent):

        def __init__(self, action_dim) -> None:
            super().__init__()
            self.action_dim = action_dim

        def predict(self, observation, state=None, mask=None, deterministic=True):
            return np.zeros(self.action_dim)

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

    def train_attacker(self, defender, iteration, index) -> Agent:
        # env params must have 'compromise_actuation_prob', 'compromise_observation_prob', and 'power'
        # rl params must have 'gamma', 'random_exploration'
        self.logger.info(f'Starting attacker training for {self.training_params["training_steps"]} steps.')
        env = gym.make('LimitedHistoritized-v0' if self.training_params['training_params_attacker_limited_history'] else 'Historitized-v0',
                       env=f'{self.env_id}Att-v0',
                       **self.env_params,
                       defender=defender
                       )\
            if self.training_params['attacker_history'] else\
            gym.make(f'{self.env_id}Att-v0',
                     **self.env_params,
                     defender=defender
                     )
        # env = gym.make(f'{self.env_id}Att-v0',
        #                **self.env_params,
        #                defender=defender
        #                )

        attacker_model = DDPG(
            policy=self.get_policy_class(self.policy_params),
            env=env,
            **self.rl_params,
            verbose=1,
            tensorboard_log=f'{self.prefix}/tb_logs' if self.training_params['tb_logging'] else None,
            full_tensorboard_log=self.training_params['tb_logging']
        )

        attacker_model.learn(
            total_timesteps=self.training_params['training_steps'],
            callback=self.callback,
            tb_log_name=f'attacker_{iteration}_{index}'
        )

        attacker_model.save(f'{self.prefix}/params/attacker-{iteration}-{index}')
        if self.training_params['attacker_history']:
            if self.training_params['attacker_limited_history']:
                return LimitedHistoryAgent(attacker_model)
            else:
                return HistoryAgent(attacker_model)
        else:
            return SimpleWrapperAgent(attacker_model)

    def train_defender(self, attacker, iteration, index) -> Agent:
        self.logger.info(f'Starting defender training for {self.training_params["training_steps"]} steps.')
        env = gym.make('LimitedHistoritized-v0' if self.training_params['defender_limited_history'] else 'Historitized-v0',
                       env=f'{self.env_id}Def-v0',
                       **self.env_params,
                       attacker=attacker
                       ) \
            if self.training_params['defender_history'] else \
            gym.make(f'{self.env_id}Def-v0',
                     **self.env_params,
                     attacker=attacker
                     )
        defender_model = DDPG(
            policy=self.get_policy_class(self.policy_params),
            env=env,
            **self.rl_params,
            verbose=1,
            tensorboard_log=f'{self.prefix}/tb_logs' if self.training_params['tb_logging'] else None,
            full_tensorboard_log=self.training_params['tb_logging']
        )

        defender_model.learn(
            total_timesteps=self.training_params['training_steps'],
            callback=self.callback,
            tb_log_name=f'defender_{iteration}_{index}'
        )

        defender_model.save(f'{self.prefix}/params/defender-{iteration}-{index}')
        if self.training_params['defender_history']:
            if self.training_params['defender_limited_history']:
                return LimitedHistoryAgent(defender_model)
            else:
                return HistoryAgent(defender_model)
        else:
            return SimpleWrapperAgent(defender_model)

    def get_payoff(self, attacker: Agent, defender: Agent, repeat=20):
        env = gym.make(f'{self.env_id}Def-v0',
                       **self.env_params,
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

                r += reward * self.rl_params['gamma'] ** iter
                iter += 1

        return -r / repeat, r / repeat

    def initialize_strategies(self):
        attacker = self.NoOpAgent(4)  # TODO this should not be a constant 4
        self.logger.info('Initializing a defender against NoOp attacker...')
        defender = self.train_defender_parallel(attacker, 0)

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
