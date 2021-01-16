import copy
import logging

import numpy as np
import tensorflow as tf
import wandb
from stable_baselines import DDPG
from stable_baselines.common.noise import NormalActionNoise
from stable_baselines.ddpg import LnMlpPolicy

from agents.RLAgents import Agent, SimpleWrapperAgent, SinglePolicyMixedStrategyAgent, ZeroAgent
from envs.control.adversarial_control import AdversarialControlEnv
from envs.control.envs import BioReactorDefender, BioReactorAttacker, ThreeTankDefender, ThreeTankAttacker
from trainer.trainer import Trainer


class RCTrainer(Trainer):

    def __init__(self, prefix, env_id, env_params=None, rl_params=None, policy_params=None,
                 training_params=None) -> None:
        super().__init__(prefix, env_params, rl_params, policy_params, training_params)
        self.logger = logging.getLogger(__name__)
        self.env_id = env_id

    @staticmethod
    def callback(locals_, globals_):
        self_ = locals_['self']

        variables = ['u', 'x', 'dx', 'a', 'o', 'd']

        if 'info' in locals_:
            for var in variables:
                if var in locals_['info']:
                    for i in range(len(locals_['info'][var])):
                        if 'writer' in locals_ and locals_['writer'] is not None:
                            summary = tf.Summary(
                                value=[tf.Summary.Value(tag=f'env/{var}{i}', simple_value=locals_['info'][var][i])])
                            locals_['writer'].add_summary(summary, self_.num_timesteps)

        return True

    @staticmethod
    def wandb_callback(locals_, globals_):
        self_ = locals_['self']

        variables = ['u', 'x', 'dx', 'a', 'o', 'd', 'c']

        # if 'info' in locals_:
        #     for var in variables:
        #         if var in locals_['info']:
        #             for i in range(len(locals_['info'][var])):
        #                 wandb.log({f'env/{var}{i}': locals_['info'][var][i]}, commit=False)

        if 'reward' in locals_:
            wandb.log({f'rewards/step': locals_['reward']}, commit=False)

        if 'episode_reward' in locals_:
            wandb.log({f'rewards/episode': locals_['episode_reward']}, commit=False)

        wandb.log({})
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
        elif self.env_id == 'TT':
            env = ThreeTankAttacker(defender, **self.env_params)
        else:
            raise Exception('Invalid Environment')

        attacker_model = DDPG(
            policy=self.get_policy_class(self.policy_params),
            env=env,
            action_noise=NormalActionNoise(0, self.training_params['action_noise_sigma']),
            **self.rl_params,
            verbose=2,
            tensorboard_log=f'{self.prefix}/tb_logs' if self.training_params['tb_logging'] else None,
            full_tensorboard_log=self.training_params['tb_logging']
        )

        attacker_model.learn(
            total_timesteps=self.training_params['training_steps'],
            # callback=self.wandb_callback,
            tb_log_name=f'attacker_{iteration}'
        )

        attacker_model.save(f'{self.prefix}/params/attacker-{iteration}')

        return SimpleWrapperAgent(attacker_model)

    def train_defender(self, attacker, iteration):
        self.logger.info(f'Starting defender training for {self.training_params["training_steps"]} steps.')
        if self.env_id == 'BRP':
            env = BioReactorDefender(attacker, **self.env_params)
        elif self.env_id == 'TT':
            env = ThreeTankDefender(attacker, **self.env_params)
        else:
            raise Exception('Invalid Environment')

        defender_model = DDPG(
            policy=self.get_policy_class(self.policy_params),
            env=env,
            action_noise=NormalActionNoise(0, self.training_params['action_noise_sigma']),
            **self.rl_params,
            verbose=2,
            tensorboard_log=f'{self.prefix}/tb_logs' if self.training_params['tb_logging'] else None,
            full_tensorboard_log=self.training_params['tb_logging']
        )

        defender_model.learn(
            total_timesteps=self.training_params['training_steps'],
            # callback=self.wandb_callback,
            tb_log_name=f'defender_{iteration}'
        )

        defender_model.save(f'{self.prefix}/params/defender-{iteration}')

        return SimpleWrapperAgent(defender_model)

    def get_payoff(self, attacker: Agent, defender: Agent, repeat=64, compromise=None, log=True):

        params = copy.deepcopy(self.env_params)
        params.update({
            't_epoch': 200,
            # 'noise_sigma': 0.00,
            'test_env': True,
        })

        env = AdversarialControlEnv(f'{self.env_id}-v0', attacker, defender, **params)

        ra = 0
        rd = 0
        total_steps = 0

        columns = ['reward', 'x_0', 'x_1', 'x_2', 'a_0', 'a_1', 'd_0', 'd_1', 'u_0', 'u_1', 'dx_0', 'dx_1', 'dx_2', 'o_0', 'o_1', 'c_0', 'c_1', 'c_2', 'c_3']
        report_table = wandb.Table(columns=['step'] + columns)

        for e in range(repeat):
            compromise_vector = compromise(e) if compromise is not None else None
            env.reset(compromise_vector)
            done = False
            iter = 0
            while not done:
                (att_obs, def_obs), (reward_a, reward_d), done, info = env.step()

                ra += reward_a * .9 ** iter ## Evaluation Gamma should Always be the same
                rd += reward_d * .9 ** iter
                iter += 1

                report_table.add_data(
                    total_steps,
                    reward_d,
                    info['x'][0],
                    info['x'][1],
                    info['x'][2] if info['x'].shape[0] == 3 else 0,
                    info['a'][0],
                    info['a'][1],
                    info['d'][0],
                    info['d'][1],
                    info['u'][0],
                    info['u'][1],
                    info['dx'][0],
                    info['dx'][1],
                    info['dx'][2] if info['dx'].shape[0] == 3 else 0,
                    info['o'][0],
                    info['o'][1],
                    info['c'][0],
                    info['c'][1],
                    info['c'][2],
                    info['c'][3],
                )

                # if log:
                #     wandb.log({
                #         'test/reward': reward_d,
                #         'test/a0': info['a'][0],
                #         'test/a1': info['a'][1],
                #         'test/d0': info['d'][0],
                #         'test/d1': info['d'][1],
                #         'test/u0': info['u'][0],
                #         'test/u1': info['u'][1],
                #         'test/dx0': info['dx'][0],
                #         'test/dx1': info['dx'][1],
                #         'test/dx2': info['dx'][2] if info['dx'].shape[0] == 3 else 0,
                #         'test/x0': info['x'][0],
                #         'test/x1': info['x'][1],
                #         'test/x2': info['x'][2] if info['x'].shape[0] == 3 else 0,
                #         'test/o0': info['o'][0],
                #         'test/o1': info['o'][1],
                #         'test/c_0': info['c'][0],
                #         'test/c_1': info['c'][1],
                #         'test/c_2': info['c'][2],
                #         'test/c_3': info['c'][3],
                #     })

                total_steps += 1

        # log = {}
        # for column in columns:
        #     log[column] = wandb.plot.line(report_table, 'step', column)

        return ra / repeat, rd / repeat, report_table

    def initialize_strategies(self):
        attacker = ZeroAgent(4)  # TODO this should not be a constant 4
        self.logger.info('Initializing a defender against NoOp attacker...')
        defender = self.train_defender(attacker, 0)

        au, du, _ = self.get_payoff(attacker, defender)

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

        wandb.run.summary.update({'base_defender_payoff': du})

        return attacker_ms, defender_ms
