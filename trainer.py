import gym
import os
import numpy as np
import tensorflow as tf
from stable_baselines import *
from stable_baselines.ddpg.policies import LnMlpPolicy
import gym_control
from agents.ddpg_agent import DDPGWrapper
import math
import nashpy as nash
from os import listdir
from os.path import isfile, join
from scipy import optimize as op
import logging


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


class Trainer:

    def __init__(self, total_training_steps,
                 env,
                 activation_function=tf.nn.elu,
                 layers=[64] * 4,
                 compromise_actuation_prob=.5,
                 compromise_observation_prob=.5,
                 gamma=.9,
                 tb_log=True,
                 exploration=.1,
                 attacker_power=.3):
        # Columns are attackers, rows are defenders
        self.attacker_power = attacker_power
        self.exploration = exploration
        self.compromise_observation_prob = compromise_observation_prob
        self.compromise_actuation_prob = compromise_actuation_prob

        self.gamma = gamma
        self.layers = layers
        self.tb_log = tb_log

        self.activation_function = activation_function

        self.attacker_payoff_table = np.array([[]])
        self.defender_payoff_table = np.array([[]])

        if isfile('tb_logs/attacker_payoff.npy') and isfile('tb_logs/defender_payoff.npy'):
            self.load_tables()

        self.total_training_steps = total_training_steps
        self.logger = logging.getLogger(__name__)
        self.env = env

    def get_policy_class(self, layers, act_fun):
        class CustomPolicy(LnMlpPolicy):
            def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
                super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                 act_fun=act_fun,
                                 layers=layers,
                                 **_kwargs)
        return CustomPolicy

    def train_attacker(self, attacker_choice, defender_choice):  # TODO propose new name for defender_choice!
        model = DDPG(
            self.get_policy_class(self.layers, self.activation_function),
            gym.make(f'{self.env}Att-v0',
                     defender=None,
                     compromise_actuation_prob=self.compromise_actuation_prob,
                     compromise_observation_prob=self.compromise_observation_prob,
                     power=self.attacker_power),  # This is a dummy env
            verbose=1,
            random_exploration=self.exploration,
            gamma=self.gamma,
            full_tensorboard_log=self.tb_log,
            tensorboard_log='tb_logs'
        )

        for i in range(len(defender_choice)):
            if int(self.total_training_steps * defender_choice[i]) > 150:
                env = gym.make(f'{self.env}Att-v0',
                               defender=DDPGWrapper.load(f'params/defender-{i}'),
                               compromise_actuation_prob=self.compromise_actuation_prob,
                               compromise_observation_prob=self.compromise_observation_prob,
                               power=self.attacker_power)

                model.set_env(env)

                model.learn(total_timesteps=round(defender_choice[i] * self.total_training_steps),
                            callback=callback,
                            tb_log_name=f'Attacker-{i}'
                            )

        model.save(f'params/attacker-{len(attacker_choice)}')

        return f'attacker-{len(attacker_choice)}'

    def update_attacker_payoff_table(self, attacker_util, defender_util):
        self.attacker_payoff_table = np.hstack([self.attacker_payoff_table, attacker_util.reshape((attacker_util.size, 1))])
        self.defender_payoff_table = np.hstack([self.defender_payoff_table, defender_util.reshape((defender_util.size, 1))])
        self.save_tables()

    def train_defender(self, attacker_choice, defender_choice):
        model = DDPG(
            self.get_policy_class(self.layers, self.activation_function),
            gym.make(f'{self.env}Def-v0',
                     attacker=None,
                     compromise_actuation_prob=self.compromise_actuation_prob,
                     compromise_observation_prob=self.compromise_observation_prob),  # This is a dummy env
            verbose=1,
            random_exploration=self.exploration,
            gamma=self.gamma,
            full_tensorboard_log=self.tb_log,
            tensorboard_log='tb_logs'
        )

        for i in range(len(attacker_choice)):
            if int(self.total_training_steps * attacker_choice[i]) > 150:
                env = gym.make(f'{self.env}Def-v0',
                               attacker=DDPGWrapper.load(f'params/attacker-{i}'),
                               compromise_actuation_prob=self.compromise_actuation_prob,
                               compromise_observation_prob=self.compromise_observation_prob
                               )

                model.set_env(env)

                model.learn(total_timesteps=round(attacker_choice[i] * self.total_training_steps),
                            callback=callback,
                            tb_log_name=f'Defender-{i}'
                            )

        model.save(f'params/defender-{len(defender_choice)}')

        return f'defender-{len(defender_choice)}'

    def update_defender_payoff_table(self, attacker_util, defender_util):
        self.attacker_payoff_table = np.vstack([self.attacker_payoff_table, attacker_util.reshape((1, attacker_util.size))])
        self.defender_payoff_table = np.vstack([self.defender_payoff_table, defender_util.reshape((1, defender_util.size))])
        self.save_tables()

    def bootstrap_defender(self):
        model = DDPG(
            self.get_policy_class(self.layers, self.activation_function),
            env=gym.make(f'{self.env}-v0'),
            verbose=1,
            random_exploration=self.exploration,
            gamma=self.gamma,
            full_tensorboard_log=self.tb_log,
            tensorboard_log='tb_logs'
        )

        model.learn(self.total_training_steps,
                    callback=callback,
                    tb_log_name='Defender-0'
                    )

        model.save('params/defender-0')

    def bootstrap_attacker(self):
        model = DDPG(
            self.get_policy_class(self.layers, self.activation_function),
            env=gym.make(f'{self.env}Att-v0',
                         defender=DDPGWrapper.load(f'params/defender-0'),
                         compromise_actuation_prob=self.compromise_actuation_prob,
                         compromise_observation_prob=self.compromise_observation_prob,
                         power=.3),
            verbose=1,
            random_exploration=self.exploration,
            gamma=self.gamma,
            full_tensorboard_log=self.tb_log,
            tensorboard_log='tb_logs'
        )

        model.learn(self.total_training_steps,
                    callback=callback,
                    tb_log_name='Attacker-0'
                    )

        model.save('params/attacker-0')

    def solve_equilibrium(self):
        return self.find_zero_sum_mixed_ne(self.defender_payoff_table),\
               self.find_zero_sum_mixed_ne(-self.defender_payoff_table.transpose())  # Or it can be self.attacker_payoff_table.transpose()

    def save_tables(self):
        np.save('attacker_payoff', self.attacker_payoff_table)
        np.save('defender_payoff', self.defender_payoff_table)

    def load_tables(self):
        self.attacker_payoff_table = np.load('tb_logs/attacker_payoff.npy')
        self.defender_payoff_table = np.load('tb_logs/defender_payoff.npy')

    def evaluate(self, attacker, defender, episodes=10):
        env = gym.make(f'{self.env}Att-v0',
                       defender=DDPGWrapper.load(f'params/{defender}'),
                       compromise_actuation_prob=self.compromise_actuation_prob,
                       compromise_observation_prob=self.compromise_observation_prob,
                       power=.3)
        attacker_model = DDPGWrapper.load(f'params/{attacker}')

        episode_reward = 0

        for i in range(episodes):
            obs = env.reset()
            done = False

            while not done:
                action = attacker_model.predict(obs)
                obs, reward, done, action = env.step(action)

                episode_reward += reward

        average_reward = episode_reward / episodes

        return average_reward, -average_reward

    def get_attacker_payoff(self, attacker):
        defenders = [f.split('/')[-1].split('.')[0] for f in listdir('params') if
                     isfile(join('params', f)) and 'defender' in f]

        attacker_utilities = list()
        defender_utilities = list()

        for defender in defenders:
            a_u, d_u = self.evaluate(attacker, defender)
            attacker_utilities.append(a_u)
            defender_utilities.append(d_u)

        return np.array(attacker_utilities), np.array(defender_utilities)

    def get_defender_payoff(self, defender):
        attackers = [f.split('/')[-1].split('.')[0] for f in listdir('params') if
                     isfile(join('params', f)) and 'attacker' in f]

        attacker_utilities = list()
        defender_utilities = list()

        for attacker in attackers:
            a_u, d_u = self.evaluate(attacker, defender)
            attacker_utilities.append(a_u)
            defender_utilities.append(d_u)

        return np.array(attacker_utilities), np.array(defender_utilities)

    def get_mixed_payoff(self):
        ae, de = self.solve_equilibrium()
        return ((np.outer(de, ae) * self.attacker_payoff_table).sum(),  # Attacker mixed payoff
                (np.outer(de, ae) * self.defender_payoff_table).sum())  # Defender mixed payoff

    def evaluate_new_defender_mixed_attacker(self, defender_utilities):
        ae, _ = self.solve_equilibrium()
        return (ae * defender_utilities).sum()

    def evaluate_new_attacker_mixed_defender(self, attacker_utilities):
        _, de = self.solve_equilibrium()
        return (de * attacker_utilities).sum()

    def find_zero_sum_mixed_ne(self, payoff):
        """
        Function for returning mixed strategies of the first step of double oracle iterations.
        :param payoff: Two dimensinal array. Payoff matrix of the players.
        The row is defender and column is attcker. This is the payoff for row player.
        :return: List, mixed strategy of the attacker and defender at NE by solving maxmini problem.
        """
        # This implementation is based on page 88 of the book multiagent systems (Shoham etc.)
        # http://www.masfoundations.org/mas.pdf
        # n_action = payoff.shape[0]
        m, n = payoff.shape
        c = np.zeros(n)
        c = np.append(c, 1)
        A_ub = np.concatenate((payoff, np.full((m, 1), -1)), axis=1)
        b_ub = np.zeros(m)
        A_eq = np.full(n, 1)
        A_eq = np.append(A_eq, 0)
        A_eq = np.expand_dims(A_eq, axis=0)
        b_eq = np.ones(1)
        bound = ()
        for i in range(n):
            bound += ((0, None),)
        bound += ((None, None),)

        self.logger.debug(f'T:\t\t{payoff.shape}')
        self.logger.debug(f'A_ub:\t{A_ub.shape}\tb_ub:\t{b_ub.shape}')
        self.logger.debug(f'A_eq:\t{A_eq.shape}\tb_eq:\t{b_eq.shape}')
        self.logger.debug(f'c:\t\t{c.shape}')

        res_attacker = op.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bound, method="interior-point")

        self.logger.debug(f'x:\t\t{res_attacker.x.shape}')
        return res_attacker.x[0:n]

