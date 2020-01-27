import gym
import os
import numpy as np
import tensorflow as tf
from stable_baselines import *
from stable_baselines.ddpg.policies import LnMlpPolicy
import gym_control
from gym_control.envs.BioReactor import AttackerMode
from agents.ddpg_agent import DDPGWrapper
import math
import nashpy as nash
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

TOTAL_TRAINING_STEPS = 500 * 1000

def callback(locals_, globals_):
    self_ = locals_['self']

    if 'info' in locals_:

        for i in range(2):
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=f'env/u{i}', simple_value=locals_['info']['u'][i])])
            locals_['writer'].add_summary(summary, self_.num_timesteps)

            summary = tf.Summary(
                value=[tf.Summary.Value(tag=f'env/x{i}', simple_value=locals_['info']['x'][i])])
            locals_['writer'].add_summary(summary, self_.num_timesteps)

            summary = tf.Summary(
                value=[tf.Summary.Value(tag=f'env/dx{i}', simple_value=locals_['info']['dx'][i])])
            locals_['writer'].add_summary(summary, self_.num_timesteps)

            if 'a' in locals_['info']:
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag=f'env/a{i}', simple_value=locals_['info']['a'][i])])
                locals_['writer'].add_summary(summary, self_.num_timesteps)

            if 'o' in locals_['info']:
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag=f'env/o{i}', simple_value=locals_['info']['o'][i])])
                locals_['writer'].add_summary(summary, self_.num_timesteps)

    return True


class CustomPolicy(LnMlpPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                         act_fun=tf.nn.elu,
                         layers=[64] * 4,
                         **_kwargs)


class Trainer:

    def __init__(self):
        # Columns are attackers, rows are defenders
        self.attacker_payoff_table = np.array([[]])
        self.defender_payoff_table = np.array([[]])

        if isfile('attacker_payoff.npy') and isfile('defender_payoff.npy'):
            self.load_tables()

    def train_attacker(self, attacker_choice, defender_choice):  # TODO propose new name for defender_choice!
        model = DDPG(
            CustomPolicy,
            gym.make('BRPAtt-v0',
                     defender=None,
                     mode=AttackerMode.Actuation,
                     range=np.float(0.3)),  # This is a dummy env
            verbose=2,
            random_exploration=.1,
            gamma=.95,
            full_tensorboard_log=True,
            tensorboard_log='tb_logs'
        )

        for i in range(len(defender_choice)):
            if defender_choice[i] != 0.0:
                env = gym.make('BRPAtt-v0',
                               defender=DDPGWrapper.load(f'params/defender-{i}'),
                               mode=AttackerMode.Actuation,
                               range=np.float(0.3))

                model.set_env(env)

                model.learn(total_timesteps=round(defender_choice[i] * TOTAL_TRAINING_STEPS),
                            callback=callback,
                            tb_log_name=f'Attacker')

        model.save(f'params/attacker-{len(attacker_choice)}')

        attacker_util, defender_util = self.get_attacker_payoff(f'attacker-{len(attacker_choice)}')
        self.attacker_payoff_table = np.hstack([self.attacker_payoff_table, attacker_util.reshape((attacker_util.size, 1))])
        self.defender_payoff_table = np.hstack([self.defender_payoff_table, defender_util.reshape((defender_util.size, 1))])
        self.save_tables()

    def train_defender(self, attacker_choice, defender_choice):
        model = DDPG(
            CustomPolicy,
            gym.make('BRPDef-v0',
                     attacker=None),  # This is a dummy env
            verbose=2,
            random_exploration=.1,
            gamma=.95,
            full_tensorboard_log=True,
            tensorboard_log='tb_logs'
        )

        for i in range(len(attacker_choice)):
            if attacker_choice[i] != 0.0:
                env = gym.make('BRPDef-v0',
                               attacker=DDPGWrapper.load(f'params/attacker-{i}'))

                model.set_env(env)

                model.learn(total_timesteps=round(attacker_choice[i] * TOTAL_TRAINING_STEPS),
                            callback=callback,
                            tb_log_name=f'Defender')

        model.save(f'params/defender-{len(defender_choice)}')

        attacker_util, defender_util = self.get_defender_payoff(f'defender-{len(defender_choice)}')
        self.attacker_payoff_table = np.vstack([self.attacker_payoff_table, attacker_util.reshape((1, attacker_util.size))])
        self.defender_payoff_table = np.vstack([self.defender_payoff_table, defender_util.reshape((1, defender_util.size))])
        self.save_tables()

    def bootstrap_defender(self):
        model = DDPG(
            CustomPolicy,
            env=gym.make('BRP-v0'),
            verbose=2,
            random_exploration=0.1,
            gamma=0.9,
            full_tensorboard_log=True,
            tensorboard_log='tb_logs'
        )

        model.learn(TOTAL_TRAINING_STEPS,
                    callback=callback,
                    tb_log_name='Defender')

        model.save('params/defender-0')

    def bootstrap_attacker(self):
        model = DDPG(
            CustomPolicy,
            env=gym.make('BRPAtt-v0',
                         defender=DDPGWrapper.load(f'params/defender-0'),
                         mode=AttackerMode.Actuation,
                         range=np.float(0.3)),
            verbose=2,
            random_exploration=0.1,
            gamma=0.9,
            full_tensorboard_log=True,
            tensorboard_log='tb_logs'
        )

        model.learn(TOTAL_TRAINING_STEPS,
                    callback=callback,
                    tb_log_name='Attacker')

        model.save('params/attacker-0')

    def solve_equilibrium(self):
        assert self.attacker_payoff_table.shape == self.defender_payoff_table.shape
        if self.attacker_payoff_table.size == 1:
            return np.array([1.]), np.array([1.])
        elif self.attacker_payoff_table.shape == (1, 2):
            d = np.array([1.])
            a = np.array([0., 0.])
            a[np.argmax(self.attacker_payoff_table)] = 1.
            return a, d
        elif self.attacker_payoff_table.shape == (2, 1):
            a = np.array([1.])
            d = np.array([0., 0.])
            d[np.argmax(self.defender_payoff_table)] = 1.
            return a, d

        game = nash.Game(self.attacker_payoff_table, self.defender_payoff_table)
        equilibrium = game.lemke_howson(initial_dropped_label=0)
        return equilibrium[1], equilibrium[0]

    def save_tables(self):
        np.save('attacker_payoff', self.attacker_payoff_table)
        np.save('defender_payoff', self.defender_payoff_table)

    def load_tables(self):
        self.attacker_payoff_table = np.load('attacker_payoff.npy')
        self.defender_payoff_table = np.load('defender_payoff.npy')

    def evaluate(self, attacker, defender, episodes=10):
        env = gym.make('BRPAtt-v0',
                       defender=DDPGWrapper.load(f'params/{defender}'),
                       mode=AttackerMode.Observation,
                       range=np.float(0.3))
        attacker_model = DDPGWrapper.load(f'params/{attacker}')

        episode_reward = 0

        for i in tqdm(range(episodes)):
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
