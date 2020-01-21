import gym
import os
import numpy as np
import tensorflow as tf
from stable_baselines import *
from stable_baselines.ddpg.policies import LnMlpPolicy
import gym_control
from agents.ddpg_agent import DDPGWrapper
from evaluate import *
import math
import nashpy as nash

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


# Columns are attackers, rows are defenders
attacker_payoff_table = np.array([[]])
defender_payoff_table = np.array([[]])


def train_attacker(defender_choice):  # TODO propose new name for defender_choice!
    global attacker_payoff_table, defender_payoff_table

    model = DDPG(
        CustomPolicy,
        None,  # TODO check this none
        verbose=2,
        random_exploration=.1,
        gamma=.95,
        full_tensorboard_log=True,
        tensorboard_log='tb_logs'
    )

    for i in range(len(defender_choice)):
        env = gym.make('BRPAtt-v0',
                       defender=DDPGWrapper.load(f'params/defender-{i}'),
                       mode=AttackerMode.Observation,
                       range=np.float(0.3))

        model.set_env(env)

        model.learn(total_timesteps=round(defender_choice[i] * TOTAL_TRAINING_STEPS),
                    callback=callback,
                    tb_log_name=f'Attacker')

    model.save(f'params/attacker-{len(defender_choice)}')

    attacker_util, defender_util = get_attacker_payoff(f'attacker-{len(defender_choice)}')
    attacker_payoff_table = np.hstack([attacker_payoff_table, attacker_util.transpose()])
    defender_payoff_table = np.hstack([defender_payoff_table, defender_util.transpose()])
    save_tables()


def train_defender(attacker_choice):
    global attacker_payoff_table, defender_payoff_table

    model = DDPG(
        CustomPolicy,
        None,
        verbose=2,
        random_exploration=.1,
        gamma=.95,
        full_tensorboard_log=True,
        tensorboard_log='tb_logs'
    )

    for i in range(len(attacker_choice)):
        env = gym.make('BRPDef-v0',
                       attacker=DDPGWrapper.load(f'params/attacker-{i}'))

        model.set_env(env)

        model.learn(total_timesteps=round(attacker_choice[i] * TOTAL_TRAINING_STEPS),
                    callback=callback,
                    tb_log_name=f'Defender')

    model.save(f'params/defender-{len(attacker_choice)}')

    attacker_util, defender_util = get_defender_payoff(f'defender-{len(attacker_choice)}')
    attacker_payoff_table = np.vstack([attacker_payoff_table, attacker_util])
    defender_payoff_table = np.vstack([defender_payoff_table, defender_util])
    save_tables()


def bootstrap_defender():
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


def bootstrap_attacker():
    model = DDPG(
        CustomPolicy,
        env=gym.make('BRPAtt-v0',
                     defender=DDPGWrapper.load(f'params/defender-0'),
                     mode=AttackerMode.Observation,
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


def solve_equilibrium():
    global attacker_payoff_table, defender_payoff_table
    game = nash.Game(attacker_payoff_table, defender_payoff_table)
    equilibrium = game.lemke_howson(initial_dropped_label=0)
    return equilibrium[0], equilibrium[1]


def save_tables():
    np.save('attacker_payoff', attacker_payoff_table)
    np.save('defender_payoff', defender_payoff_table)


def load_tables():
    return np.load('attacker_payoff'), np.load('defender_payoff')
