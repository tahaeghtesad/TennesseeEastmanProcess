import gym
import os
import numpy as np
import tensorflow as tf
from stable_baselines import *
from stable_baselines.ddpg.policies import LnMlpPolicy
import gym_control
from agents.ddpg_agent import DDPGWrapper

if __name__ == '__main__':

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

                summary = tf.Summary(
                    value=[tf.Summary.Value(tag=f'env/a{i}', simple_value=locals_['info']['a'][i])])
                locals_['writer'].add_summary(summary, self_.num_timesteps)

                summary = tf.Summary(
                    value=[tf.Summary.Value(tag=f'env/o{i}', simple_value=locals_['info']['o'][i])])
                locals_['writer'].add_summary(summary, self_.num_timesteps)

        return True

    class CustomPolicy(LnMlpPolicy):
        def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
            super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                             act_fun=tf.nn.elu,
                             layers=[32] * 4,
                             **_kwargs)


    env = gym.make('BRPAtt-v0', defender=DDPGWrapper.load('bioreactor'))

    model = DDPG(
        CustomPolicy,
        env,
        verbose=2,
        random_exploration=.1,
        gamma=.95,
        full_tensorboard_log=True,
        tensorboard_log='tb_logs'
    )

    # model.load('bioreactor')
    # model.random_exploration = 0.0

    try:
        model.learn(total_timesteps=500 * 1000,
                    callback=callback,
                    tb_log_name='DDPG_Attacker')

    except KeyboardInterrupt:
        print('Interrupting...')
    finally:
        model.save('bioreactor_att')
        print('Model Saved.')
