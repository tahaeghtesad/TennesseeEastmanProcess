import gym
import os
import numpy as np
import tensorflow as tf
from stable_baselines import *
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
import gym_control

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

        if 'done' in locals_ and locals_['done'] is True:
            if len(locals_['epoch_episode_rewards'][-100:-1]) == 0:
                mean100 = -np.inf
            else:
                mean100 = float(np.mean(locals_['epoch_episode_rewards'][-100:-1]))
                if locals_['epoch_episode_rewards'][-1] > 0.:
                    print(f"Mean of Last 100 episodes: {mean100:.2f} - Last Episode: {locals_['epoch_episode_rewards'][-1]:.2f}")
            # return mean100 < 99.7 or len(locals_['epoch_episode_rewards']) < 100 # If avg reward is less than that number, end the training

        return True


    class CustomPolicy(LnMlpPolicy):
        def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
            super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                             act_fun=tf.nn.elu,
                             layers=[64] * 4,
                             **_kwargs)


    env = gym.make('BRP-v0')

    model = DDPG(
        CustomPolicy,
        env,
        verbose=2,
        action_noise=OrnsteinUhlenbeckActionNoise(mean=0, sigma=0.3),
        param_noise=AdaptiveParamNoiseSpec(desired_action_stddev=0.1),
        gamma=.95,
        full_tensorboard_log=True,
        tensorboard_log='tb_logs'
    )

    # model.load('bioreactor')
    # model.random_exploration = 0.0

    try:
        model.learn(300 * 1000,
                    callback=callback,
                    tb_log_name='DDPG_Noise')

    except KeyboardInterrupt:
        print('Intrupting')
    finally:
        model.save('bioreactor_noise')
