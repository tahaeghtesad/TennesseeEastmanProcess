from stable_baselines import DDPG
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from gym_control.envs.BioReactorEnv import BioReactor
import numpy as np
import tensorflow as tf

def callback(locals_, globals_):
    self_ = locals_['self']

    if 'info' in locals_:
        summary = tf.Summary(
            value=[tf.Summary.Value(tag='game/x0', simple_value=locals_['info']['x'][0][0])])
        locals_['writer'].add_summary(summary, self_.num_timesteps)

        summary = tf.Summary(
            value=[tf.Summary.Value(tag='game/dx0', simple_value=locals_['info']['dx'][0][0])])
        locals_['writer'].add_summary(summary, self_.num_timesteps)

        summary = tf.Summary(
            value=[tf.Summary.Value(tag='game/x1', simple_value=locals_['info']['x'][1][0])])
        locals_['writer'].add_summary(summary, self_.num_timesteps)

        summary = tf.Summary(
            value=[tf.Summary.Value(tag='game/dx1', simple_value=locals_['info']['dx'][1][0])])
        locals_['writer'].add_summary(summary, self_.num_timesteps)

        summary = tf.Summary(
            value=[tf.Summary.Value(tag='game/u0', simple_value=locals_['info']['u'][0])])
        locals_['writer'].add_summary(summary, self_.num_timesteps)

        summary = tf.Summary(
            value=[tf.Summary.Value(tag='game/u1', simple_value=locals_['info']['u'][1])])
        locals_['writer'].add_summary(summary, self_.num_timesteps)

    return True


if __name__ == '__main__':

    action_space = BioReactor.action_space
    observation_space = BioReactor.observation_space
    # env = SubprocVecEnv([BioReactor for _ in range(4)])
    env = BioReactor()

    model = DDPG(
        MlpPolicy,
        env,
        # param_noise=AdaptiveParamNoiseSpec(),
        # action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(2,), sigma=float(0.2) * np.ones(2,)),
        gamma=0.9,
        full_tensorboard_log=True,
        tensorboard_log='tb_logs'
    )

    model.learn(500 * 1000, callback=callback)
    model.save("bioreactor")
