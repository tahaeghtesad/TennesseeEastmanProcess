import gym
import safety_gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ppo1 import PPO1
import tensorflow as tf


def get_policy_class(policy_params):
    # policy params must have 'act_fun' and 'layers'
    class CustomPolicy(MlpPolicy):
        def __init__(self, *args, **_kwargs):
            super().__init__(*args, **_kwargs, **policy_params)

    return CustomPolicy

robot = 'Car'
env = gym.make(f'Safexp-{robot}Goal0-v0')

attacker_model = PPO1(
            policy=get_policy_class(dict(
                net_arch=dict(vf=[128, 64],
                              pi=[64, 64]),
                act_fun=tf.nn.tanh
            )),
            env=env,
            gamma=0.995,
            verbose=2,
            entcoeff=0,
            lam=0.97,
            optim_batchsize=1024,
            tensorboard_log='tb_logs/',
            full_tensorboard_log=True
        )

attacker_model.learn(
    total_timesteps=5_000_000
)