import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from agents.RLAgents import SimpleWrapperAgent
from envs.env_helpers import PIDHelper
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

lines = []


def get_policy_class(policy_params):
    # policy params must have 'act_fun' and 'layers'
    class CustomPolicy(MlpPolicy):
        def __init__(self, *args, **_kwargs):
            super().__init__(*args, **_kwargs, **policy_params)

    return CustomPolicy


def train_nominal(env, name):
    if os.path.isfile(f'{name}.zip'):
        # model = DDPG.load(f'{name}.zip', env)
        model = PPO2.load(f'{name}.zip', env)
    else:
        # model = DDPG(
        #     policy=get_policy_class(dict(
        #         layers=[512, 512],
        #         act_fun=tf.nn.relu
        #     )),
        #     env=env,
        #     gamma=0.95,
        #     verbose=1,
        #     full_tensorboard_log=True,
        #     tensorboard_log='./logs/',
        #     # random_exploration=0.1,
        #     # action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2)),
        #     action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2)),
        #     # critic_lr=1e-4,
        #     # actor_lr=1e-5,
        #     critic_lr=LinearSchedule(80_000, 1e-5, 1e-3),
        #     actor_lr=LinearSchedule(80_000, 1e-6, 1e-4),
        # )

        model = PPO2(
            policy=get_policy_class(dict(
                net_arch=[dict(vf=[256, 128],
                               pi=[256, 128])],
                act_fun=tf.nn.relu)),
            env=env,
            gamma=0.95,
            verbose=1,
            full_tensorboard_log=True,
            tensorboard_log='./logs/',
            ent_coef=0.0000,
            learning_rate=2.5e-5,

        )

    try:
        model.learn(
            total_timesteps=1_000_000,
            tb_log_name=f'{name}',
        )
    except KeyboardInterrupt:
        pass

    model.save(f'{name}')
    agented_model = SimpleWrapperAgent(f'{name}', model)
    return agented_model


def eval(env, agented_model):
    obs_0 = []
    obs_1 = []
    obs_2 = []
    action_0 = []
    action_1 = []
    rewards = []
    final_reward = 0

    env.reset()

    initial_state = np.array([0.3585652266831299, 0.17014550938055367, 0.22622737593724257])
    obs = np.repeat(initial_state[:2], 3)
    # env.adversarial_control_env.x = initial_state
    env.env.x = initial_state
    lines.append([0, initial_state[0], initial_state[1], initial_state[2], 0, 0, -0.08349962])

    for step in range(50):
        action = agented_model.predict(obs)
        obs, reward, done, info = env.step(action)
        obs_0.append(info['x'][0])
        obs_1.append(info['x'][1])
        obs_2.append(info['x'][2])
        action_0.append(action[0])
        action_1.append(action[1])
        rewards.append(reward)

        final_reward += (0.99 ** step) * reward

        lines.append([f'{c:2.8f}' for c in [step + 1, obs[0], obs[1], info['x'][2], action[0], action[1], reward]])
        lines[-1][0] = step + 1

    plt.plot(obs_0)
    plt.plot(obs_1)
    plt.plot(obs_2)
    plt.legend(['0', '1', '2'])
    plt.title('State')
    plt.show()

    plt.plot(action_0)
    plt.plot(action_1)
    plt.title('Action')
    plt.legend(['0', '1'])
    plt.show()

    plt.plot(rewards)
    plt.title('Reward')
    plt.show()

    print(final_reward)

    with open('tt_ddpg_base.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows([['step', 'obs_0', 'obs_1', 'obs_2', 'action_0', 'action_1', 'reward']])
        csv_writer.writerows(lines)


if __name__ == '__main__':
    # env = ThreeTankDefender(attacker=ZeroAgent('zero', 4),
    #                         history_length=8,
    #                         include_compromise=False,
    #                         compromise_observation_prob=0,
    #                         compromise_actuation_prob=0,
    #                         test_env=False,
    #                         noise_sigma=0.00007,
    #                         t_epoch=25)

    env = PIDHelper('TT-v0', test_env=False, noise_sigma=0.0, t_epoch=100)

    controller = train_nominal(env, 'pid_support')
    eval(env, controller)


