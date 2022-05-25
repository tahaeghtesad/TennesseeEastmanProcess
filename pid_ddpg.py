import os

import tensorflow as tf
from tqdm import tqdm

from agents.HeuristicAgents import PIDController
from agents.RLAgents import SimpleWrapperAgent
from envs.control.envs import ThreeTankAttacker
from envs.control.threat.tep_threat import TEPThreatModel
from stable_baselines import DDPG
from stable_baselines.ddpg import MlpPolicy


def get_policy_class(policy_params):
    # policy params must have 'act_fun' and 'layers'
    class CustomPolicy(MlpPolicy):
        def __init__(self, *args, **_kwargs):
            super().__init__(*args, **_kwargs, **policy_params)

    return CustomPolicy


def train_attacker(env):
    if os.path.isfile(f'pid_adversary.zip'):
        model = DDPG.load(f'pid_adversary.zip')
    else:
        model = DDPG(
            policy=get_policy_class(dict(
                layers=[64, 64],
                act_fun=tf.nn.tanh
            )),
            env=env,
            gamma=0.99,
            verbose=1,
            full_tensorboard_log=False
        )

        model.learn(
            total_timesteps=200_000
    )

    model.save(f'pid_adversary')
    agented_model = SimpleWrapperAgent('pid_adv', model)
    return agented_model


def eval_agents(env, attacker, defender):
    epochs = 25
    print(f'Evaluating {attacker} vs. {defender} in environment {env} for {epochs} epoch(s).')

    env.set_attacker(attacker)
    env.set_defender(defender)

    defender_rewards = []
    attacker_rewards = []
    lengths = []

    for epoch in tqdm(range(epochs)):
        done = False
        _ = env.reset()
        adversary_episode_reward = 0
        defender_episode_reward = 0
        step = 0
        while not done:
            _, (r_a, r_d), done, _ = env.step()
            adversary_episode_reward += r_a
            defender_episode_reward += r_d
            step += 1
        defender_rewards.append(defender_episode_reward)
        attacker_rewards.append(adversary_episode_reward)
        lengths.append(step)


    print(f'Average episode length: {sum(lengths)/len(lengths)}')
    print(f'Average episode reward for adversary: {sum(attacker_rewards) / len(attacker_rewards)}')
    print(f'Average episode reward for defender: {sum(defender_rewards) / len(defender_rewards)}')
    print(f'Average step reward for adversary: {sum(attacker_rewards) / sum(lengths)}')
    print(f'Average step reward for defender: {sum(defender_rewards) / sum(lengths)}')

    return sum(attacker_rewards)/len(attacker_rewards), sum(defender_rewards)/len(defender_rewards)


if __name__ == '__main__':
    env = ThreeTankAttacker(defender=None,
                            history_length=1,
                            include_compromise=False,
                            compromise_observation_prob=1,
                            compromise_actuation_prob=1,
                            test_env=False,
                            noise_sigma=0.00007)

    eval_env = TEPThreatModel('TT-v0',
                              attacker=None,
                              defender=None,
                              history_length=1,
                              include_compromise=False,
                              compromise_observation_prob=1,
                              compromise_actuation_prob=1,
                              test_env=False,
                              noise_sigma=0.00007
                              )

    defender = PIDController(2.531264075100467, 0.3170905108653836, 0.001,
                             env.adversarial_control_env.env.action_space,
                             env.adversarial_control_env.env.goal[:2], 'PID')

    env.adversarial_control_env.set_defender(defender)

    attacker = train_attacker(env)


    eval_agents(eval_env, attacker, defender)
