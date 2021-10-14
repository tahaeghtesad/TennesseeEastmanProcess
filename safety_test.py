import os

import gym
import safety_gym
from agents.RLAgents import ZeroAgent, SimpleWrapperAgent
from envs.control.envs.safety import SafetyEnvAttacker, SafetyEnvDefender
from envs.control.threat.safety_theat import SafetyThreatModel
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ppo2 import PPO2
import tensorflow as tf
import numpy as np


def get_policy_class(policy_params):
    # policy params must have 'act_fun' and 'layers'
    class CustomPolicy(MlpPolicy):
        def __init__(self, *args, **_kwargs):
            super().__init__(*args, **_kwargs, **policy_params)

    return CustomPolicy


def callback(locals_, globals_):
    self_ = locals_['self']
    if 'reward' in locals_:
        if 'writer' in locals_ and locals_['writer'] is not None:
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=f'env/reward', simple_value=locals_['reward'])])
            locals_['writer'].add_summary(summary, self_.num_timesteps)


def eval_agents(env, attacker, defender):
    env = SafetyThreatModel(env, attacker, defender)
    _ = env.reset()

    defender_rewards = []
    attacker_rewards = []

    done = False
    while not done:
        _, (r_a, r_d), done, _ = env.step()
        defender_rewards.append(r_d)
        attacker_rewards.append(r_a)

    return sum(attacker_rewards)/len(attacker_rewards), sum(defender_rewards)/len(defender_rewards)


robot = 'Car'
env_name = f'Safexp-{robot}Goal0-v0'
base_model_path = 'lee-models'
repeat = 1
train_length = 1_000

if __name__ == '__main__':
    if not os.path.isdir(base_model_path):
        os.makedirs(base_model_path, exist_ok=True)

    # Train Nominal
    nominal_models = []
    nominal_average_rewards = []

    for i in range(repeat):

        nominal_model = PPO2(
            policy=get_policy_class(dict(
                net_arch=[dict(vf=[128, 64],
                               pi=[64, 64])],
                act_fun=tf.nn.tanh
            )),
            env=SafetyEnvDefender(env_name, ZeroAgent(2)),
            gamma=0.995,
            verbose=0,
            lam=0.97,
            tensorboard_log='tb_logs/',
            full_tensorboard_log=True
        )

        nominal_model.learn(
            total_timesteps=train_length,
            callback=callback,
            tb_log_name=f'nominal_{i}'
        )
        nominal_model.save(f'{base_model_path}/nominal_{i}')
        agented_model = SimpleWrapperAgent(nominal_model)
        agent_reward = eval_agents(env_name, ZeroAgent(2), agented_model)[1]
        nominal_models.append(agented_model)
        nominal_average_rewards.append(agent_reward)
        print(f'Nominal/{i} - reward: {agent_reward:.2f}')

    # Train Att 1

    best_nominal = nominal_models[np.argmax(nominal_average_rewards)]
    att1_models = []
    att1_average_rewards = []

    for i in range(repeat):
        att1_model = PPO2(
            policy=get_policy_class(dict(
                net_arch=[dict(vf=[128, 64],
                               pi=[128, 128])],
                act_fun=tf.nn.tanh
            )),
            env=SafetyEnvAttacker(env_name, best_nominal),
            gamma=0.995,
            verbose=0,
            lam=0.97,
            tensorboard_log='tb_logs/',
            full_tensorboard_log=True
        )

        att1_model.learn(
            total_timesteps=train_length,
            callback=callback,
            tb_log_name=f'att1_{i}'
        )
        att1_model.save(f'{base_model_path}/att1_{i}')
        agented_model = SimpleWrapperAgent(att1_model)
        agent_reward = eval_agents(env_name, agented_model, best_nominal)[0]
        att1_models.append(agented_model)
        att1_average_rewards.append(agent_reward)
        print(f'Att1/{i} - reward: {agent_reward:.2f}')

    # Training robust

    best_att1 = att1_models[np.argmax(att1_average_rewards)]
    robust_models = []
    robust_average_rewards = []

    for i in range(repeat):

        robust_model = PPO2(
            policy=get_policy_class(dict(
                net_arch=[dict(vf=[128, 64],
                               pi=[64, 64])],
                act_fun=tf.nn.tanh
            )),
            env=SafetyEnvDefender(env_name, best_att1),
            gamma=0.995,
            verbose=0,
            lam=0.97,
            tensorboard_log='tb_logs/',
            full_tensorboard_log=True
        )

        robust_model.learn(
            total_timesteps=train_length,
            callback=callback,
            tb_log_name=f'robust_{i}'
        )
        robust_model.save(f'{base_model_path}/robust_{i}')
        agented_model = SimpleWrapperAgent(robust_model)
        agent_reward = eval_agents(env_name, best_att1, agented_model)[1]
        robust_models.append(agented_model)
        robust_average_rewards.append(agent_reward)
        print(f'Robust/{i} - reward: {agent_reward:.2f}')

    # Training att 2

    best_robust = robust_models[np.argmax(robust_average_rewards)]
    att2_models = []
    att2_average_rewards = []

    for i in range(repeat):
        att2_model = PPO2(
            policy=get_policy_class(dict(
                net_arch=[dict(vf=[128, 64],
                               pi=[128, 128])],
                act_fun=tf.nn.tanh
            )),
            env=SafetyEnvAttacker(env_name, best_robust),
            gamma=0.995,
            verbose=0,
            lam=0.97,
            tensorboard_log='tb_logs/',
            full_tensorboard_log=True
        )

        att2_model.learn(
            total_timesteps=train_length,
            callback=callback,
            tb_log_name=f'att1_{i}'
        )
        att2_model.save(f'{base_model_path}/att2_{i}')
        agented_model = SimpleWrapperAgent(att2_model)
        agent_reward = eval_agents(env_name, agented_model, best_robust)[0]
        att2_models.append(agented_model)
        att2_average_rewards.append(agent_reward)
        print(f'Att2/{i} - reward: {agent_reward:.2f}')
