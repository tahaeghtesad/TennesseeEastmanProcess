import os
import sys

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


def train_attacker(name, env_name, defender):
    model = PPO2(
        policy=get_policy_class(dict(
            net_arch=[dict(vf=[128, 64],
                           pi=[128, 128])],
            act_fun=tf.nn.tanh
        )),
        env=SafetyEnvAttacker(env_name, defender),
        gamma=0.995,
        verbose=1,
        lam=0.97,
        tensorboard_log='tb_logs/',
        full_tensorboard_log=True
    )

    model.learn(
        total_timesteps=train_length,
        callback=callback,
        tb_log_name=f'{name}'
    )
    model.save(f'{base_model_path}/{name}')
    agented_model = SimpleWrapperAgent(model)
    agent_reward = eval_agents(env_name, agented_model, defender)[0]
    print(f'Agent defender {name} evaluated: {agent_reward:.2f}')
    return agented_model


def train_defender(name, env_name, attacker):
    model = PPO2(
        policy=get_policy_class(dict(
            net_arch=[dict(vf=[128, 64],
                           pi=[64, 64])],
            act_fun=tf.nn.tanh
        )),
        env=SafetyEnvDefender(env_name, attacker),
        gamma=0.995,
        verbose=1,
        lam=0.97,
        tensorboard_log='tb_logs/',
        full_tensorboard_log=True
    )

    model.learn(
        total_timesteps=train_length,
        callback=callback,
        tb_log_name=f'{name}'
    )
    model.save(f'{base_model_path}/{name}')
    agented_model = SimpleWrapperAgent(model)
    agent_reward = eval_agents(env_name, attacker, agented_model)[1]
    print(f'Agent attacker {name} evaluated: {agent_reward:.2f}')
    return agented_model


def train_nominal(name, env):
    model = PPO2(
        policy=get_policy_class(dict(
            net_arch=[dict(vf=[128, 64],
                           pi=[64, 64])],
            act_fun=tf.nn.tanh
        )),
        env=gym.make(env),
        gamma=0.995,
        verbose=1,
        lam=0.97,
        tensorboard_log='tb_logs/',
        full_tensorboard_log=True
    )

    model.learn(
        total_timesteps=train_length,
        callback=callback,
        tb_log_name=f'{name}'
    )
    model.save(f'{base_model_path}/{name}')
    agented_model = SimpleWrapperAgent(model)
    agent_reward = eval_agents(env_name, ZeroAgent(2), agented_model)[1]
    print(f'Agent attacker {name} evaluated: {agent_reward:.2f}')
    return agented_model


def callback(locals_, globals_):
    self_ = locals_['self']
    if 'rewards' in locals_:
        if 'writer' in locals_ and locals_['writer'] is not None:
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=f'env/reward', simple_value=locals_['rewards'][0])])
            locals_['writer'].add_summary(summary, self_.model.num_timesteps)
    return True


def eval_agents(env, attacker, defender):
    env = SafetyThreatModel(env, attacker, defender)
    _ = env.reset()

    defender_rewards = []
    attacker_rewards = []

    for epoch in range(10):
        done = False
        while not done:
            _, (r_a, r_d), done, _ = env.step()
            defender_rewards.append(r_d)
            attacker_rewards.append(r_a)

        return sum(attacker_rewards)/len(attacker_rewards), sum(defender_rewards)/len(defender_rewards)


if __name__ == '__main__':

    robot = 'Car'
    env_name = f'Safexp-{robot}Goal0-v0'
    base_model_path = 'lee-models'
    repeat = 5
    train_length = 2_000_000

    if not os.path.isdir(base_model_path):
        os.makedirs(base_model_path, exist_ok=True)

    slurm_id = int(sys.argv[1])

    nominal = train_nominal(f'nominal-{slurm_id}', env_name)
    att1 = train_attacker(f'adversary-1-{slurm_id}', env_name, nominal)
    robust = train_defender(f'robust-{slurm_id}', env_name, att1)
    att2 = train_attacker(f'adversary-2-{slurm_id}', env_name, robust)
