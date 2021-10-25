import os
import sys

import gym
import safety_gym
from agents.RLAgents import ZeroAgent, SimpleWrapperAgent
from envs.control.envs.safety import SafetyEnvAttacker, SafetyEnvDefender
from envs.control.threat.safety_threat import SafetyThreatModel
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ppo2 import PPO2
import tensorflow as tf
from gym.envs.registration import register
from tqdm import tqdm
import numpy as np

temp_path = os.environ['TMPDIR']


def register_envs():
    for env_id in ['PointGoal2.0-v1', 'CarGoal2.0-v1']:
        env = gym.make(env_id)
        config = env.config
        config['placements_extents'] = [-2.0, -2.0, 2.0, 2.0]
        config['lidar_max_dist'] = 8 * config['placements_extents'][3]
        register(id=env_id,
                 entry_point='safety_gym.envs.mujoco:Engine',
                 kwargs={'config': config})


def get_policy_class(policy_params):
    # policy params must have 'act_fun' and 'layers'
    class CustomPolicy(MlpPolicy):
        def __init__(self, *args, **_kwargs):
            super().__init__(*args, **_kwargs, **policy_params)

    return CustomPolicy


def train_attacker(name, env_name, defender):
    env = make_vec_env(SafetyEnvAttacker, env_kwargs=dict(env=env_name, defender=defender), n_envs=2)

    if os.path.isfile(f'{base_model_path}/{name}.zip'):
        model = PPO2.load(f'{base_model_path}/{name}.zip', env=env)
    else:
        model = PPO2(
            policy=get_policy_class(dict(
                net_arch=[dict(vf=[128, 64],
                               pi=[128, 128])],
                act_fun=tf.nn.tanh
            )),
            env=env,
            gamma=0.995,
            verbose=1,
            lam=0.97,
            tensorboard_log=f'{temp_path}/tb_logs/',
            full_tensorboard_log=True
        )

        model.learn(
            total_timesteps=train_length,
            callback=callback,
            tb_log_name=f'{name}'
        )
    model.save(f'{base_model_path}/{name}')
    agented_model = SimpleWrapperAgent(name, model)
    agent_reward = eval_agents(env_name, agented_model, defender)[0]
    print(f'Agent {name} evaluated: {agent_reward:.4f}')
    return agented_model


def train_defender(name, env_name, attacker):
    env = make_vec_env(SafetyEnvDefender, env_kwargs=dict(env=env_name, attacker=attacker), n_envs=2)

    if os.path.isfile(f'{base_model_path}/{name}.zip'):
        model = PPO2.load(f'{base_model_path}/{name}.zip', env=env)
    else:
        model = PPO2(
            policy=get_policy_class(dict(
                net_arch=[dict(vf=[128, 64],
                               pi=[64, 64])],
                act_fun=tf.nn.tanh
            )),
            env=env,
            gamma=0.995,
            verbose=1,
            lam=0.97,
            tensorboard_log=f'{temp_path}/tb_logs/',
            full_tensorboard_log=True
        )

        model.learn(
            total_timesteps=train_length,
            callback=callback,
            tb_log_name=f'{name}'
        )
        model.save(f'{base_model_path}/{name}')
    agented_model = SimpleWrapperAgent(name, model)
    agent_reward = eval_agents(env_name, attacker, agented_model)[1]
    print(f'Agent {name} evaluated: {agent_reward:.4f}')
    return agented_model


def train_nominal(name, env):
    env = make_vec_env(env, n_envs=2)

    if os.path.isfile(f'{base_model_path}/{name}.zip'):
        model = PPO2.load(f'{base_model_path}/{name}.zip', env=env)
    else:
        model = PPO2(
            policy=get_policy_class(dict(
                net_arch=[dict(vf=[128, 64],
                               pi=[64, 64])],
                act_fun=tf.nn.tanh
            )),
            env=env,
            gamma=0.995,
            verbose=1,
            lam=0.97,
            tensorboard_log=f'{temp_path}/tb_logs/',
            full_tensorboard_log=True
        )

        model.learn(
            total_timesteps=train_length,
            callback=callback,
            tb_log_name=f'{name}'
        )
        model.save(f'{base_model_path}/{name}')
    agented_model = SimpleWrapperAgent(name, model)
    agent_reward = eval_agents(env_name, ZeroAgent('Zero', 2), agented_model)[1]
    print(f'Agent {name} evaluated: {agent_reward:.4f}')
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
    epochs = 25
    print(f'Evaluating {attacker} vs. {defender} in environment {env} for {epochs} epoch(s).')
    env = SafetyThreatModel(env, attacker, defender)

    defender_rewards = []
    attacker_rewards = []
    lengths = []

    for epoch in range(epochs):
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

    robot = 'Car'
    env_name = f'Safexp-{robot}Goal0-v0'
    base_model_path = 'lee-models'
    repeat = 5
    train_length = 1_000_000

    register_envs()

    if not os.path.isdir(base_model_path):
        os.makedirs(base_model_path, exist_ok=True)

    slurm_id = int(sys.argv[1])

    nominal = train_nominal(f'nominal-{slurm_id}', env_name)
    att1 = train_attacker(f'adversary-1-{slurm_id}', env_name, nominal)
    robust = train_defender(f'robust-{slurm_id}', env_name, att1)
    att2 = train_attacker(f'adversary-2-{slurm_id}', env_name, robust)
