import logging
import sys

import click
import numpy as np

from agents.RLAgents import ConstantAgent, ZeroAgent
from envs.control.heuristics.attackers import AlternatingAttacker
from trainer import MTDTrainer, RCTrainer
import tensorflow as tf
from util.nash_helpers import find_general_sum_mixed_ne, find_zero_sum_mixed_ne_gambit, get_payoff_from_table
import json
import os
import copy
import wandb

tmpdir = os.environ['TMPDIR']


def init_logger(path):
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(f'{path}/log.log', mode='w')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    return rootLogger


def do_marl(group, params, max_iter, trainer_class, nash_solver):
    # index = f'{index}_{str(np.random.randint(0, 1000)).zfill(4)}'

    datadir = f'{tmpdir}/data'

    os.makedirs(f'{datadir}')
    os.makedirs(f'{datadir}/tb_logs')
    os.makedirs(f'{datadir}/params')

    logger = init_logger(f'{datadir}')

    logger.info('Starting Double Oracle Framework on DO with parameters:')
    logger.info(f'{params}')

    with open(f'{datadir}/info.json', 'w') as fd:
        params_back = copy.deepcopy(params)
        params_back['policy_params']['layers'] = str(params_back['policy_params']['layers'])
        params_back['policy_params']['act_fun'] = params_back['policy_params']['act_fun'].__name__
        params_back['max_iter'] = max_iter
        json.dump(params_back, fd)

    run = wandb.init(project='tep', config=params_back, dir=f'{tmpdir}/wandb/', group=group, reinit=True, settings=wandb.Settings(_disable_stats=True))
    run.save(f'{datadir}/info.json')
    run.save(f'{datadir}/params/*')
    run.save(f'{datadir}/*.npy')

    trainer = trainer_class(
        f'{datadir}',
        **params,
    )

    logger.info('Initializing Heuristic Strategies...')
    attacker_ms, defender_ms = trainer.initialize_strategies()
    attacker_strategy, defender_strategy = nash_solver(trainer.attacker_payoff_table,
                                                       trainer.defender_payoff_table)
    logging.info(f'Attacker MSNE: {attacker_strategy}')
    logging.info(f'Defender MSNE: {defender_strategy}')
    attacker_ms.update_probabilities(attacker_strategy)
    defender_ms.update_probabilities(defender_strategy)
    wandb.log({'payoffs/attacker': trainer.attacker_payoff_table[0, 0],
               'payoffs/defender': trainer.defender_payoff_table[0, 0],
               'strategies/attacker': wandb.Histogram(np_histogram=(
                   np.pad(attacker_strategy, (0, max_iter - attacker_strategy.shape[0])),
                   np.arange(max_iter + 1)
               )),
               'strategies/defender': wandb.Histogram(np_histogram=(
                   np.pad(defender_strategy, (0, max_iter - defender_strategy.shape[0])),
                   np.arange(max_iter + 1)
               ))})

    attacker_iteration = len(attacker_ms.policies)
    defender_iteration = len(defender_ms.policies)

    logger.info(f'Attacker Heuristics: {attacker_iteration}')
    logger.info(f'Defender Heuristics: {defender_iteration}')

    while attacker_iteration < max_iter or defender_iteration < max_iter:
        # Train Attacker
        logger.info(f'Training Attacker {attacker_iteration}')
        attacker_strategy, defender_strategy = nash_solver(trainer.attacker_payoff_table,
                                                           trainer.defender_payoff_table)
        logging.info(f'Defender MSNE: {defender_strategy}')
        defender_ms.update_probabilities(defender_strategy)
        attacker_policy = trainer.train_attacker(defender_ms, attacker_iteration)
        attacker_ms.add_policy(attacker_policy)
        payoffs = [trainer.get_payoff(attacker_policy, defender_policy) for defender_policy in defender_ms.policies]
        trainer.update_attacker_payoff_table(np.array([au for (au, du, _) in payoffs]),
                                             np.array([du for (au, du, _) in payoffs]))
        attacker_iteration += 1
        au, du = get_payoff_from_table(nash_solver, trainer.attacker_payoff_table, trainer.defender_payoff_table)
        wandb.log({'payoffs/attacker': au,
                   'payoffs/defender': du,
                   'strategies/attacker': wandb.Histogram(np_histogram=(
                       np.pad(attacker_strategy, (0, max_iter - attacker_strategy.shape[0])),
                       np.arange(max_iter + 1)
                   )),
                   'strategies/defender': wandb.Histogram(np_histogram=(
                       np.pad(defender_strategy, (0, max_iter - defender_strategy.shape[0])),
                       np.arange(max_iter + 1)
                   ))})
        logging.info(f'MSNE Attacker vs MSNE Defender Payoff: {au, du}')

        # Train Defender
        logger.info(f'Training Defender {defender_iteration}')
        attacker_strategy, defender_strategy = nash_solver(trainer.attacker_payoff_table,
                                                           trainer.defender_payoff_table)
        logging.info(f'Attacker MSNE: {attacker_strategy}')
        attacker_ms.update_probabilities(attacker_strategy)
        defender_policy = trainer.train_defender(attacker_ms, defender_iteration)
        defender_ms.add_policy(defender_policy)
        payoffs = [trainer.get_payoff(attacker_policy, defender_policy) for attacker_policy in attacker_ms.policies]
        trainer.update_defender_payoff_table(np.array([au for (au, du, _) in payoffs]),
                                             np.array([du for (au, du, _) in payoffs]))
        defender_iteration += 1
        au, du = get_payoff_from_table(nash_solver, trainer.attacker_payoff_table, trainer.defender_payoff_table)
        wandb.log({'payoffs/attacker': au,
                   'payoffs/defender': du,
                   'strategies/attacker': wandb.Histogram(np_histogram=(
                       np.pad(attacker_strategy, (0, max_iter - attacker_strategy.shape[0])),
                       np.arange(max_iter + 1)
                   )),
                   'strategies/defender': wandb.Histogram(np_histogram=(
                       np.pad(defender_strategy, (0, max_iter - defender_strategy.shape[0])),
                       np.arange(max_iter + 1)
                   ))})
        logging.info(f'MSNE Attacker vs MSNE Defender Payoff: {au, du}')

    log_results(attacker_ms, defender_ms, params, trainer, nash_solver)

    run.finish(0)


def get_best_strategy(strategy):
    sort_list = np.argsort(strategy)
    if sort_list.shape[0] == 1:
        return sort_list[0]

    if sort_list[0] == 0:
        return sort_list[1]

    return sort_list[0]


def log_results(attacker_ms, defender_ms, params, trainer, nash_solver, repeat=64):
    compromise_list = {}

    def gen_compromise(step):
        if not step in compromise_list:
            compromise_list[step] = np.concatenate(
                (np.random.rand(2) < params['env_params']['compromise_observation_prob'],
                 np.random.rand(2) < params['env_params']['compromise_actuation_prob'])
                , axis=0).astype(np.float)
        return compromise_list[step]

    attacker_strategy, defender_strategy = nash_solver(trainer.attacker_payoff_table,
                                                       trainer.defender_payoff_table)

    ## defender utility from msne_attacker vs. base defender > msne_defender vs. base attacker.

    attacker_ms.update_probabilities(attacker_strategy)
    defender_ms.update_probabilities(defender_strategy)

    _, du_na, no_attack = trainer.get_payoff(  # No Attack system operation
        ZeroAgent(4),
        defender_ms,
        repeat=repeat,
        compromise=gen_compromise,
        log=False
    )

    _, du_msne, msne = trainer.get_payoff(  # What would defense look like!
        attacker_ms,
        defender_ms,
        repeat=repeat,
        compromise=gen_compromise,
        log=False
    )

    _, du_nd, no_defense = trainer.get_payoff(  # What would happen if we just had the first RL.
        attacker_ms,
        defender_ms.policies[0],
        repeat=repeat,
        compromise=gen_compromise,
        log=False
    )

    _, du_alternating, alternating = trainer.get_payoff(
        AlternatingAttacker(4),
        defender_ms,
        repeat=repeat,
        compromise=gen_compromise,
        log=False
    )

    _, du_msne_br, msne_br = trainer.get_payoff(
        attacker_ms.policies[1] if len(attacker_ms.policies) > 1 else attacker_ms,
        defender_ms,
        repeat=repeat,
        compromise=gen_compromise,
        log=False
    )

    _, du_base_br, base_br = trainer.get_payoff(
        attacker_ms.policies[1] if len(attacker_ms.policies) > 1 else attacker_ms,
        defender_ms.policies[0],
        repeat=repeat,
        compromise=gen_compromise,
        log=False
    )

    # for step in range(len(no_attack.data)):
    #     columns = no_attack.columns[1:]
    #     log = {}
    #     for i, col in enumerate(columns):
    #         if col.startswith('c'):  # This is just the compromise vector. It is the same for all envs.
    #             assert no_attack.data[step][i + 1] == no_defense.data[step][i + 1] == msne.data[step][i + 1]
    #             log[f'report/{col}'] = no_attack.data[step][i + 1]
    #         else:
    #             log[f'report/{col}/no_attack'] = no_attack.data[step][i + 1]
    #             log[f'report/{col}/no_defense'] = no_defense.data[step][i + 1]
    #             log[f'report/{col}/msne'] = msne.data[step][i + 1]
    #             log[f'report/{col}/alternating'] = alternating.data[step][i+1]
    #
    #     wandb.log(log)

    wandb.run.summary.update({
        'final_payoff/no_defense': du_nd,
        'final_payoff/no_attack': du_na,
        'final_payoff/msne_eval': du_msne,
        'final_payoff/alternating': du_alternating,
        'final_payoff/msne_table':
            get_payoff_from_table(nash_solver, trainer.attacker_payoff_table, trainer.defender_payoff_table)[1],
        'final_payoff/msne_br': du_msne_br,
        'final_payoff/base_br': du_base_br
    })


def to_bool(input):
    if type(input) is bool:
        return input
    return input.lower() in ['true']


@click.command(name='mtd')
@click.option('--group', help='Group of this run', required=True)
@click.option('--training_params_training_steps', default=500 * 1000,
              help='Number of training steps in each iteration of DO.',
              show_default=True)
@click.option('--training_params_tb_logging', default=True, help='Whether to store Tensorboard logs', show_default=True)
@click.option('--max_iter', default=15, help='Maximum iteration for DO.', show_default=True)
@click.option('--training_params_include_heuristics', default=True, help='Whether to include the heuristics',
              show_default=True)
@click.option('--env_params_m', default=10, help='Number of servers in game.', show_default=True)
@click.option('--env_params_utenv', default=0, help='Utility Environment', show_default=True)
@click.option('--env_params_setting', default=0, help='Environment Setting', show_default=True)
@click.option('--env_params_ca', default=0.05, help='Cost of Attack', show_default=True)
@click.option('--env_params_downtime', default=7, help='$\Delta$ or downtime', show_default=True)
@click.option('--env_params_alpha', default=0.05, help='$\\alpha$ or knowledge gain', show_default=True)
@click.option('--env_params_probe_detection', default=0, help='$\\nu$ or probe detection', show_default=True)
@click.option('--rl_params_exploration_fraction', default=0.2, help='Exploration Fraction', show_default=True)
@click.option('--rl_params_exploration_final_eps', default=0.02, help='Final Exploration', show_default=True)
@click.option('--rl_params_gamma', default=0.99, help='Gamma', show_default=True)
@click.option('--rl_params_double_q', default=False, help='Double_Q MLP Network', show_default=True)
@click.option('--rl_params_prioritized_replay', default=False, help='Prioritized Replay', show_default=True)
@click.option('--rl_params_concurrent_runs', default=4, help='Number of concurrent runs', show_default=True)
@click.option('--policy_params_activation', default='relu', help='Activation Function', show_default=True)
@click.option('--policy_params_layers', default='64, 64', help='MLP Network Layers', show_default=True)
@click.option('--policy_params_dueling', default=True, help='Dueling MLP Network', show_default=True)
@click.option('--policy_params_normalization', default=True, help='Layer Normalization', show_default=True)
def do_mtd(group,
           training_params_training_steps,
           rl_params_concurrent_runs,
           training_params_tb_logging,
           max_iter,
           training_params_include_heuristics,
           env_params_m,
           env_params_utenv,
           env_params_setting,
           env_params_ca,
           env_params_downtime,
           env_params_alpha,
           env_params_probe_detection,
           rl_params_exploration_fraction,
           rl_params_exploration_final_eps,
           rl_params_gamma,
           rl_params_double_q,
           rl_params_prioritized_replay,
           policy_params_activation,
           policy_params_layers,
           policy_params_dueling,
           policy_params_normalization):
    params = {
        'training_params': {
            'training_steps': training_params_training_steps,
            'tb_logging': to_bool(training_params_tb_logging),
            'include_heuristics': to_bool(training_params_include_heuristics),
        },
        'env_params': {
            'm': env_params_m,
            'utenv': env_params_utenv,
            'setting': env_params_setting,
            'ca': env_params_ca,
            'downtime': env_params_downtime,
            'alpha': env_params_alpha,
            'probe_detection': env_params_probe_detection
        },
        'rl_params': {
            'exploration_fraction': rl_params_exploration_fraction,
            'exploration_final_eps': rl_params_exploration_final_eps,
            'gamma': rl_params_gamma,
            'double_q': to_bool(rl_params_double_q),
            'prioritized_replay': rl_params_prioritized_replay,
            'n_cpu_tf_sess': rl_params_concurrent_runs
        },
        'policy_params': {
            'act_fun': getattr(tf.nn, policy_params_activation),
            'layers': [int(l) for l in policy_params_layers.split(',')],
            'dueling': to_bool(policy_params_dueling),
            'layer_norm': to_bool(policy_params_normalization)
        }
    }

    group = None if group is None or group == '' else group

    do_marl(group, params, max_iter, MTDTrainer, find_general_sum_mixed_ne)


@click.command(name='rc')
@click.option('--env_id', help='Name of the training environment', required=True)
@click.option('--group', default=None, help='WandB group name', show_default=True)
@click.option('--max_iter', default=15, help='Maximum iteration for DO.', show_default=True)
@click.option('--training_params_training_steps', default=200 * 1000,
              help='Number of training steps in each iteration of DO.',
              show_default=True)
@click.option('--training_params_concurrent_runs', default=4, help='Number of concurrent runs', show_default=True)
@click.option('--training_params_tb_logging', default=True, help='Whether to store Tensorboard logs', show_default=True)
@click.option('--training_params_action_noise_sigma', default=.5, help='Action Noise for the learning agent')
@click.option('--env_params_compromise_actuation_prob', default=0.5, help='Actuation compromise probability')
@click.option('--env_params_compromise_observation_prob', default=0.5, help='Observation compromise probability')
@click.option('--env_params_power', default=0.3, help='Power of attacker')
@click.option('--env_params_noise_sigma', default=0.07, help='Environment noise sigma')
@click.option('--env_params_history_length', default=12, help='Length of agent history')
@click.option('--env_params_include_compromise', default=True, help='Whether to include compromise to observation')
@click.option('--env_params_test_env', default=False, help='Whether to use set point as starting point of env')
@click.option('--env_params_t_epoch', default=50, help='The number of time steps before an environment is reset')
@click.option('--rl_params_random_exploration', default=0.1, help='Exploration Fraction', show_default=True)
@click.option('--rl_params_gamma', default=0.90, help='Gamma', show_default=True)
@click.option('--rl_params_nb_train', default=30, help='Number of train steps after rollout', show_default=True)
@click.option('--rl_params_nb_rollout', default=100, help='Number of rollout steps before fitting', show_default=True)
@click.option('--rl_params_batch_size', default=128, help='Sample size from experience replay buffer',
              show_default=True)
@click.option('--rl_params_buffer_size', default=5000, help='experience replay buffer size', show_default=True)
@click.option('--policy_params_activation', default='tanh', help='Activation Function', show_default=True)
@click.option('--policy_params_layers', default='32, 32', help='MLP Network Layers', show_default=True)
def do_rc(env_id,
          group,
          max_iter,
          training_params_training_steps,
          training_params_concurrent_runs,
          training_params_tb_logging,
          training_params_action_noise_sigma,
          rl_params_random_exploration,
          rl_params_gamma,
          rl_params_nb_train,
          rl_params_buffer_size,
          rl_params_batch_size,
          rl_params_nb_rollout,
          policy_params_activation,
          policy_params_layers,
          env_params_compromise_actuation_prob,
          env_params_compromise_observation_prob,
          env_params_power,
          env_params_noise_sigma,
          env_params_history_length,
          env_params_include_compromise,
          env_params_test_env,
          env_params_t_epoch):
    params = {
        'env_id': env_id,
        'training_params': {
            'training_steps': training_params_training_steps,
            'concurrent_runs': training_params_concurrent_runs,
            'tb_logging': to_bool(training_params_tb_logging),
            'action_noise_sigma': training_params_action_noise_sigma
        },
        'env_params': {
            'compromise_actuation_prob': env_params_compromise_actuation_prob,
            'compromise_observation_prob': env_params_compromise_observation_prob,
            'history_length': env_params_history_length,
            'power': env_params_power,
            'noise_sigma': env_params_noise_sigma,
            'include_compromise': to_bool(env_params_include_compromise),
            'test_env': to_bool(env_params_test_env),
            't_epoch': env_params_t_epoch
        },
        'rl_params': {
            'random_exploration': rl_params_random_exploration,
            'gamma': rl_params_gamma,
            'nb_train_steps': rl_params_nb_train,
            'nb_rollout_steps': rl_params_nb_rollout,
            'batch_size': rl_params_batch_size,
            'buffer_size': rl_params_buffer_size,
        },
        'policy_params': {
            'act_fun': getattr(tf.nn, policy_params_activation),
            'layers': [int(l) for l in policy_params_layers.split(',')]
        }
    }

    group = None if group is None or group == '' else group

    do_marl(group, params, max_iter, RCTrainer, find_zero_sum_mixed_ne_gambit)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    pass


cli.add_command(do_rc)
cli.add_command(do_mtd)

if __name__ == '__main__':
    cli()
