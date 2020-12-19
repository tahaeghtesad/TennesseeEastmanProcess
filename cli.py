import logging
import sys

import click
import numpy as np
from mpi4py import MPI

from agents.RLAgents import ZeroAgent
from trainer import MTDTrainer, RCTrainer
import tensorflow as tf
from util.nash_helpers import find_general_sum_mixed_ne, find_zero_sum_mixed_ne_gambit, get_payoff_from_table
import json
import os
import copy
import wandb


def init_logger(prefix, index):
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(f'{prefix}/{index}/log.log', mode='w')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    return rootLogger


def do_marl(prefix, index, group, params, max_iter, trainer_class, nash_solver):
    # index = f'{index}_{str(np.random.randint(0, 1000)).zfill(4)}'

    if not os.path.exists(f'{prefix}'):
        os.makedirs(f'{prefix}')
    if not os.path.exists(f'{prefix}/{index}'):
        os.makedirs(f'{prefix}/{index}')
        os.makedirs(f'{prefix}/{index}/tb_logs')
        os.makedirs(f'{prefix}/{index}/params')

    logger = init_logger(prefix, index)

    logger.info(f'Prefix: {prefix}')
    logger.info(f'Run ID: {index}')

    logger.info('Starting Double Oracle Framework on DO with parameters:')
    logger.info(f'{params}')

    with open(f'{prefix}/{index}/info.json', 'w') as fd:
        params_back = copy.deepcopy(params)
        params_back['policy_params']['layers'] = str(params_back['policy_params']['layers'])
        params_back['policy_params']['act_fun'] = params_back['policy_params']['act_fun'].__name__
        json.dump(params_back, fd)

    run = wandb.init(project='tep', config=params_back, dir=f'{prefix}/{index}', group=group)
    run.save(f'{prefix}/{index}/info.json')
    run.save(f'{prefix}/{index}/log.log')
    run.save(f'{prefix}/{index}/params/*')
    run.save(f'{prefix}/{index}/*.npy')

    trainer = trainer_class(
        f'{prefix}/{index}',
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

    wandb.log({'payoffs/attacker': trainer.attacker_payoff_table[0, 0], 'payoffs/defender': trainer.defender_payoff_table[0, 0]})

    attacker_iteration = len(attacker_ms.policies)
    defender_iteration = len(defender_ms.policies)

    logger.info(f'Attacker Heuristics: {attacker_iteration}')
    logger.info(f'Defender Heuristics: {defender_iteration}')

    aus, dus = [trainer.attacker_payoff_table[0, 0]], [trainer.defender_payoff_table[0, 0]]

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
        au, du, _ = get_payoff_from_table(nash_solver, trainer.attacker_payoff_table, trainer.defender_payoff_table)
        wandb.log({'payoffs/attacker': au, 'payoffs/defender': du})
        logging.info(f'MSNE Attacker vs MSNE Defender Payoff: {au, du}')
        logging.info(f'Improvement: {(du - aus[-1])/aus[-1]:}')

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
        au, du, _ = get_payoff_from_table(nash_solver, trainer.attacker_payoff_table, trainer.defender_payoff_table)
        wandb.log({'payoffs/attacker': au, 'payoffs/defender': du})
        logging.info(f'MSNE Attacker vs MSNE Defender Payoff: {au, du}')

    # wandb.run.summary.update({'base_defender_payoff': du})

    log_results(attacker_ms, defender_ms, params, trainer, nash_solver)

    # wandb.run.summary['defense'] = defense
    # wandb.run.summary['no_attack'] = no_attack
    # wandb.run.summary['no_defense'] = no_defense
    # wandb.run.summary.update({})
    run.finish(0)


def log_results(attacker_ms, defender_ms, params, trainer, nash_solver):
    attacker_strategy, defender_strategy = nash_solver(trainer.attacker_payoff_table,
                                                       trainer.defender_payoff_table)
    compromise = np.concatenate(
        (np.random.rand(2) < params['env_params']['compromise_observation_prob'],
         np.random.rand(2) < params['env_params']['compromise_actuation_prob'])
        , axis=0).astype(np.float)

    _, du_na, no_attack = trainer.get_payoff(
        attacker_ms.policies[0],
        defender_ms.policies[0],
        repeat=10,
        compromise=compromise,
        log=False
    )

    _, du_nd, no_defense = trainer.get_payoff(
        attacker_ms.policies[np.argmax(attacker_strategy)],
        defender_ms.policies[0],
        repeat=10,
        compromise=compromise,
        log=False
    )

    _, du_d, defense = trainer.get_payoff(
        attacker_ms.policies[np.argmax(attacker_strategy)],
        defender_ms.policies[np.argmax(defender_strategy)],
        repeat=10,
        compromise=compromise,
        log=False
    )
    for step in range(len(no_attack.data)):
        columns = no_attack.columns[1:]
        log = {}
        for i, col in enumerate(columns):
            log[f'report/{col}/no_attack'] = no_attack.data[step][i + 1]
            log[f'report/{col}/defense'] = defense.data[step][i + 1]
            log[f'report/{col}/no_defense'] = no_defense.data[step][i + 1]
        wandb.log(log)

    wandb.run.summary.update({
        'final_payoff/defense': du_d,
        'final_payoff/no_defense': du_nd,
        'final_payoff/no_attack': du_na,
    })


def to_bool(input):
    if type(input) is bool:
        return input
    return input.lower() in ['true']


@click.command(name='mtd')
@click.option('--prefix', default='runs', help='Prefix folder of run results', show_default=True)
@click.option('--index', help='Index for this run', required=True)
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
def do_mtd(prefix, index,
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

    do_marl(prefix, index, params, max_iter, MTDTrainer, find_general_sum_mixed_ne)


@click.command(name='rc')
@click.option('--env_id', help='Name of the training environment', required=True)
@click.option('--prefix', default='runs', help='Prefix folder of run results', show_default=True)
@click.option('--group', default=None, help='WandB group name', show_default=True)
@click.option('--index', help='Index for this run', required=True)
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
@click.option('--policy_params_activation', default='tanh', help='Activation Function', show_default=True)
@click.option('--policy_params_layers', default='32, 32', help='MLP Network Layers', show_default=True)
def do_rc(env_id,
          prefix,
          index,
          group,
          max_iter,
          training_params_training_steps,
          training_params_concurrent_runs,
          training_params_tb_logging,
          training_params_action_noise_sigma,
          rl_params_random_exploration,
          rl_params_gamma,
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
            'test_env': env_params_test_env,
            't_epoch': env_params_t_epoch
        },
        'rl_params': {
            'random_exploration': rl_params_random_exploration,
            'gamma': rl_params_gamma,
        },
        'policy_params': {
            'act_fun': getattr(tf.nn, policy_params_activation),
            'layers': [int(l) for l in policy_params_layers.split(',')]
        }
    }

    group = None if group is None or group=='' else group

    do_marl(prefix, index, group, params, max_iter, RCTrainer, find_zero_sum_mixed_ne_gambit)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    pass


cli.add_command(do_rc)
cli.add_command(do_mtd)

if __name__ == '__main__':
    print(os.environ['PATH'])
    cli()
