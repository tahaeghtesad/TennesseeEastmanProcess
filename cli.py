import logging
import sys

import click
import numpy as np

from trainer import MTDTrainer, RCTrainer
import tensorflow as tf
from util.nash_helpers import find_general_sum_mixed_ne, find_zero_sum_mixed_ne_gambit
import json
import os
import copy


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


def do_marl(prefix, index, params, max_iter, trainer_class, nash_solver):
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
        params_back['policy_params']['act_fun'] = params_back['policy_params']['act_fun'].__name__
        json.dump(params_back, fd)

    trainer = trainer_class(
        f'{prefix}/{index}',
        **params
    )

    logger.info('Initializing Heuristic Strategies...')
    attacker_ms, defender_ms = trainer.initialize_strategies()
    attacker_strategy, defender_strategy = nash_solver(trainer.attacker_payoff_table,
                                                                     trainer.defender_payoff_table)
    logging.info(f'Attacker MSNE: {attacker_strategy}')
    logging.info(f'Defender MSNE: {defender_strategy}')
    attacker_ms.update_probabilities(attacker_strategy)
    defender_ms.update_probabilities(defender_strategy)

    attacker_iteration = len(attacker_ms.policies)
    defender_iteration = len(defender_ms.policies)

    logger.info(f'Attacker Heuristics: {attacker_iteration}')
    logger.info(f'Defender Heuristics: {defender_iteration}')

    while attacker_iteration < max_iter or defender_iteration < max_iter:
        # Train Defender
        logger.info(f'Training Defender {defender_iteration}')
        attacker_strategy, _ = nash_solver(trainer.attacker_payoff_table,
                                           trainer.defender_payoff_table)
        logging.info(f'Attacker MSNE: {attacker_strategy}')
        attacker_ms.update_probabilities(attacker_strategy)
        defender_policy = trainer.train_defender_parallel(attacker_ms, defender_iteration)
        defender_ms.add_policy(defender_policy)
        payoffs = [trainer.get_payoff(attacker_policy, defender_policy) for attacker_policy in attacker_ms.policies]
        trainer.update_defender_payoff_table(np.array([au for (au, du) in payoffs]),
                                             np.array([du for (au, du) in payoffs]))
        defender_iteration += 1
        logging.info(f'MSNE Attacker vs New Defender Payoff: {trainer.get_payoff(attacker_ms, defender_policy)}')

        # Train Attacker
        logger.info(f'Training Attacker {attacker_iteration}')
        _, defender_strategy = nash_solver(trainer.attacker_payoff_table,
                                           trainer.defender_payoff_table)
        logging.info(f'Defender MSNE: {defender_strategy}')
        defender_ms.update_probabilities(defender_strategy)
        attacker_policy = trainer.train_attacker_parallel(defender_ms, attacker_iteration)
        attacker_ms.add_policy(attacker_policy)
        payoffs = [trainer.get_payoff(attacker_policy, defender_policy) for defender_policy in defender_ms.policies]
        trainer.update_attacker_payoff_table(np.array([au for (au, du) in payoffs]),
                                             np.array([du for (au, du) in payoffs]))
        attacker_iteration += 1
        logging.info(f'New Attacker vs MSNE Defender Payoff: {trainer.get_payoff(attacker_policy, defender_ms)}')


def to_bool(input):
    if type(input) is bool:
        return input
    return input.lower() in ['true']


@click.command(name='mtd')
@click.option('--prefix', default='runs', help='Prefix folder of run results', show_default=True)
@click.option('--index', help='Index for this run', required=True)
@click.option('--training_steps', default=500 * 1000, help='Number of training steps in each iteration of DO.',
              show_default=True)
@click.option('--concurrent_runs', default=4, help='Number of concurrent runs', show_default=True)
@click.option('--tb_logging', default=True, help='Whether to store Tensorboard logs', show_default=True)
@click.option('--max_iter', default=15, help='Maximum iteration for DO.', show_default=True)
@click.option('--include_heuristics', default=True, help='Whether to include the heuristics', show_default=True)
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
@click.option('--policy_params_activation', default='relu', help='Activation Function', show_default=True)
@click.option('--policy_params_layers', default='64, 64', help='MLP Network Layers', show_default=True)
@click.option('--policy_params_dueling', default=True, help='Dueling MLP Network', show_default=True)
@click.option('--policy_params_normalization', default=True, help='Layer Normalization', show_default=True)
def do_mtd(prefix, index,
           training_steps,
           concurrent_runs,
           tb_logging,
           max_iter,
           include_heuristics,
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
        'training_steps': training_steps,
        'concurrent_runs': concurrent_runs,
        'tb_logging': to_bool(tb_logging),
        'include_heuristics': to_bool(include_heuristics),
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
            'prioritized_replay': rl_params_prioritized_replay
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
@click.option('--index', help='Index for this run', required=True)
@click.option('--training_steps', default=200 * 1000, help='Number of training steps in each iteration of DO.',
              show_default=True)
@click.option('--concurrent_runs', default=4, help='Number of concurrent runs', show_default=True)
@click.option('--tb_logging', default=True, help='Whether to store Tensorboard logs', show_default=True)
@click.option('--max_iter', default=15, help='Maximum iteration for DO.', show_default=True)
@click.option('--observation_history', default=True, help='Whether to include the agent history')
@click.option('--env_params_compromise_actuation_prob', default=0.5, help='Actuation compromise probability')
@click.option('--env_params_compromise_observation_prob', default=0.5, help='Observation compromise probability')
@click.option('--env_params_power', default=0.3, help='Power of attacker')
@click.option('--rl_params_random_exploration', default=0.1, help='Exploration Fraction', show_default=True)
@click.option('--rl_params_gamma', default=0.90, help='Gamma', show_default=True)
@click.option('--policy_params_activation', default='tanh', help='Activation Function', show_default=True)
@click.option('--policy_params_layers', default='32, 32', help='MLP Network Layers', show_default=True)
def do_rc(env_id, prefix, index,
          training_steps,
          concurrent_runs,
          tb_logging,
          max_iter,
          observation_history,
          rl_params_random_exploration,
          rl_params_gamma,
          policy_params_activation,
          policy_params_layers,
          env_params_compromise_actuation_prob,
          env_params_compromise_observation_prob,
          env_params_power):
    params = {
        'env_id': env_id,
        'training_steps': training_steps,
        'concurrent_runs': concurrent_runs,
        'tb_logging': to_bool(tb_logging),
        'observation_history': observation_history,
        'env_params': {
            'compromise_actuation_prob': env_params_compromise_actuation_prob,
            'compromise_observation_prob': env_params_compromise_observation_prob,
            'power': env_params_power
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

    do_marl(prefix, index, params, max_iter, RCTrainer, find_zero_sum_mixed_ne_gambit)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    pass

cli.add_command(do_rc)
cli.add_command(do_mtd)

if __name__ == '__main__':
    print(os.environ['PATH'])
    cli()
