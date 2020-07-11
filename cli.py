import logging
import sys

import click
import numpy as np

from trainer import MTDTrainer
import tensorflow as tf
from util.nash_helpers import find_general_sum_mixed_ne
import json
import os


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


def do_mtd(prefix, index, params, max_iter):
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
        params_back = params.copy()
        params_back['policy_params']['activation'] = 'relu'
        json.dump(params_back, fd)

    trainer = MTDTrainer(
        f'{prefix}/{index}',
        **params
    )

    logger.info('Initializing Heuristic Strategies...')
    attacker_ms, defender_ms = trainer.initialize_strategies()
    attacker_strategy, defender_strategy = find_general_sum_mixed_ne(trainer.attacker_payoff_table, trainer.defender_payoff_table)
    logging.info(f'Attacker MSNE: {attacker_strategy}')
    logging.info(f'Defender MSNE: {defender_strategy}')
    attacker_ms.update_probabilities(attacker_strategy)
    defender_ms.update_probabilities(defender_strategy)

    attacker_iteration = len(attacker_ms.policies)
    defender_iteration = len(defender_ms.policies)

    logger.info(f'Attacker Heuristics: {attacker_iteration}')
    logger.info(f'Defender Heuristics: {defender_iteration}')

    while attacker_iteration < max_iter or defender_iteration < max_iter:
        #Train Attacker
        logger.info(f'Training Attacker {attacker_iteration}')
        _, defender_strategy = find_general_sum_mixed_ne(trainer.attacker_payoff_table,
                                                         trainer.defender_payoff_table)
        logging.info(f'Defender MSNE: {defender_strategy}')
        defender_ms.update_probabilities(defender_strategy)
        attacker_policy = trainer.train_attacker_parallel(defender_ms, attacker_iteration)
        attacker_ms.add_policy(attacker_policy)
        payoffs = [trainer.get_payoff(attacker_policy, defender_policy) for defender_policy in defender_ms.policies]
        trainer.update_attacker_payoff_table(np.array([au for (au, du) in payoffs]), np.array([du for (au, du) in payoffs]))
        attacker_iteration += 1
        logging.info(f'New Attacker vs MSNE Defender Payoff: {trainer.get_payoff(attacker_policy, defender_ms)}')

        #Train Defender
        logger.info(f'Training Defender {defender_iteration}')
        attacker_strategy, _ = find_general_sum_mixed_ne(trainer.attacker_payoff_table,
                                                         trainer.defender_payoff_table)
        logging.info(f'Attacker MSNE: {attacker_strategy}')
        attacker_ms.update_probabilities(attacker_strategy)
        defender_policy = trainer.train_defender_parallel(attacker_ms, defender_iteration)
        defender_ms.add_policy(defender_policy)
        payoffs = [trainer.get_payoff(attacker_policy, defender_policy) for attacker_policy in attacker_ms.policies]
        trainer.update_defender_payoff_table(np.array([au for (au, du) in payoffs]), np.array([du for (au, du) in payoffs]))
        defender_iteration += 1
        logging.info(f'MSNE Attacker vs New Defender Payoff: {trainer.get_payoff(attacker_ms, defender_policy)}')


@click.command()
@click.option('--prefix', default='runs', help='Prefix folder of run results', show_default=True)
@click.option('--index', help='Index for this run', required=True)
@click.option('--training_steps', default=500 * 1000, help='Number of training steps in each iteration of DO.', show_default=True)
@click.option('--concurrent_runs', default=4, help='Number of concurrent runs', show_default=True)
@click.option('--max_iter', default=15, help='Maximum iteration for DO.', show_default=True)
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
@click.option('--policy_params_dueling', default=0, help='Dueling MLP Network', show_default=True)
@click.option('--policy_params_normalization', default=True, help='Layer Normalization', show_default=True)
def run_mtd(prefix, index,
            training_steps,
            concurrent_runs,
            max_iter,
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
    # TODO policy_params has not been used.
    params = {
        'training_steps': training_steps,
        'concurrent_runs': concurrent_runs,
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
            'double_q': rl_params_double_q,
            'prioritized_replay': rl_params_prioritized_replay
        },
        'policy_params': {
            'activation': getattr(tf.nn, policy_params_activation),
            'layers': [int(l) for l in policy_params_layers.split(',')],
            'dueling': policy_params_dueling,
            'normalization': policy_params_normalization
        }
    }
    do_mtd(prefix, index, params, max_iter)


if __name__ == '__main__':
    run_mtd()
