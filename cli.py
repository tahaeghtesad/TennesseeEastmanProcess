import logging
import sys

import numpy as np

from trainer import MTDTrainer
from tensorflow.nn import relu
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


def do_mtd(prefix, index):
    if not os.path.exists(f'{prefix}'):
        os.makedirs(f'{prefix}')
    if not os.path.exists(f'{prefix}/{index}'):
        os.makedirs(f'{prefix}/{index}')
        os.makedirs(f'{prefix}/{index}/tb_logs')
        os.makedirs(f'{prefix}/{index}/params')

    logger = init_logger(prefix, index)

    params = {
        'training_steps': 1 * 1000,
        'env_params': {
            'm': 10,
            'utenv': 0,
            'setting': 0,
            'ca': .05,
            'downtime': 7,
            'alpha': 0.05,
            'probe_detection': 0
        },
        'rl_params': {
            'exploration_fraction': .2,
            'exploration_final_eps': 0.02,
            'gamma': .98,
            'double_q': False,
            'prioritized_replay': False
        },
        'policy_params': {
            'activation': relu,
            'layers': [32] * 2,
            'dueling': False,
            'normalization': True
        }
    }

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

    while attacker_iteration < 15 or defender_iteration < 15:
        #Train Attacker
        logger.info(f'Training Attacker {attacker_iteration}')
        _, defender_strategy = find_general_sum_mixed_ne(trainer.attacker_payoff_table,
                                                         trainer.defender_payoff_table)
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
        attacker_ms.update_probabilities(attacker_strategy)
        defender_policy = trainer.train_defender_parallel(attacker_ms, defender_iteration)
        defender_ms.add_policy(defender_policy)
        payoffs = [trainer.get_payoff(attacker_policy, defender_policy) for attacker_policy in attacker_ms.policies]
        trainer.update_defender_payoff_table(np.array([au for (au, du) in payoffs]), np.array([du for (au, du) in payoffs]))
        defender_iteration += 1
        logging.info(f'MSNE Attacker vs New Defender Payoff: {trainer.get_payoff(attacker_ms, defender_policy)}')


if __name__ == '__main__':
    do_mtd('runs', 1)
