from trainer import Trainer
from datetime import datetime
import logging
import numpy as np
import sys

if __name__ == '__main__':
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler('log.log', mode='w')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    trainer = Trainer()

    rootLogger.info('Bootstrapping Defender...')
    trainer.bootstrap_defender()

    rootLogger.info('Bootstrapping Attacker...')
    trainer.bootstrap_attacker()

    rootLogger.info('Initializing the payoff table...')
    au, du = trainer.evaluate('attacker-0', 'defender-0')

    trainer.attacker_payoff_table = np.array([[au]])
    trainer.defender_payoff_table = np.array([[du]])


    try:
        while True:
            rootLogger.info('Training new defender...')
            ae, de = trainer.solve_equilibrium()
            rootLogger.info(f'Payoff_tables: {trainer.attacker_payoff_table, trainer.defender_payoff_table}')
            rootLogger.info(f'Equilibrium: {ae, de}')
            trainer.train_defender(ae, de)

            rootLogger.info('Training new attacker...')
            ae, de = trainer.solve_equilibrium()
            rootLogger.info(f'Payoff_tables: {trainer.attacker_payoff_table, trainer.defender_payoff_table}')
            rootLogger.info(f'Equilibrium: {ae, de}')
            trainer.train_attacker(ae, de)
    except KeyboardInterrupt:
        rootLogger.info(f'Interrupting...')
