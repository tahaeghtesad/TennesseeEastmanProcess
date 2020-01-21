from trainer import *
from datetime import datetime
import logging

if __name__ == '__main__':
    rootLogger = logging.getLogger()
    logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # consoleHandler = logging.StreamHandler(sys.stdout)
    # consoleHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler('log.log', mode='w')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    rootLogger.info('Bootstrapping Defender...')
    bootstrap_defender()

    rootLogger.info('Bootstrapping Attacker...')
    bootstrap_attacker()

    rootLogger.info('Initializing the payoff table...')
    au, du = evaluate('attacker-0', 'defender-0')

    attacker_payoff_table = np.array([[au]])
    defender_payoff_table = np.array([[du]])

    save_tables()

    try:
        while True:
            rootLogger.info('Training new defender...')
            ae, _ = solve_equilibrium()
            rootLogger.info(f'Equilibrium: {ae}')
            train_defender(ae)

            rootLogger.info('Training new defender...')
            _, de = solve_equilibrium()
            rootLogger.info(f'Equilibrium: {de}')
            train_attacker(de)
    except KeyboardInterrupt:
        rootLogger.info(f'Interrupting...')