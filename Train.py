from trainer import Trainer
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

    trainer = Trainer(total_training_steps=1 * 1000, env='BRP', exploration=.01)

    cont = False

    if cont:
        rootLogger.info('Bootstrapping Defender...')
        trainer.bootstrap_defender()

        rootLogger.info('Bootstrapping Attacker...')
        trainer.bootstrap_attacker()

        rootLogger.info('Initializing the payoff table...')
        au, du = trainer.evaluate('attacker-0', 'defender-0')

        trainer.attacker_payoff_table = np.array([[au]])
        trainer.defender_payoff_table = np.array([[du]])

        iteration = 1
    else:
        rootLogger.info('Loading previous payoff tables... ')
        trainer.load_tables()
        iteration = min(trainer.defender_payoff_table.shape[0], trainer.defender_payoff_table.shape[1])
        trainer.defender_payoff_table = trainer.defender_payoff_table[:iteration+1, :iteration+1]
        trainer.attacker_payoff_table = trainer.attacker_payoff_table[:iteration+1, :iteration+1]

    rootLogger.info(f'Initial payoff table: {trainer.attacker_payoff_table, trainer.defender_payoff_table}')

    try:
        while True:
            # Training new defender
            rootLogger.info(f'Starting Iteration {iteration}')
            rootLogger.info('Training new defender...')
            ae, de = trainer.solve_equilibrium()
            rootLogger.info(f'Payoff_tables: {trainer.attacker_payoff_table, trainer.defender_payoff_table}')
            rootLogger.info(f'Equilibrium: {ae, de}')
            defender = trainer.train_defender(ae, de)

            au, du = trainer.get_defender_payoff(defender)
            ndu = trainer.evaluate_new_defender_mixed_attacker(du)  # New_Defender_Mixed_Utility
            _, mdu = trainer.get_mixed_payoff()  # Mixed_Defender_Utility
            rootLogger.info(f'Mixed Defender Utility: {mdu}')
            rootLogger.info(f'New Defender Utility Vs. Mixed Attacker Equilibrium: {ndu}')
            rootLogger.info(f'Defender Improvement: {(1 - (ndu / mdu)) * 100:.2f}%')
            trainer.update_defender_payoff_table(au, du)

            # Training new attacker
            rootLogger.info('Training new attacker...')
            ae, de = trainer.solve_equilibrium()
            rootLogger.info(f'Payoff_tables: {trainer.attacker_payoff_table, trainer.defender_payoff_table}')
            rootLogger.info(f'Equilibrium: {ae, de}')
            attacker = trainer.train_attacker(ae, de)

            au, du = trainer.get_attacker_payoff(attacker)
            nau = trainer.evaluate_new_attacker_mixed_defender(au)
            mau, _ = trainer.get_mixed_payoff()
            rootLogger.info(f'Mixed Attacker Utility: {mau}')
            rootLogger.info(f'New Attacker Utility Vs. Mixed Defender Equilibrium: {nau}')
            rootLogger.info(f'Attacker Improvement: {(1 - (nau / mau)) * 100:.2f}%')
            trainer.update_attacker_payoff_table(au, du)

            iteration += 1

    except KeyboardInterrupt:
        rootLogger.info(f'Interrupting...')
