import os
import subprocess


def run(params):
    print(f"Submitting... {['sbatch', 'run.srun.sh']}")
    subprocess.run(['sbatch', 'run.rnslab.sh'] + params)
    return 0


if __name__ == '__main__':
    confs = [
        ['BRP', 'False', 'False', '64, 64, 64, 64', '500000'],
        ['BRP', 'True', 'False', '64, 64, 64, 64', '1000000'],
        ['BRP', 'True', 'True', '64, 64, 64, 64', '1000000'],
        ['TT', 'False', 'False', '64, 64, 64, 64', '1000000'],
        ['TT', 'True', 'False', '64, 64, 64, 64', '2000000'],
        ['TT', 'True', 'True', '64, 64, 64, 64', '1000000'],
        ['BRP', 'True', 'False', '64, 64, 64, 64, 64', '2000000'],
        ['BRP', 'True', 'True', '64, 64, 64, 64, 64', '2000000'],
        ['TT', 'True', 'False', '128, 128, 128, 128, 128', '2000000'],
        ['TT', 'True', 'True', '128, 128, 128, 128, 128', '2000000'],
        ['BRP', 'True', 'False', '64, 64, 64, 64, 64', '2000000'],
        ['TT', 'True', 'False', '64, 64, 64, 64, 64', '2000000'],
        ['TT', 'True', 'False', '64, 64', '2000000'],
    ]
    index = 101

    for conf in confs:
        run(['rc',
             '--env_id', conf[0],
             '--index', str(index),
             '--max_iter', '1',
             '--training_params_training_steps', conf[4],
             '--training_params_concurrent_runs', '8',
             '--training_params_tb_logging', 'True',
             '--training_params_attacker_history', 'False',
             '--training_params_defender_history', conf[1],
             '--training_params_attacker_limited_history', 'False',
             '--training_params_defender_limited_history', conf[2],
             '--env_params_compromise_actuation_prob', '0.0',
             '--env_params_compromise_observation_prob', '0.5',
             '--env_params_noise', 'True',
             '--rl_params_random_exploration', '0.1',
             '--rl_params_gamma', '0.9',
             '--policy_params_activation', 'tanh',
             '--policy_params_layers', conf[3],
             ])
        index += 1
