import os
import subprocess


def run(params):
    print(f"Submitting... {['sbatch', 'run.srun.sh']}")
    subprocess.run(['sbatch', 'run.srun.sh'] + params)
    return 0


if __name__ == '__main__':
    confs = [
        ['BRP', '0.1', 'True', '1'],
        ['BRP', '0.05', 'True', '1'],
        ['BRP', '0.01', 'True', '1'],
        ['BRP', '0.005', 'True', '1'],

        ['BRP', '0.1', 'False', '1'],
        ['BRP', '0.05', 'False', '1'],
        ['BRP', '0.01', 'False', '1'],
        ['BRP', '0.005', 'False', '1'],

        ['BRP', '0.1', 'True', '2'],
        ['BRP', '0.05', 'True', '2'],
        ['BRP', '0.01', 'True', '2'],
        ['BRP', '0.005', 'True', '2'],

        ['BRP', '0.1', 'False', '2'],
        ['BRP', '0.05', 'False', '2'],
        ['BRP', '0.01', 'False', '2'],
        ['BRP', '0.005', 'False', '2'],

        ['BRP', '0.1', 'True', '4'],
        ['BRP', '0.05', 'True', '4'],
        ['BRP', '0.01', 'True', '4'],
        ['BRP', '0.005', 'True', '4'],

        ['BRP', '0.1', 'False', '4'],
        ['BRP', '0.05', 'False', '4'],
        ['BRP', '0.01', 'False', '4'],
        ['BRP', '0.005', 'False', '4'],

        ['BRP', '0.1', 'True', '16'],
        ['BRP', '0.05', 'True', '16'],
        ['BRP', '0.01', 'True', '16'],
        ['BRP', '0.005', 'True', '16'],

        ['BRP', '0.1', 'False', '16'],
        ['BRP', '0.05', 'False', '16'],
        ['BRP', '0.01', 'False', '16'],
        ['BRP', '0.005', 'False', '16'],
    ]

    index = 2000
    count = 0

    for conf in confs:
        for _ in range(10):
            run(['rc',
                 '--env_id', conf[0],
                 '--index', f'{index}',
                 '--max_iter', '1',
                 '--training_params_training_steps', '300_000',
                 '--training_params_concurrent_runs', '4',
                 '--training_params_tb_logging', 'False',
                 '--training_params_action_noise_sigma', conf[1],
                 '--env_params_compromise_actuation_prob', '0.0',
                 '--env_params_compromise_observation_prob', '0.5',
                 '--env_params_noise', conf[2],
                 '--env_params_history_length', conf[3],
                 '--env_params_include_compromise', 'True',
                 '--rl_params_gamma', '0.90',
                 '--rl_params_random_exploration', '0.0',
                 '--policy_params_activation', 'tanh',
                 '--policy_params_layers', '25,25,25,25'
                 ])
            index += 1
            count += 1

    print(f'Total {count} jobs were executed.')
