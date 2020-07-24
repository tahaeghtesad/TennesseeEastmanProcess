import os
import subprocess


def run(params):
    print(f"Submitting... {['sbatch', 'run.srun.sh']}")
    subprocess.run(['sbatch', 'run.srun.sh'] + params)
    return 0

params = [
    '--prefix',
    '--index',
    '--training_steps',
    '--concurrent_runs',
    '--max_iter',
    '--env_params_m',
    '--env_params_utenv',
    '--env_params_setting',
    '--env_params_ca',
    '--env_params_downtime',
    '--env_params_alpha',
    '--env_params_probe_detection',
    '--rl_params_exploration_fraction',
    '--rl_params_exploration_final_eps',
    '--rl_params_gamma',
    '--rl_params_double_q',
    '--rl_params_prioritized_replay',
    '--policy_params_activation',
    '--policy_params_layers',
    '--policy_params_dueling',
    '--policy_params_normalization'
]

if __name__ == '__main__':

    index = 1

    runs = [
        [0.05, 7, 0.05, 0, 0],
        [0.05, 7, 0.05, 1, 0],
        [0.05, 7, 0.05, 2, 0],
        [0.05, 7, 0.05, 0, 1],
        [0.05, 7, 0.05, 0, 2],
        [0.05, 7, 0.05, 0, 3],
        [0.05, 7, 0.1, 0, 0],
        [0.05, 7, 0.1, 1, 0],
        [0.05, 7, 0.1, 2, 0],
        [0.05, 7, 0.1, 0, 1],
        [0.05, 7, 0.1, 0, 2],
        [0.05, 7, 0.1, 0, 3],

        [0.1, 7, 0.1, 0, 0],
        [0.1, 7, 0.1, 0, 1],
        [0.1, 7, 0.1, 0, 2],
        [0.1, 7, 0.1, 0, 3],

        [0.05, 3, 0.1, 0, 0],
        [0.05, 3, 0.1, 0, 1],
        [0.05, 3, 0.1, 0, 2],
        [0.05, 3, 0.1, 0, 3],
    ]

    for config in runs:
        run(['--prefix', 'runs',
             '--index', str(index),
             '--training_steps', '400000',
             '--concurrent_runs', '1',
             '--max_iter', '20',
             '--env_params_alpha', str(config[0]),
             '--env_params_downtime', str(config[1]),
             '--env_params_ca', str(config[2]),
             '--env_params_setting', str(config[3]),
             '--env_params_utenv', str(config[4]),
             '--rl_params_double_q', 'False',
             '--policy_params_dueling', 'True',
             '--rl_params_prioritized_replay', 'False',
             '--policy_params_normalization', 'True',
             '--policy_params_layers', '32, 32'])
        index += 1
