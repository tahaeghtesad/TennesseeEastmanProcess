import os
import subprocess


def run(params):
    print(f"Submitting... {['sbatch', 'run.srun.sh']}")
    subprocess.run(['sbatch', 'run.srun.sh'] + params)
    return 0


if __name__ == '__main__':

    index = 1100

    runs = [
        [0.05, 7, 0.20, 0, 2],
        [0.05, 7, 0.05, 0, 2],
        [0.05, 7, 0.10, 0, 2],
        [0.05, 7, 0.05, 0, 2],
        [0.05, 7, 0.1, 0, 2],
        [0.05, 7, 0.2, 0, 2],

        [0.05, 7, 0.05, 0, 0],
        [0.05, 7, 0.05, 1, 0],
        [0.05, 7, 0.05, 2, 0],
        [0.05, 7, 0.1, 0, 0],
        [0.05, 7, 0.1, 1, 0],
        [0.05, 7, 0.1, 2, 0],
        [0.05, 7, 0.2, 0, 0],
        [0.05, 3, 0.1, 0, 0],

        [0.05, 7, 0.1, 0, 1],
        [0.05, 7, 0.05, 0, 1],
        [0.05, 7, 0.2, 0, 1],
        [0.05, 3, 0.1, 0, 1],

        [0.05, 7, 0.1, 0, 3],
        [0.05, 7, 0.05, 0, 3],
        [0.05, 7, 0.2, 0, 3],
    ]

    for config in runs:
        for _ in range(8):
            run(['--prefix', 'runs',
                 '--index', str(index),
                 '--training_steps', '400000',
                 '--concurrent_runs', '1',
                 '--tb_logging', 'False',
                 '--max_iter', '15',
                 '--max_iter', '10',
                 '--include_heuristics', 'False',
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

