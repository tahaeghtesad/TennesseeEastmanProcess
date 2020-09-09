import os
import subprocess


def run(params):
    print(f"Submitting... {['sbatch', 'run.srun.sh']}")
    subprocess.run(['sbatch', 'run.srun.sh'] + params)
    return 0


if __name__ == '__main__':

    index = 45

    runs = [
        [0.05, 7, 0.20, 0, 2],
        [0.05, 7, 0.20, 0, 2],
        [0.05, 7, 0.20, 0, 2],
    ]

    for config in runs:
        run(['mtd',
             '--prefix', 'runs',
             '--index', str(index),
             '--training_params_training_steps', '400000',
             '--training_params_concurrent_runs', '1',
             '--training_params_tb_logging', 'False',
             '--max_iter', '10',
             '--training_params_include_heuristics', 'False',
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
