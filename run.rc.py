import os
import subprocess


def run(params):
    print(f"Submitting... {['sbatch', 'run.srun.sh']}")
    subprocess.run(['sbatch', 'run.srun.sh'] + params)
    return 0


if __name__ == '__main__':
    confs = [
        [False, False],
        [True, False],
        [True, True]
    ]
    index = 101

    for conf in confs:
        run(['rc',
             '--env_id', 'BRP',
             '--index', str(index),
             '--max_iter', '1',
             '--training_params_training_steps', '500000',
             '--training_params_concurrent_runs', '1',
             '--training_params_tb_logging', 'True',
             '--training_params_attacker_history', 'False',
             '--training_params_defender_history', str(confs[0]),
             '--training_params_attacker_limited_history', 'False',
             '--training_params_defender_limited_history', str(confs[1]),
             '--env_params_compromise_actuation_prob', '0.0',
             '--env_params_compromise_observation_prob', '0.5',
             '--env_params_noise', 'True',
             '--rl_params_random_exploration', '0.1',
             '--rl_params_gamma', '0.9',
             '--policy_params_activation', 'tanh',
             '--policy_params_layers', "64, 64, 64, 64",
             ])
        index += 1
