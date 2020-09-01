import os
import subprocess


def run(params):
    print(f"Submitting... {['sbatch', 'run.srun.sh']}")
    subprocess.run(['sbatch', 'run.srun.sh'] + params)
    return 0


if __name__ == '__main__':
    run(['rc',
         '--env_id', 'BRP',
         '--index', '101',
         '--training_steps', '500000',
         '--concurrent_runs', '1',
         '--tb_logging', 'True',
         '--max_iter', '14',
         '--observation_history', 'True',
         '--env_params_compromise_actuation_prob', '0.5',
         '--env_params_compromise_observation_prob', '0.5',
         '--env_params_noise', 'True',
         '--rl_params_random_exploration', '0.2',
         '--rl_params_gamma', '0.9',
         '--policy_params_activation', 'tanh',
         '--policy_params_layers', "64, 64, 64, 64",
         ])
