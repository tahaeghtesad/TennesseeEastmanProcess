import os
import subprocess


def run(params):
    print(f"Submitting... {['sbatch', 'run.srun.sh']}")
    subprocess.run(['sbatch', 'run.srun.sh'] + params)
    return 0


utenvs = ['0', '1', '2', '3']
settings = ['0', '1', '2', '3']
cas = ['0.00', '0.05', '0.2', '0.5']
downtimes = ['1', '4', '7', '10']
alphas = ['0.01', '0.05', '0.1']

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

    for alpha in alphas:
        for downtime in downtimes:
            for ca in cas:
                for setting in settings:
                    for utenv in utenvs:
                        run(['--prefix', 'runs',
                             '--index', index,
                             '--training_steps', '500000',
                             '--concurrent_runs', '8',
                             '--max_iter', '20',
                             '--env_params_alpha', alpha,
                             '--env_params_downtime', downtime,
                             '--env_params_ca', ca,
                             '--env_params_setting', setting,
                             '--env_params_utenv', utenv])
                        index += 1
