import subprocess
import sys


def create_run(
        index,
        group,
        parallelization,
        env='BRP',
        max_iter=8,
        training_steps=300_000,
        tb_logging=False,
        action_noise_sigma=0.01,
        compromise_actuation_prob=0.5,
        compromise_observation_prob=0.5,
        power=0.3,
        noise_sigma=0.05,
        history_length=8,
        include_compromise=True,
        test_env=False,
        t_epoch=200,
        gamma=0.90,
        epsilon=0.05,
        act_fun='tanh',
        layers='25, 25, 25, 25',
        nb_train=30,
        buffer_size=10000,
        batch_size=128,
        nb_rollout=100,
):
    return ['rc',
            '--env_id', env,
            '--index', f'{index}',
            '--max_iter', f'{max_iter}',
            '--group', group,
            '--training_params_training_steps', f'{training_steps}',
            '--training_params_concurrent_runs', f'{parallelization}',
            '--training_params_tb_logging', f'{tb_logging}',
            '--training_params_action_noise_sigma', f'{action_noise_sigma}',
            '--env_params_compromise_actuation_prob', f'{compromise_actuation_prob}',
            '--env_params_compromise_observation_prob', f'{compromise_observation_prob}',
            '--env_params_power', f'{power}',
            '--env_params_noise_sigma', f'{noise_sigma}',
            '--env_params_history_length', f'{history_length}',
            '--env_params_include_compromise', f'{include_compromise}',
            '--env_params_test_env', f'{test_env}',
            '--env_params_t_epoch', f'{t_epoch}',
            '--rl_params_gamma', f'{gamma}',
            '--rl_params_random_exploration', f'{epsilon}',
            '--rl_params_nb_train', f'{nb_train}',
            '--rl_params_nb_rollout', f'{nb_rollout}',
            '--rl_params_batch_size', f'{batch_size}',
            '--rl_params_buffer_size', f'{buffer_size}',
            '--policy_params_activation', f'{act_fun}',
            '--policy_params_layers', f'{layers}'
            ]


def generate_runs(repeat, index, parallelization):
    count = 0

    runs = []

    runs.append(create_run(
        index,
        'do_test',
        parallelization=parallelization,
        max_iter=20,
        epsilon=0.01,
        noise_sigma=0.0,
        action_noise_sigma=0.005,
    ))

    print(f'Total {count} jobs were created.')
    return runs


default_conf = [
    '-J TEP',
    '-t 48:00:00',
    '--mem 16GB',
    '-A laszka'
]


def write_config(target, config, runs, paralelization, concurrent_runs):
    with open(target, 'w') as tf:
        tf.write('#!/bin/bash\n')

        for conf in config:
            tf.write(f'#SBATCH {conf}\n')

        tf.write(f'#SBATCH -N 1 -n {paralelization}\n\n')
        tf.write(f'#SBATCH --array=1-{len(runs)}%{concurrent_runs}\n\n')

        tf.write('''source /home/${USER}/.bashrc
conda activate tep-cpu
cd /project/laszka/TennesseeEastmanProcess/
export PATH=$PWD/gambit-project/:$PATH

python run.rc.full.py $SLURM_ARRAY_TASK_ID
''')


if __name__ == '__main__':
    parallelization = 8
    start_index = 18000
    concurrent_runs = 50
    repeat = 10
    runs = generate_runs(repeat, start_index, parallelization)
    assert len(runs) < 1001, 'Too many runs to schedule'
    if len(sys.argv) == 1:
        target = 'dynamic.full.run.srun.sh'
        write_config(target, default_conf, runs, parallelization, concurrent_runs)
        print(f'running {["sbatch", target]}')
        subprocess.run(['sbatch', target])
    if len(sys.argv) == 2:
        run_conf = int(sys.argv[1])
        print(f"running {['python', 'cli.py'] + runs[run_conf - 1]}")
        subprocess.run(['python', 'cli.py'] + runs[run_conf - 1])
