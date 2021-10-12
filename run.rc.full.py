import subprocess
import sys


def create_run(
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


def generate_runs(repeat, parallelization):
    count = 0

    runs = []

    for env in ['BRP', 'TT']:

        for _ in range(repeat):
            runs.append(create_run(
                group='do_baseline',
                parallelization=parallelization,
                env=env,
                noise_sigma=0.05,
                action_noise_sigma=0.005,
                test_env=False,
                epsilon=0.01,
                max_iter=8,
                history_length=8
            ))
            count += 1

        for _ in range(repeat):
            for hl in [1, 2, 4, 8, 16]:
                runs.append(create_run(
                    group='do_memory',
                    parallelization=parallelization,
                    env=env,
                    noise_sigma=0.05,
                    action_noise_sigma=0.005,
                    test_env=False,
                    epsilon=0.01,
                    max_iter=8,
                    history_length=hl
                ))
                count += 1

        for _ in range(repeat):
            runs.append(create_run(
                group='do_baseline_random_start',
                parallelization=parallelization,
                env=env,
                noise_sigma=0.05,
                action_noise_sigma=0.005,
                test_env=False,
                epsilon=0.01,
                max_iter=8,
            ))
            count += 1

        for ca in [0.2, 0.4, 0.8, 1.0]:
            for _ in range(repeat):
                runs.append(create_run(
                    group='do_actuation_only',
                    parallelization=parallelization,
                    env=env,
                    noise_sigma=0.0,
                    action_noise_sigma=0.005,
                    test_env=False,
                    epsilon=0.01,
                    max_iter=8,
                    compromise_actuation_prob=ca,
                    compromise_observation_prob=0.0,
                ))
                count += 1

        for co in [0.2, 0.4, 0.8, 1.0]:
            for _ in range(repeat):
                runs.append(create_run(
                    group='do_observation_only',
                    parallelization=parallelization,
                    env=env,
                    noise_sigma=0.0,
                    action_noise_sigma=0.005,
                    test_env=False,
                    epsilon=0.01,
                    max_iter=8,
                    compromise_observation_prob=co,
                    compromise_actuation_prob=0.0,
                ))
                count += 1

        for p in [0.1, 0.3, 0.5, 0.75, 1]:
            for _ in range(repeat):
                runs.append(create_run(
                    group='do_power',
                    parallelization=parallelization,
                    env=env,
                    noise_sigma=0.05,
                    action_noise_sigma=0.005,
                    test_env=False,
                    epsilon=0.01,
                    max_iter=8,
                    power=p
                ))
                count += 1

        for ss in [True]:
            for _ in range(repeat):
                runs.append(create_run(
                    group='do_start_set',
                    parallelization=parallelization,
                    env=env,
                    noise_sigma=0.05,
                    action_noise_sigma=0.005,
                    test_env=ss,
                    epsilon=0.01,
                    max_iter=8
                ))
                count += 1

        for en in [0.0005, 0.005, 0.05, 0.5]:
            for _ in range(repeat):
                runs.append(create_run(
                    group='do_env_noise',
                    parallelization=parallelization,
                    env=env,
                    noise_sigma=en,
                    action_noise_sigma=0.005,
                    test_env=False,
                    epsilon=0.01,
                    max_iter=8
                ))
                count += 1

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

python run.rc.full.py $SLURM_ARRAY_TASK_ID $SLURM_JOB_ID

mkdir -p runs/$SLURM_JOB_ID
cp -r $TMPDIR/data/* runs/$SLURM_JOB_ID/
''')


if __name__ == '__main__':
    parallelization = 1
    concurrent_runs = 50
    repeat = 6
    runs = generate_runs(repeat, parallelization)
    assert len(runs) < 1001, 'Too many runs to schedule'
    if len(sys.argv) == 1:
        target = 'dynamic.full.run.srun.sh'
        write_config(target, default_conf, runs, parallelization, concurrent_runs)
        print(f'running {["sbatch", target]}')
        subprocess.run(['sbatch', target])
    if len(sys.argv) == 3:
        run_conf = int(sys.argv[1])
        sabine_id = int(sys.argv[2])
        print(f"running {['python', 'cli.py'] + runs[run_conf - 1] + ['--sabine_id', f'{sabine_id}']}")
        subprocess.run(['python', 'cli.py'] + runs[run_conf - 1] + ['--sabine_id', f'{sabine_id}'])
