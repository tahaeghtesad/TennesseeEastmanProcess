import subprocess
import sys


def create_run(
        group,
        parallelization,
        env='BRP',
        max_iter=1,
        training_steps=300_000,
        tb_logging=False,
        action_noise_sigma=0.01,
        compromise_actuation_prob=0.0,
        compromise_observation_prob=0.0,
        power=0.3,
        noise_sigma=0.05,
        history_length=1,
        include_compromise=False,
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

        ## Baseline
        for r in range(repeat):
            runs.append(create_run(
                'baseline', env=env, parallelization=parallelization
            ))
            count += 1
    
        ## Changing The Agent Conf
    
        # Epsilon
        for e in [0.0, 0.01, 0.05, 0.1, 0.2]:
            for r in range(repeat):
                runs.append(create_run(
                    'epsilon', epsilon=e, env=env, parallelization=parallelization
                ))
                count += 1
    
        # Gamma
        for g in [0.99, 0.9, 0.5, 0.1]:
            for r in range(repeat):
                runs.append(create_run(
                    'gamma', gamma=g, env=env, parallelization=parallelization
                ))
                count += 1
    
        # act_fun
        for a in ['tanh', 'sigmoid', 'relu', 'elu']:
            for r in range(repeat):
                runs.append(create_run(
                    'act_fun', act_fun=a, env=env, parallelization=parallelization
                ))
                count += 1
    
        # layers
        for l in ['25, 25, 25, 25, 25', '25, 25, 25', '25, 25', '128, 64', '128, 64, 32']:
            for r in range(repeat):
                runs.append(create_run(
                    'layers', layers=l, env=env, parallelization=parallelization
                ))
                count += 1
    
        # action noise
        for an in [0.01, 0.05, 0.1, 0.5]:
            for r in range(repeat):
                runs.append(create_run(
                    'action_noise', action_noise_sigma=an, env=env, parallelization=parallelization
                ))
                count += 1
    
        # buffer size
        for b in [500, 1000, 5000, 10000, 50000]:
            for r in range(repeat):
                runs.append(create_run(
                    'buffer_size', buffer_size=b, env=env, parallelization=parallelization
                ))
                count += 1
    
        # batch size
        for s in [16, 32, 64, 128, 256, 512]:
            for r in range(repeat):
                runs.append(create_run(
                    'batch_size', batch_size=s, env=env, parallelization=parallelization
                ))
                count += 1
    
        ## Changing Env configuration
    
        # t_epoch
        for te in [5, 10, 50, 100, 200, 300, 400, 500]:
            for r in range(repeat):
                runs.append(create_run(
                    't_epoch', t_epoch=te, env=env, parallelization=parallelization
                ))
                count += 1
    
        # noise sigma
        for ns in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for r in range(repeat):
                runs.append(create_run(
                    'env_noise', noise_sigma=ns, env=env, parallelization=parallelization
                ))
                count += 1
    
        # history_length
        for hl in [1, 2, 4, 8, 16, 32, 64]:
            for r in range(repeat):
                runs.append(create_run(
                    'history_length', history_length=hl, env=env, parallelization=parallelization
                ))
                count += 1
    
        # start from set
        for r in range(repeat):
            runs.append(create_run(
                'start_set', test_env=True, env=env, parallelization=parallelization
            ))
            count += 1

    print(f'Total {count} jobs were created.')
    return runs


default_conf = [
    '-J TEP',
    '-t 6:00:00',
    '--mem 8GB',
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

python run.rc.py $SLURM_ARRAY_TASK_ID

mkdir -p runs/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID
cp -r $TMPDIR/data/* runs/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID/
''')


if __name__ == '__main__':
    parallelization = 2
    concurrent_runs = 100
    repeat = 10
    runs = generate_runs(repeat, parallelization)
    assert len(runs) < 1001, 'Too many runs to schedule'
    if len(sys.argv) == 1:
        target = 'dynamic.run.srun.sh'
        write_config(target, default_conf, runs, parallelization, concurrent_runs)
        print(f'running {["sbatch", target]}')
        subprocess.run(['sbatch', target])
    if len(sys.argv) == 2:
        run_conf = int(sys.argv[1])
        print(f"running {['python', 'cli.py'] + runs[run_conf - 1]}")
        subprocess.run(['python', 'cli.py'] + runs[run_conf - 1])
