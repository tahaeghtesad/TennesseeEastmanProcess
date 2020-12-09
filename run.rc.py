import subprocess
import sys


def generate_runs(index, parallelization):
    count = 0

    runs = []

    for ts in ['300_000']:
        for s in ['0.01', '0.0']:
            for n in ['True', 'False']:
                for te in ['True', 'False']:
                    for hl in ['1', '4', '8']:
                        for g in ['0.9', '.5']:
                            for e in ['.00', '0.1', '0.5']:
                                for a in ['tanh']:
                                    for l in [
                                        '25, 25, 25, 25'
                                    ]:
                                        for _ in range(4):
                                            runs += [['rc',
                                                      '--env_id', 'BRP',
                                                      '--index', f'{index}',
                                                      '--max_iter', '1',
                                                      '--training_params_training_steps', ts,
                                                      '--training_params_concurrent_runs', str(parallelization),
                                                      '--training_params_tb_logging', 'False',
                                                      '--training_params_action_noise_sigma', s,
                                                      '--env_params_compromise_actuation_prob', '0',
                                                      '--env_params_compromise_observation_prob', '0',
                                                      '--env_params_noise', n,
                                                      '--env_params_history_length', hl,
                                                      '--env_params_include_compromise', 'True',
                                                      '--env_params_test_env', te,
                                                      '--rl_params_gamma', g,
                                                      '--rl_params_random_exploration', e,
                                                      '--policy_params_activation', a,
                                                      '--policy_params_layers', l
                                                      ]]
                                            index += 1
                                            count += 1

    print(f'Total {count} jobs were created.')
    return runs


default_conf = [
    '-J TEP',
    '-t 2:00:00',
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
''')


if __name__ == '__main__':
    parallelization = 1
    start_index = 6000
    concurrent_runs = 50
    runs = generate_runs(start_index, parallelization)
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
