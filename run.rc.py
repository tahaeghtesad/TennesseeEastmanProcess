import os
import subprocess


def run(params):
    # print(f"Submitting... {['sbatch', 'run.srun.sh']}")
    # print(' '.join(['sbatch', 'run.srun.sh'] + params))
    # subprocess.run(['sbatch', 'run.srun.sh'] + params)
    return 0


if __name__ == '__main__':

    index = 6000
    count = 0
    for ts in ['300_000']:
        for s in ['0.01', '0.0']:
            for n in ['True', 'False']:
                for te in ['True', 'False']:
                    for hl in ['1', '4', '8']:
                        for g in ['0.9', '.5']:
                            for e in ['.00', '0.1']:
                                for a in ['tanh', 'elu']:
                                    for l in [
                                              '25, 25, 25, 25',
                                              '25, 25, 25, 25, 25',
                                              '25, 25, 25',
                                              '25, 25',
                                              '25'
                                              ]:
                                        for _ in range(4):
                                            run(['rc',
                                                 '--env_id', 'BRP',
                                                 '--index', f'{index}',
                                                 '--max_iter', '1',
                                                 '--training_params_training_steps', ts,
                                                 '--training_params_concurrent_runs', '2',
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
                                                 ])
                                            index += 1
                                            count += 1

    print(f'Total {count} jobs were executed.')
