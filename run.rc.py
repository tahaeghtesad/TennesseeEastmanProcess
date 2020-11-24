import os
import subprocess


def run(params):
    print(f"Submitting... {['sbatch', 'run.srun.sh']}")
    subprocess.run(['sbatch', 'run.srun.sh'] + params)
    return 0


if __name__ == '__main__':
    confs = [
        ['BRP', 'True', '0.1', '128, 64'],
        ['BRP', 'True', '0.1', '256, 128'],
        ['BRP', 'True', '0.1', '128, 128'],
        ['BRP', 'True', '0.1', '256, 128, 64'],

        ['BRP', 'False', '0.1', '128, 64'],
        ['BRP', 'False', '0.1', '256, 128'],
        ['BRP', 'False', '0.1', '128, 128'],
        ['BRP', 'False', '0.1', '256, 128, 64'],

        ['BRP', 'True', '0.0', '128, 64'],
        ['BRP', 'True', '0.0', '256, 128'],
        ['BRP', 'True', '0.0', '128, 128'],
        ['BRP', 'True', '0.0', '256, 128, 64'],

        ['BRP', 'False', '0.0', '128, 64'],
        ['BRP', 'False', '0.0', '256, 128'],
        ['BRP', 'False', '0.0', '128, 128'],
        ['BRP', 'False', '0.0', '256, 128, 64'],
    ]
    index = 1500

    for conf in confs:
        for _ in range(10):
            run(['rc',
                 '--env_id', conf[0],
                 '--index', str(index),
                 '--max_iter', '1',
                 '--training_params_training_steps', '500_000',
                 '--training_params_concurrent_runs', '1',
                 '--training_params_tb_logging', 'True',
                 '--training_params_attacker_history', 'False',
                 '--training_params_defender_history', conf[1],
                 '--training_params_attacker_limited_history', 'False',
                 '--training_params_defender_limited_history', 'False',
                 '--env_params_compromise_actuation_prob', '0.0',
                 '--env_params_compromise_observation_prob', '0.5',
                 '--env_params_noise', 'True',
                 '--rl_params_random_exploration', conf[2],
                 '--rl_params_gamma', '0.9',
                 '--policy_params_activation', 'tanh',
                 '--policy_params_layers', conf[3],
                 ])
            index += 1
