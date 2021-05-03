import gym
import envs
import click
from rl.algorithms.dqn.interface import DQN
from rl.algorithms.dqn.policies.convolution_policy import ConvolutionalPolicy
from rl.algorithms.dqn.policies.double_convolution_policy import DoubleConvolutionPolicy


@click.command(name='pong')
@click.option('--total_timesteps', default=500_000)
@click.option('--learning_rate', default=0.0005)
@click.option('--gamma', default=0.95)
@click.option('--conv_count', default=3)
@click.option('--dens_arch', default='128, 128, 128')
@click.option('--activation', default='relu')
@click.option('--replay_buffer_size', default=10000)
@click.option('--epsilon', default=0.1)
@click.option('--sample_size', default=128)
def test_pong_rl(
        total_timesteps,
        learning_rate,
        gamma,
        conv_count,
        dens_arch,
        activation,
        replay_buffer_size,
        epsilon,
        sample_size
):
    env = gym.make('Memory-v0', env='PongDeterministic-v4', memory_size=4)
    policy = ConvolutionalPolicy(env, learning_rate, gamma, conv_count, [int(l) for l in dens_arch.split(',')], activation)

    # policy = DoubleConvolutionPolicy(env, learning_rate, gamma, conv_count, [int(l) for l in dens_arch.split(',')], activation)
    dqn = DQN(env, policy, replay_buffer_size, epsilon)
    dqn.train(total_timesteps, sample_size)

    average_reward = dqn.evaluate(20_000)

    print(f'Finished Training. Average Reward: {average_reward:.2f}')


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    pass


main.add_command(test_pong_rl)

if __name__ == '__main__':
    main()
