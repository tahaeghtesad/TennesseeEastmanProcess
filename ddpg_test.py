from tensorforce import Agent, Environment, Runner
import gym
import envs
import uuid

def get_env():
        # env = Environment.create(environment='gym',
        #                          level='TT-v0')
    env = dict(environment='gym', level='MountainCarContinuous-v0')
    return env


concurrent_runs = 8
id = f'{uuid.uuid4().__str__().split("-")[0]}'
print(f'ID: {id}')

agent = Agent.create(agent='ddpg',
                     environment=Environment.create(get_env()),
                     batch_size=256,
                     memory=10_000,
                     parallel_interactions=concurrent_runs,
                     summarizer=dict(
                         directory=f'test_log/tb_logs',
                         filename=f'base-{id}',
                         # list of labels, or 'all'
                         summaries='all'
                     ))

# agent = Agent.create(agent='ddpg',
#                        environment=get_env(),
#                        memory=50_000,
#                        batch_size=256,
#                        network=[dict(type='dense', size=64, activation='elu') for _ in range(5)],
#                        use_beta_distribution=True,
#                        update_frequency=1,  # default: batch_size
#                        # start_updating=128,  # default: batch_size
#                        learning_rate=1e-4,
#                        discount=.90,
#                        # critic=dict(),
#                        critic_optimizer=1.0,
#                        exploration=.1,
#                        parallel_interactions=concurrent_runs,
#                        summarizer=dict(
#                            directory=f'test_log/tb_logs',
#                            filename=f'base',
#                            # list of labels, or 'all'
#                            summaries='all'
#                        ))
#
training_envs = [get_env() for _ in range(concurrent_runs)]
try:
    runner = Runner(
        agent=agent,
        environments=training_envs,
        num_parallel=concurrent_runs,
        blocking=True,
        remote='multiprocessing'
    )

    runner.run(
        num_episodes=2000,
        batch_agent_calls=True,
    )
finally:
    runner.close()
