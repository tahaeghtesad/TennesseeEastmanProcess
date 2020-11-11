import gym
import envs
from tensorforce import Runner, Environment
from tensorforce import Agent as TFAgent
from agents.RLAgents import Agent, SimpleWrapperAgent, MixedStrategyAgent, HistoryAgent, LimitedHistoryAgent, \
    SinglePolicyMixedStrategyAgent, NoOpAgent
from trainer.trainer import Trainer
import tensorflow as tf
import numpy as np
import logging


class RCTrainer(Trainer):

    def __init__(self, prefix, env_id, env_params=None, rl_params=None, policy_params=None,
                 training_params=None) -> None:
        super().__init__(prefix, env_params, rl_params, policy_params, training_params)
        self.logger = logging.getLogger(__name__)
        self.env_id = env_id

    @staticmethod
    def callback(locals_, globals_):
        # self_ = locals_['self']
        #
        # variables = ['u', 'x', 'dx', 'a', 'o']
        #
        # if 'info' in locals_ and 'writer' in locals_ and locals_['writer'] is not None:
        #     for var in variables:
        #         if var in locals_['info']:
        #             for i in range(len(locals_['info'][var])):
        #                 summary = tf.Summary(
        #                     value=[tf.Summary.Value(tag=f'env/{var}{i}', simple_value=locals_['info'][var][i])])
        #                 locals_['writer'].add_summary(summary, self_.num_timesteps)
        return True

    def get_policy_class(self, policy_params):
        # # policy params must have 'act_fun' and 'layers'
        # class CustomPolicy(LnMlpPolicy):
        #     def __init__(self, *args, **_kwargs):
        #         super().__init__(*args, **_kwargs, **policy_params)
        #
        # return CustomPolicy
        return None

    def train_attacker(self, defender, iteration):
        # env params must have 'compromise_actuation_prob', 'compromise_observation_prob', and 'power'
        # rl params must have 'gamma', 'random_exploration'
        self.logger.info(f'Starting attacker training for {self.training_params["training_steps"]} steps.')
        def get_env():
            env = Environment.create(environment='gym',
                                     level='LimitedHistoritized-v0' if self.training_params['training_params_attacker_limited_history'] else 'Historitized-v0',
                                     env=f'{self.env_id}Att-v0',
                                     **self.env_params,
                                     defender=defender)\
                if self.training_params['attacker_history'] else \
                Environment.create(environment='gym',
                                   level=f'{self.env_id}Att-v0',
                                   **self.env_params,
                                   defender=defender)
            return env

        agent = TFAgent.create(agent='ddpg',
                               environment=get_env(),
                               memory=50_000,
                               batch_size=128,
                               network='auto',
                               # use_beta_distribution=True,
                               # update_frequency=1,  # default: batch_size
                               # start_updating=128, # default: batch_size
                               learning_rate=1e-4,
                               discount=.90,
                               # critic=dict(),
                               critic_optimizer=1.0,
                               exploration=.1,
                               parallel_interactions=self.training_params['concurrent_runs'],
                               summarizer=dict(
                                   directory=f'{self.prefix}/tb_logs',
                                   filename=f'attacker-{iteration}',
                                   # list of labels, or 'all'
                                   summaries='all'
                               ))

        runner = Runner(
            agent=agent,
            environments=[get_env() for _ in range(self.training_params['concurrent_runs'])],
            # num_parallel=self.training_params['concurrent_runs']
        )

        runner.run(num_timesteps=self.training_params['training_steps'])

        agent.save(directory=f'{self.prefix}/params/', filename=f'attacker-{iteration}', format='numpy')

        def initializer():
            ddpg_model = TFAgent.load(directory=f'{self.prefix}/params/', filename=f'attacker-{iteration}', format='numpy')
            if self.training_params['attacker_history']:
                if self.training_params['attacker_limited_history']:
                    return LimitedHistoryAgent(ddpg_model)
                else:
                    return HistoryAgent(ddpg_model)
            else:
                return SimpleWrapperAgent(ddpg_model)

        return initializer

    def train_defender(self, attacker, iteration):
        self.logger.info(f'Starting defender training for {self.training_params["training_steps"]} steps.')

        def get_env():
            env = Environment.create(environment='gym',
                                     level='LimitedHistoritized-v0' if self.training_params[
                                         'training_params_defender_limited_history'] else 'Historitized-v0',
                                     env=f'{self.env_id}Def-v0',
                                     **self.env_params,
                                     attacker=attacker) \
                if self.training_params['defender_history'] else \
                Environment.create(environment='gym',
                                   level=f'{self.env_id}Def-v0',
                                   **self.env_params,
                                   attacker=attacker)
            return env

        agent = TFAgent.create(agent='ddpg',
                               environment=get_env(),
                               memory=50_000,
                               batch_size=256,
                               network=[dict(type='dense', size=64, activation='elu') for _ in range(5)],
                               use_beta_distribution=True,
                               update_frequency=1,  # default: batch_size
                               # start_updating=128,  # default: batch_size
                               learning_rate=1e-4,
                               discount=.90,
                               # critic=dict(),
                               critic_optimizer=1.0,
                               exploration=.1,
                               parallel_interactions=self.training_params['concurrent_runs'],
                               summarizer=dict(
                                   directory=f'{self.prefix}/tb_logs',
                                   filename=f'defender-{iteration}',
                                   # list of labels, or 'all'
                                   summaries='all'
                               ))

        training_envs = [get_env() for _ in range(self.training_params['concurrent_runs'])]
        runner = Runner(
            agent=agent,
            # environment=get_env()
            environments=training_envs,
            blocking=True,
            remote='multiprocessing'
        )

        runner.run(
            num_episodes=self.training_params['training_steps'] / 200,
            batch_agent_calls=True,
        )

        agent.save(directory=f'{self.prefix}/params/', filename=f'defender-{iteration}', format='numpy')

        agent.close()
        runner.close()
        for env in training_envs:
            env.close()

        def initializer():
            ddpg_model = TFAgent.load(directory=f'{self.prefix}/params/', filename=f'defender-{iteration}', format='numpy')
            if self.training_params['defender_history']:
                if self.training_params['defender_limited_history']:
                    return LimitedHistoryAgent(ddpg_model)
                else:
                    return HistoryAgent(ddpg_model)
            else:
                return SimpleWrapperAgent(ddpg_model)

        return initializer

    def get_payoff(self, attacker: Agent, defender: Agent, repeat=20):
        env = gym.make(f'{self.env_id}Def-v0',
                       **self.env_params,
                       attacker=attacker
                       )

        r = 0

        for _ in range(repeat):
            def_obs = env.reset()
            done = False
            iter = 0
            while not done:
                action = defender.predict(def_obs)
                def_obs, reward, done, info = env.step(action)

                r += reward * self.rl_params['gamma'] ** iter
                iter += 1

        return -r / repeat, r / repeat

    def initialize_strategies(self):
        attacker = NoOpAgent(4)  # TODO this should not be a constant 4
        self.logger.info('Initializing a defender against NoOp attacker...')
        defender = self.train_defender(attacker, 0)()

        au, du = self.get_payoff(attacker, defender)

        self.defender_payoff_table = np.zeros((1, 1))
        self.attacker_payoff_table = np.zeros((1, 1))

        attacker_ms = SinglePolicyMixedStrategyAgent()
        defender_ms = SinglePolicyMixedStrategyAgent()

        self.defender_payoff_table[0, 0] = du
        self.attacker_payoff_table[0, 0] = au

        attacker_ms.add_policy(attacker)
        defender_ms.add_policy(defender)

        attacker_ms.update_probabilities(np.ones(1))
        defender_ms.update_probabilities(np.ones(1))

        return attacker_ms, defender_ms
