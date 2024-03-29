import logging

from agents.RLAgents import Agent, SimpleWrapperAgent, MixedStrategyAgent
from envs.mtd.processors import AttackerProcessor, DefenderProcessor
from trainer.trainer import Trainer
import tensorflow as tf
from stable_baselines.deepq.policies import FeedForwardPolicy, LnMlpPolicy
from stable_baselines import DQN
import gym
import envs
from envs.mtd.heuristics import *
from tqdm import tqdm


class MTDTrainer(Trainer):

    def __init__(self, prefix, env_params=None, rl_params=None,
                 policy_params=None, training_params=None) -> None:
        super().__init__(prefix, env_params, rl_params, policy_params, training_params)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def callback(locals_, globals_):
        # self_ = locals_['self']
        #
        # if 'action' in locals_:
        #     summary = tf.Summary(value=[tf.Summary.Value(tag='game/actions', simple_value=locals_['action'])])
        #     locals_['writer'].add_summary(summary, self_.num_timesteps)
        #
        # if 'update_eps' in locals_:
        #     summary = tf.Summary(value=[tf.Summary.Value(tag='input_info/eps', simple_value=locals_['update_eps'])])
        #     locals_['writer'].add_summary(summary, self_.num_timesteps)
        #
        # if 'info' in locals_:
        #     summary = tf.Summary(
        #         value=[tf.Summary.Value(tag='game/attacker_reward', simple_value=locals_['info']['rewards']['att'])])
        #     locals_['writer'].add_summary(summary, self_.num_timesteps)
        #
        #     summary = tf.Summary(
        #         value=[tf.Summary.Value(tag='game/defender_reward', simple_value=locals_['info']['rewards']['def'])])
        #     locals_['writer'].add_summary(summary, self_.num_timesteps)

        return True

    def get_policy_class(self, policy_params):
        class CustomPolicy(FeedForwardPolicy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, **policy_params, feature_extraction="mlp")
        return CustomPolicy

    def initialize_strategies(self):
        if self.training_params['include_heuristics']:
            attackers = [BaseAttacker, MaxProbeAttacker, UniformAttacker, ControlThresholdAttacker]
            defenders = [BaseDefender, ControlThresholdDefender, PCPDefender, UniformDefender, MaxProbeDefender]

            self.defender_payoff_table = np.zeros((4, 5))
            self.attacker_payoff_table = np.zeros((4, 5))

            attacker_ms = MixedStrategyAgent()
            defender_ms = MixedStrategyAgent()

            for i, attacker in enumerate(attackers):
                for j, defender in enumerate(defenders):

                    au, du = self.get_payoff(attacker(m=self.env_params['m']), defender(m=self.env_params['m']))
                    self.defender_payoff_table[i, j] = du
                    self.attacker_payoff_table[i, j] = au

            for attacker in attackers:
                attacker_ms.add_policy(attacker(m=self.env_params['m']))
            for defender in defenders:
                defender_ms.add_policy(defender(m=self.env_params['m']))
        else:
            attacker = BaseAttacker(m=self.env_params['m'])
            defender = BaseDefender(m=self.env_params['m'])

            self.defender_payoff_table = np.zeros((1, 1))
            self.attacker_payoff_table = np.zeros((1, 1))

            attacker_ms = MixedStrategyAgent()
            defender_ms = MixedStrategyAgent()

            au, du = self.get_payoff(attacker, defender)
            self.defender_payoff_table[0, 0] = du
            self.attacker_payoff_table[0, 0] = au

            attacker_ms.add_policy(attacker)
            defender_ms.add_policy(defender)

        self.save_tables()
        return attacker_ms, defender_ms

    def train_attacker(self, defender, iteration):
        self.logger.info(f'Starting attacker training for {self.training_params["training_steps"]} steps.')
        env = gym.make('MTDAtt-v0', **self.env_params,
                   defender=defender)

        attacker_model = DQN(
            policy=self.get_policy_class(self.policy_params),
            env=env,
            **self.rl_params,
            verbose=2,
            tensorboard_log=f'{self.prefix}/tb_logs' if self.training_params['tb_logging'] else None,
            full_tensorboard_log=self.training_params['tb_logging']
        )

        for _ in tqdm(range(0, self.training_params['training_steps'], 10000)):
            attacker_model.learn(
                total_timesteps=10000,
                callback=self.callback,
                tb_log_name=f'attacker_{iteration}'
            )

        attacker_model.save(f'{self.prefix}/params/attacker-{iteration}')

        def initializer():
            dqn_model = DQN.load(f'{self.prefix}/params/attacker-{iteration}')
            return SimpleWrapperAgent(dqn_model)

        return initializer

    def train_defender(self, attacker, iteration):
        self.logger.info(f'Starting defender training for {self.training_params["training_steps"]} steps.')
        env = gym.make('MTDDef-v0', **self.env_params,
                       attacker=attacker)
        defender_model = DQN(
            policy=self.get_policy_class(self.policy_params),
            env=env,
            **self.rl_params,
            verbose=2,
            tensorboard_log=f'{self.prefix}/tb_logs' if self.training_params['tb_logging'] else None,
            full_tensorboard_log=self.training_params['tb_logging']
        )

        for _ in tqdm(range(0, self.training_params['training_steps'], 10000)):
            defender_model.learn(
                total_timesteps=10000,
                callback=self.callback,
                tb_log_name=f'defender_{iteration}'
            )

        defender_model.save(f'{self.prefix}/params/defender-{iteration}')

        def initializer():
            dqn_model = DQN.load(f'{self.prefix}/params/defender-{iteration}')
            return SimpleWrapperAgent(dqn_model)

        return initializer

    def get_payoff(self, attacker: Agent, defender: Agent, repeat=5):
        env = gym.make('MTD-v0', **self.env_params)
        ap = AttackerProcessor(m=self.env_params['m'])
        dp = DefenderProcessor(m=self.env_params['m'])

        ar, dr = 0, 0

        for i in range(repeat):

            initial_obs = env.reset()
            done = False
            iter = 0

            a_obs = ap.process_observation(initial_obs)
            d_obs = dp.process_observation(initial_obs)

            while not done:
                obs, reward, done, info = env.step((
                    ap.process_action(attacker.predict(a_obs)),
                    dp.process_action(defender.predict(d_obs))
                ))

                a_obs, a_r, a_d, a_i = ap.process_step(obs, reward, done, info)
                d_obs, d_r, d_d, d_i = dp.process_step(obs, reward, done, info)

                ar += a_r * self.rl_params['gamma'] ** iter
                dr += d_r * self.rl_params['gamma'] ** iter

                iter += 1

        return ar / repeat, dr / repeat