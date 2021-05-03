import gym
import numpy as np
from tqdm import tqdm
from rl.algorithms.dqn.policies.convolution_policy import ConvolutionalPolicy
from rl.common.experience import Experience
from rl.common.replay import ExperienceReplayBuffer
from rl.exceptions.common import EmptyReplayBufferException


class DQN:
    def __init__(self, env: gym.Env, policy, replay_buffer_size, epsilon) -> None:
        super().__init__()

        self.env = env
        self.policy = policy
        self.replay_buffer = ExperienceReplayBuffer(replay_buffer_size)
        self.epsilon = epsilon

    def train(self, total_timesteps, sample_size):

        prev_obs = self.env.reset()

        for step in tqdm(range(total_timesteps)):

            if np.random.rand() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = self.policy.predict(prev_obs)

            obs, reward, done, info = self.env.step(action)

            self.replay_buffer.add_experience(Experience(
                prev_obs,
                action,
                obs,
                reward,
                done
            ))

            if done:
                prev_obs = self.env.reset()
            else:
                prev_obs = obs

            try:
                samples = self.replay_buffer.sample(sample_size)
                self.policy.train(samples)
            except EmptyReplayBufferException:
                pass

    def evaluate(self, total_timesteps):
        obs = self.env.reset()

        rewards = []

        for step in range(total_timesteps):

            action = self.policy.predict(obs)

            obs, reward, done, info = self.env.step(action)

            if done:
                obs = self.env.reset()

            rewards.append(reward)

        return sum(rewards)/len(rewards)
