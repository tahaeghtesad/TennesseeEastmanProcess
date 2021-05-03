from typing import List

import gym
import numpy as np
from rl.algorithms.dqn.policies.convolution_policy import ConvolutionalPolicy
from rl.common.experience import Experience


class DoubleConvolutionPolicy(ConvolutionalPolicy):
    def __init__(self, env, learning_rate, gamma, conv_count, dens_arch, activation) -> None:

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.model_0 = self.create_model(env, conv_count, dens_arch, activation)
        self.model_1 = self.create_model(env, conv_count, dens_arch, activation)

    def create_model(self, env: gym.Env, conv_count, dens_arch, activation):
        return super(DoubleConvolutionPolicy, self).create_model(env, conv_count, dens_arch, activation)

    def predict(self, obs):
        assert self.model_0 is not None and self.model_1 is not None, 'Model is not initialized'

        if np.random.rand() < 0.5:
            selected_model = self.model_0
        else:
            selected_model = self.model_1

        return np.argmax(selected_model.predict(np.array([obs]))[0])

    def train(self, samples: List[Experience]):
        assert self.model_0 is not None and self.model_1 is not None, 'Model is not initialized'

        samples_0, samples_1 = np.array_split(samples, 2)

        states_0 = np.array([sample.state for sample in samples_0])
        states_1 = np.array([sample.state for sample in samples_1])

        next_states_0 = np.array([sample.next_state for sample in samples_0])
        next_states_1 = np.array([sample.next_state for sample in samples_1])

        q_values_0 = self.model_0.predict(states_0)
        q_values_1 = self.model_1.predict(states_1)

        q_next_values_0 = self.model_1.predict(next_states_1)
        q_next_values_1 = self.model_0.predict(next_states_0)

        for i in range(len(samples_0)):
            if samples[i].done:
                q_values_0[i][samples[i].action] = samples[i].reward
            else:
                q_values_0[i][samples[i].action] = samples[i].reward + self.gamma * np.argmax(q_next_values_1[i])

        for i in range(len(samples_1)):
            if samples[i].done:
                q_values_1[i][samples[i].action] = samples[i].reward
            else:
                q_values_1[i][samples[i].action] = samples[i].reward + self.gamma * np.argmax(q_next_values_0[i])

        self.model_0.fit(states_0, q_values_0, verbose=1, batch_size=len(samples))
        self.model_1.fit(states_1, q_values_1, verbose=1, batch_size=len(samples))