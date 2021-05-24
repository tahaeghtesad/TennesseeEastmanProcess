from typing import List

import gym
import numpy as np
import tensorflow as tf
from gym.spaces import Discrete
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam

from rl.common.experience import Experience


class ConvolutionalPolicy:
    def __init__(self, env, learning_rate, gamma, conv_count, dens_arch, activation) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.model = self.create_model(env, conv_count, dens_arch, activation)

    def create_model(self, env: gym.Env, conv_count, dens_arch, activation):
        assert isinstance(env.action_space, Discrete), 'Action Space should be discrete.'

        model = Sequential()
        model.add(Input((env.observation_space.shape)))
        for _ in range(conv_count):
            model.add(Conv2D(filters=4, kernel_size=(3, 3), strides=1, data_format="channels_first", activation=activation,
                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Flatten())
        for n in dens_arch:
            model.add(Dense(n, activation=activation, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Dense(env.action_space.n, activation='linear'))
        optimizer = Adam(self.learning_rate)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        model.summary()

        return model

    def predict(self, obs):
        assert self.model is not None, 'Model is not initialized'
        return np.argmax(self.model.predict(np.array([obs]))[0])

    def train(self, samples: List[Experience]):
        assert self.model is not None, 'Model is not initialized'

        states = np.array([sample.state for sample in samples])
        next_states = np.array([sample.next_state for sample in samples])

        q_values = self.model.predict(states)
        q_next_values = self.model.predict(next_states)

        for i in range(len(samples)):
            if samples[i].done:
                q_values[i][samples[i].action] = samples[i].reward
            else:
                q_values[i][samples[i].action] = samples[i].reward + self.gamma * np.argmax(q_next_values[i])

        self.model.fit(states, q_values, verbose=0, batch_size=len(samples))