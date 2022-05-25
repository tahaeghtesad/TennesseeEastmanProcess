from agents.RLAgents import Agent
import numpy as np


class PIDController(Agent):

    def __init__(self, p, i, d, action_space, desired_state, name):
        super().__init__(name)
        self.integral = 0
        self.derivative = 0
        self.prev_error = 0

        self.desired_state = desired_state
        self.action_space = action_space

        self.p = p
        self.i = i
        self.d = d

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, observation, state=None, mask=None, deterministic=True):
        error = self.desired_state - observation

        self.integral += error
        self.derivative = error - self.prev_error
        self.prev_error = error

        pid = self.p * error + self.d * self.derivative + self.i * self.integral

        action = self.sigmoid(pid) * (self.action_space.high - self.action_space.low) + self.action_space.low

        return action

    def reset(self):
        self.integral = 0
        self.derivative = 0
        self.prev_error = 0
