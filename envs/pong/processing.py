import gym
import tensorflow as tf


class ProcessingEnv(gym.Env):

    def __init__(self, env, **kwargs) -> None:
        super().__init__()

        self.env = gym.make(env, **kwargs)

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 1))
        # self.observation_space = gym.spaces.Box(0, 255, (210, 160, 1))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.process_image(obs), reward, done, info

    def process_image(self, img):
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.resize(img, [84, 84])
        return img

    def reset(self):
        return self.process_image(self.env.reset())

    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)