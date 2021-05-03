import gym


class MemoryEnv(gym.Env):

    def __init__(self, env, memory_size, **kwargs) -> None:
        super().__init__()

        self.memory_size = memory_size

        self.env = gym.make('Processing-v0', env=env, **kwargs)

        self.memory = []

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(0, 255, (self.memory_size, ) + self.env.observation_space.shape)

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        self.memory.append(obs)
        if len(self.memory) > self.memory_size:
            del self.memory[0]

        return self.memory, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.memory = [obs] * self.memory_size
        return self.memory

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)