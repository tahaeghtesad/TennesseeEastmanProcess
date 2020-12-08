import gym


class ControlEnv(gym.Env):

    def __init__(self, test_env=False, noise=True) -> None:
        super().__init__()
        self.action_dim: int = 2
        self.observation_dim: int = 2
        self.test_env = test_env
        self.noise = noise
