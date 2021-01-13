import gym


class ControlEnv(gym.Env):

    def __init__(self, test_env=False, noise_sigma=0.07, t_epoch=200) -> None:
        super().__init__()
        self.action_dim: int = 2
        self.observation_dim: int = 2
        self.test_env = test_env
        self.noise_sigma = noise_sigma
        self.t_epoch = t_epoch
