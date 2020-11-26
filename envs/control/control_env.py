import gym


class ControlEnv(gym.Env):

    def __init__(self) -> None:
        super().__init__()
        self.action_dim: int = 2
        self.observation_dim: int = 2
