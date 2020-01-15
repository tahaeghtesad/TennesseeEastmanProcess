from stable_baselines import DDPG


class DDPGWrapper(DDPG):
    def predict(self, observation, state=None, mask=None, deterministic=True):
        return super().predict(observation, state, mask, deterministic)[0]
