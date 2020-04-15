from stable_baselines import DDPG


class DDPGWrapper(DDPG):
    def predict(self, observation, state=None, mask=None, deterministic=True):
        try:
            return super().predict(observation, state, mask, deterministic)[0]
        except ValueError:
            # TODO make sure this is appropriate for your env
            return super().predict(observation[:2], state, mask, deterministic)[0]
