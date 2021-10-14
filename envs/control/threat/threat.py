import gym

from agents.RLAgents import Agent


class ThreatModel:
    def __init__(self, env,
                 attacker: Agent,
                 defender: Agent,
                 **env_kwargs,
                 ) -> None:

        super().__init__()

        self.attacker = attacker
        self.defender = defender

        if isinstance(env, str):
            self.env = gym.make(f'{env}', **env_kwargs)
        elif isinstance(env, gym.Env):
            self.env = env
        else:
            raise Exception('Invalid Environment.')

    def set_attacker(self, attacker):
        self.attacker = attacker

    def set_defender(self, defender):
        self.defender = defender

    def reset(self):
        if self.attacker is not None:
            self.attacker.reset()
        if self.defender is not None:
            self.defender.reset()
