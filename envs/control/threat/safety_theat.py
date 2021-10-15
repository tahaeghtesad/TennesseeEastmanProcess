from agents.RLAgents import Agent
from envs.control.threat.threat import ThreatModel
import numpy as np


class SafetyThreatModel(ThreatModel):
    def __init__(self, env, attacker: Agent, defender: Agent, **env_kwargs) -> None:
        super().__init__(env, attacker, defender, **env_kwargs)
        self.adversarial_goal = None
        self.last_adversarial_distance_to_goal = 0

        self.last_env_observation = None

    def reset(self):
        super().reset()
        _ = self.env.reset()
        self.adversarial_goal = self.env.goal_pos
        self.last_env_observation = self.env.reset()
        lidar_to_adversary = self.env.obs_lidar([self.adversarial_goal], 0)
        adversary_obs = np.hstack((self.env.action_space.low, lidar_to_adversary))
        return adversary_obs, self.last_env_observation

    def step(self):
        defender_action = self.defender.predict(self.last_env_observation)
        attacker_action = self.attacker.predict(np.hstack((defender_action, self.last_env_observation)))

        self.last_env_observation, env_reward, done, info = self.env.step(attacker_action + defender_action)
        adversarial_distance_to_goal = self.env.dist_xy(self.adversarial_goal)

        goal_penalty = 0
        if self.env.dist_xy(self.env.goal_pos) <= 0.3:
            goal_penalty = -1
            if self.env.dist_xy(self.adversarial_goal) <= 0.3:
                goal_penalty = 0
        adversary_reward = (self.last_adversarial_distance_to_goal - adversarial_distance_to_goal + goal_penalty) * 0.01
        self.last_adversarial_distance_to_goal = adversarial_distance_to_goal
        if self.env.dist_xy(self.env.goal_pos) <= 0.3 or self.env.dist_xy(self.adversarial_goal) < 0.3:
            done = True

        lidar_to_adversary = self.env.obs_lidar([self.adversarial_goal], 0)
        adversary_obs = np.hstack((defender_action, lidar_to_adversary))

        # obs, reward, done, info
        return (adversary_obs, self.last_env_observation), self.last_env_observation,\
               (adversary_reward, env_reward), done, info
