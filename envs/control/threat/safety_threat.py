from agents.RLAgents import Agent
from envs.control.threat.threat import ThreatModel
import numpy as np
from safety_gym.envs.engine import Engine


class SafetyThreatModel(ThreatModel):
    def __init__(self, env, attacker: Agent, defender: Agent, **env_kwargs) -> None:
        super().__init__(env, attacker, defender, **env_kwargs)
        self.adversarial_goal = None
        self.last_adversarial_distance_to_goal = 0

        self.last_env_observation = None

        config = self.env.config
        config['placements_extents'] = [-2.0, -2.0, 2.0, 2.0]
        config['lidar_max_dist'] = 8 * config['placements_extents'][3]
        self.env = Engine(config)

    def reset(self):
        super().reset()
        _ = self.env.reset()
        self.adversarial_goal = self.env.goal_pos
        self.last_env_observation = self.env.reset()
        return np.hstack((self.env.action_space.low, self.env.obs_lidar([self.adversarial_goal], 0))), self.last_env_observation

    def step(self):
        defender_action = self.defender.predict(self.last_env_observation)
        attacker_action = self.attacker.predict(np.hstack((defender_action, self.env.obs_lidar([self.adversarial_goal], 0))))

        self.last_env_observation, env_reward, done, info = self.env.step(attacker_action + defender_action)
        adversarial_distance_to_goal = self.env.dist_xy(self.adversarial_goal)

        goal_penalty = 0
        if self.env.dist_xy(self.env.goal_pos) <= 0.3:
            goal_penalty = -1
            if self.env.dist_xy(self.adversarial_goal) <= 0.3:
                goal_penalty = 0

        adversary_reward = (self.last_adversarial_distance_to_goal - adversarial_distance_to_goal + goal_penalty)
        self.last_adversarial_distance_to_goal = adversarial_distance_to_goal
        if self.env.dist_xy(self.env.goal_pos) <= 0.3:
            done = True
        if adversarial_distance_to_goal < 0.3:
            done = True

        # obs, reward, done, info
        return (np.hstack((defender_action, self.env.obs_lidar([self.adversarial_goal], 0))), self.last_env_observation),\
               (adversary_reward, env_reward), done, info
