import gym
from gym import spaces
import numpy as np

class Monopoly(gym.Env):
    metadata = None

    def __init__(self, num_players=None):
        self.observation_space = None
        self.action_space = None

    def _get_obs():
        return None

    def _get_info():
        return None

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation
    
    def step(self, action):
        observation = self._get_obs()
        reward = 0
        done = False
        info = self._get_info()
        return observation, reward, done, info