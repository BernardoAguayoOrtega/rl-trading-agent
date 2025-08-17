import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StrategyEnv(gym.Env):
    """
    Custom Environment for the RL agent to learn how to build trading strategies.
    """
    def __init__(self, df):
        super(StrategyEnv, self).__init__()
        pass

    def step(self, action):
        pass

    def reset(self, seed=None, options=None):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass