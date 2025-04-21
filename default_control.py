import logging

import gymnasium as gym
import numpy as np

import sinergym
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.wrappers import (
    CSVLogger,
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
)

# Logger
terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(
    name='MAIN',
    level=logging.INFO
)

# Create environment and apply wrappers for normalization and logging
env = gym.make('Eplus-5zone-cool-continuous-stochastic-v1',
               actuators={},
               action_space=gym.spaces.Box(
                   low=0,
                   high=0,
                   shape=(0,)))
# env = NormalizeAction(env)
# env = NormalizeObservation(env)
env = LoggerWrapper(env)
env = CSVLogger(env)

# Execute 3 episodes
for i in range(10):

    # Reset the environment to start a new episode
    obs, info = env.reset()

    truncated = terminated = False

    while not (terminated or truncated):

        # Random action selection
        a = env.action_space.sample()

        # Perform action and receive env information
        obs, reward, terminated, truncated, info = env.step(a)

env.close()
