import gymnasium as gym
import numpy as np
from sinergym.utils.logger import TerminalLogger
import logging

import sinergym
from sinergym.utils.wrappers import (CSVLogger, LoggerWrapper, NormalizeAction,
                                     NormalizeObservation)

# Optional: Terminal log in the same format as Sinergym.
# Logger info can be replaced by print.
terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(
    name='MAIN',
    level=logging.INFO
)

# Creating environment and applying wrappers for normalization and logging
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
env = NormalizeAction(env)
env = NormalizeObservation(env)
env = LoggerWrapper(env)
env = CSVLogger(env)

# Execute interactions during 3 episodes
for i in range(3):
    # Reset the environment to start a new episode
    obs, info = env.reset()
    rewards = []
    truncated = terminated = False
    current_month = 0
    while not (terminated or truncated):
        # Random action control
        a = env.action_space.sample()
        # Read observation and reward
        obs, reward, terminated, truncated, info = env.step(a)
        rewards.append(reward)
        # If this timestep is a new month start
        if info['month'] != current_month:  # display results every month
            current_month = info['month']
            # Print information
            logger.info('Reward: {}'.format(sum(rewards)))
            logger.info('Info: {}'.format(info))
    # Final episode information print
    logger.info('Episode {} - Mean reward: {} - Cumulative Reward: {}'.format(i,
                                                                              np.mean(rewards), sum(rewards)))
# Close the environment
env.close()
