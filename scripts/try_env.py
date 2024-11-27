import logging

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.normalize import NormalizeReward

import sinergym
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.wrappers import (LoggerWrapper, NormalizeAction,
                                     NormalizeObservation)

terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(
    name='MAIN',
    level=logging.INFO
)

env = gym.make('Eplus-demo-v1')
env = NormalizeAction(env)
env = NormalizeObservation(env)
env = NormalizeReward(env)
env = LoggerWrapper(env)

# Execute 1 episode
for i in range(1):
    # Reset the environment to start a new episode
    obs, info = env.reset()
    rewards = []
    truncated = terminated = False
    current_month = 0
    while not (terminated or truncated):
        # Random action
        a = env.action_space.sample()

        # Perform action
        obs, reward, terminated, truncated, info = env.step(a)

        rewards.append(reward)

        #  Display results every simulated month
        if info['month'] != current_month:
            current_month = info['month']
            logger.info('Reward: {}'.format(sum(rewards)))
            logger.info('Info: {}'.format(info))

    logger.info('Episode {} - Mean reward: {} - Cumulative Reward: {}'.format(i,
                                                                              np.mean(rewards), sum(rewards)))
env.close()
