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
env = gym.make(
    'Eplus-5zone-hot-continuous-stochastic-v1',
    max_ep_data_store_num=1)
env = NormalizeAction(env)
env = NormalizeObservation(env)
env = LoggerWrapper(env)
env = CSVLogger(env)

obs, info = env.reset()

env.get_wrapper_attr('info')()

for i in range(1):
    obs, info = env.reset()
    truncated = terminated = False
    while not (terminated or truncated):
        action = np.array([14.0, 24.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

env.close()
