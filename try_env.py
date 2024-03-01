import gymnasium as gym
import numpy as np

import sinergym
from sinergym.utils.wrappers import (
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
    IncrementalWrapper,
    RoundActionWrapper)

# Creating environment and applying wrappers for normalization and logging
env = gym.make('Eplus-radiant-hot-continuous-stochastic-v1')
env = RoundActionWrapper(env)
env = IncrementalWrapper(env, incremental_variables_definition={
    'water_temperature': (2.0, 0.5)
},
    initial_values=[30.0]
)
env = NormalizeAction(env)
env = NormalizeObservation(env)
env = LoggerWrapper(env)

# Execute interactions during 3 episodes
for i in range(3):
    # Reset the environment to start a new episode
    obs, info = env.reset()
    truncated = terminated = False

    while not (terminated or truncated):
        # Random action control
        a = env.action_space.sample()
        # Read observation and reward
        obs, reward, terminated, truncated, info = env.step(a)


# Close the environment
env.close()
