import gymnasium as gym
import numpy as np

import sinergym
from sinergym.utils.wrappers import (LoggerWrapper, NormalizeAction,
                                     NormalizeObservation)

# Creating environment and applying wrappers for normalization and logging
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
env = NormalizeAction(env)
env = NormalizeObservation(env)
env = LoggerWrapper(env)

# Execute interactions during 3 episodes
for i in range(3):
    # Reset the environment to start a new episode
    obs, info = env.reset()
    rewards = []
    terminated = False
    current_month = 0
    while not terminated:
        # Random action control
        a = env.action_space.sample()
        # Read observation and reward
        obs, reward, terminated, truncated, info = env.step(a)
        rewards.append(reward)
        # If this timestep is a new month start
        if info['month'] != current_month:  # display results every month
            current_month = info['month']
            # Print information
            print('Reward: ', sum(rewards), info)
    # Final episode information print
    print(
        'Episode ',
        i,
        'Mean reward: ',
        np.mean(rewards),
        'Cumulative reward: ',
        sum(rewards))
# Close the environment
env.close()
