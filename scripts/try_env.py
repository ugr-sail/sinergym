import gymnasium as gym
import numpy as np
from gymnasium.wrappers.normalize import NormalizeReward

import sinergym
from sinergym.utils.wrappers import (LoggerWrapper, NormalizeAction,
                                     NormalizeObservation)

env = gym.make('Eplus-demo-v1')
env = NormalizeAction(env)
env = NormalizeObservation(env)
env = NormalizeReward(env)
env = LoggerWrapper(env)

for i in range(1):
    obs, info = env.reset()
    rewards = []
    truncated = terminated = False
    current_month = 0
    while not (terminated or truncated):
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        rewards.append(reward)
        if info['month'] != current_month:  # display results every month
            current_month = info['month']
            print('Reward: ', sum(rewards), info)
    print(
        'Episode ',
        i,
        'Mean reward: ',
        np.mean(rewards),
        'Cumulative reward: ',
        sum(rewards))
env.close()
