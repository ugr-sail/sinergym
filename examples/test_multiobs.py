"""
Example of gym wrappers use to combine multiple observations
"""

import gym
import energym
import numpy as np

from collections import deque

class MultiObsWrapper(gym.Wrapper):
    def __init__(self, env, n=5):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.history = deque([], maxlen = n)
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=-5e6, high=5e6, shape=((n,) + shape), dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n):
            self.history.append(obs)
        return self._get_obs()

    def _get_obs(self):
        return np.array(self.history)

env = gym.make('Eplus-demo-v1')

wrapped_env = MultiObsWrapper(env)

for i in range(1):
    obs = wrapped_env.reset()
    rewards = []
    done = False
    current_month = 0
    while not done:
        a = env.action_space.sample()
        obs, reward, done, info = wrapped_env.step(a)
        rewards.append(reward)
        if info['month'] != current_month: # display results every month
            current_month = info['month']
            print('Reward: ', sum(rewards), info)
    print('Episode ', i, 'Mean reward: ', np.mean(rewards), 'Cumulative reward: ', sum(rewards))
wrapped_env.close()