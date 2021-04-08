"""Implementation of custom Gym environments."""

import numpy as np
import gym

from collections import deque

class NormalizeObservation(gym.ObservationWrapper):
    """Normalize the observation to range [-1, 1]."""
    def __init__(self, env):
        super(NormalizeObservation, self).__init__(env)

    def observation(self, obs):
        return np.tanh(obs)

class MultiObsWrapper(gym.Wrapper):
    """Stack multiple observations."""
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