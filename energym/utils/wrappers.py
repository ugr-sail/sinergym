"""Implementation of custom Gym environments."""

import numpy as np
import gym

from collections import deque
from energym.utils.common import RANGES_5ZONE


class NormalizeObservation(gym.ObservationWrapper):

    def __init__(self, env, ranges=RANGES_5ZONE):
        """Observations normalized to range [0, 1].

        Args:
            env (object): Original Gym environment.
            ranges: Observation variables ranges to apply normalization (rely on environment)
        """
        super(NormalizeObservation, self).__init__(env)
        self.unwrapped_observation = None
        self.ranges = ranges

    def observation(self, obs):
        """Applies normalization to observation.

        Args:
            obs (object): Original observation.

        Returns:
            object: Normalized observation.
        """
        # Save original obs in class attribute
        self.unwrapped_observation = obs
        # Don't have variables name, we need to get it and add manually news
        # len(variables)!=len(obs)
        keys = self.env.variables["observation"]
        keys.append('day')
        keys.append('month')
        keys.append('hour')
        obs_dict = dict(zip(keys, obs))

        for key in obs_dict:
            # normalization
            value = obs_dict[key]
            value = (value-self.ranges[key][0]) / \
                (self.ranges[key][1]-self.ranges[key][0])
            # If value is out
            if value > 1:
                value = 1
            if value < 0:
                value = 0
            obs_dict[key] = value
        # Return obs values in the SAME ORDER than obs argument.
        return np.array(list(obs_dict.values()))

    def get_unwrapped_obs(self):
        return self.unwrapped_observation


class MultiObsWrapper(gym.Wrapper):

    def __init__(self, env, n=5):
        """Stack of observations.

        Args:
            env (object): Original Gym environment.
            n (int, optional): Number of observations to be stacked. Defaults to 5.
        """
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.history = deque([], maxlen=n)
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-5e6, high=5e6, shape=((n,) + shape), dtype=np.float32)

    def reset(self):
        """Resets the environment.

        Returns:
            list: Stacked previous observations.
        """
        obs = self.env.reset()
        for _ in range(self.n):
            self.history.append(obs)
        return self._get_obs()

    def _get_obs(self):
        """Get observation history.

        Returns:
            np.array: Array of previous observations.
        """
        return np.array(self.history)
