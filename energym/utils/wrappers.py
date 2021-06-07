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

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        normalized_obs = self.observation(observation)
        if self.flag_discrete and type(action) == int:
            action_ = self.action_mapping[action]
        else:
            action_ = action
        # Eliminate day,month, hour from observation
        self.logger.log_step_normalize(timestep=info['timestep'],
                                       date=[info['month'],
                                             info['day'], info['hour']],
                                       observation=normalized_obs[:-3],
                                       action=action_,
                                       simulation_time=info['time_elapsed'],
                                       reward=reward,
                                       total_power_no_units=info['total_power_no_units'],
                                       comfort_penalty=info['comfort_penalty'],
                                       done=done)
        return normalized_obs, reward, done, info

    def observation(self, obs):
        """Applies normalization to observation.

        Args:
            obs (object): Original observation.

        Returns:
            object: Normalized observation.
        """
        # Save original obs in class attribute
        self.unwrapped_observation = obs.copy()
        variables = self.env.variables["observation"]

        # NOTE: If you want to recor day, month and our. You should add to variables that keys
        for i, variable in enumerate(variables):
            # normalization
            obs[i] = (obs[i]-self.ranges[variable][0]) / \
                (self.ranges[variable][1]-self.ranges[variable][0])
            # If value is out
            if obs[i] > 1:
                obs[i] = 1
            if obs[i] < 0:
                obs[i] = 0
        # Return obs values in the SAME ORDER than obs argument.
        return np.array(obs)

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
