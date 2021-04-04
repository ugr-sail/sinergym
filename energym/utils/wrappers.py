"""Script for implementing custom Gym environments."""


import numpy as np
import gym


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize the observation to range [-1, 1]."""
    def __init__(self, env):
        super(NormalizeObservation, self).__init__(env)

    def observation(self, obs):
        return np.tanh(obs)