import pytest
import numpy as np


def test_env_wrappers(env_wrapper):
    # env_wrapper history should be empty at the beginning
    assert len(env_wrapper.history) == 0
    for i in range(1):  # Only need 1 episode
        obs = env_wrapper.reset()
        # This obs should be normalize --> [-1,1]
        assert (obs >= 0).all() and (obs <= 1).all()

        done = False
        while not done:
            a = env_wrapper.action_space.sample()
            obs, reward, done, info = env_wrapper.step(a)

    # Let's check if history has been completed succesfully
    assert len(env_wrapper.history) == 5
    assert type(env_wrapper.history[0]) == np.ndarray
    env_wrapper.close()
