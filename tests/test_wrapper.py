import csv
import os

from collections import deque
import numpy as np
import pytest
from stable_baselines3.common.env_util import is_wrapped

from sinergym.utils.wrappers import NormalizeObservation
from sinergym.utils.common import RANGES_5ZONE


@pytest.mark.parametrize('env_name',
                         [('env_wrapper_normalization'),
                          ('env_all_wrappers'),
                          ])
def test_normalization_wrapper(env_name, request):
    env = request.getfixturevalue(env_name)

    # Check if new attributes have been created in environment
    assert hasattr(env, 'unwrapped_observation')
    assert hasattr(env, 'ranges')

    # Check initial values of that attributes
    assert env.unwrapped_observation is None
    assert env.ranges == RANGES_5ZONE

    # Initialize env
    obs = env.reset()

    # Check observation normalization
    assert (obs >= 0).all() and (obs <= 1).all()
    # Check original observation recording
    assert env.unwrapped_observation is not None

    # Simulation random step
    a = env.action_space.sample()
    obs, reward, done, info = env.step(a)

    assert (obs >= 0).all() and (obs <= 1).all()
    assert env.unwrapped_observation is not None


def test_multiobs_wrapper(env_wrapper_multiobs, env_demo_continuous):
    shape = env_wrapper_multiobs

    # Check attributes don't exist in original env
    assert not (hasattr(
        env_demo_continuous,
        'n') or hasattr(
        env_demo_continuous,
        'ind_flat') or hasattr(
            env_demo_continuous,
        'history'))
    # Check attributes exist in wrapped env
    assert hasattr(
        env_wrapper_multiobs,
        'n') and hasattr(
        env_wrapper_multiobs,
        'ind_flat') and hasattr(
            env_wrapper_multiobs,
        'history')

    # Check history
    assert env_wrapper_multiobs.history == deque([])

    # Check observation space transformation
    original_shape = env_demo_continuous.observation_space.shape[0]
    wrapped_shape = env_wrapper_multiobs.observation_space.shape[0]
    assert wrapped_shape == original_shape * env_wrapper_multiobs.n

    # Check reset obs
    obs = env_wrapper_multiobs.reset()
    assert len(obs) == wrapped_shape
    for i in range(env_wrapper_multiobs.n - 1):
        # Check store same observation n times
        assert (obs[original_shape * i:original_shape *
                    (i + 1)] == obs[0:original_shape]).all()
        # Check history save same observation n times
        assert (env_wrapper_multiobs.history[i] ==
                env_wrapper_multiobs.history[i + 1]).all()

    # Check step obs
    a = env_wrapper_multiobs.action_space.sample()
    obs, reward, done, info = env_wrapper_multiobs.step(a)

    # Last observation must be different of the rest of them
    assert (obs[original_shape * (env_wrapper_multiobs.n - 1):]
            != obs[0:original_shape]).any()
    assert (env_wrapper_multiobs.history[0] !=
            env_wrapper_multiobs.history[-1]).any()


@ pytest.mark.parametrize('env_name',
                          [('env_wrapper_logger'), ('env_all_wrappers'), ])
def test_logger_wrapper(env_name, request):
    env = request.getfixturevalue(env_name)
    logger = env.logger
    env.reset()

    # Check CSV's have been created and linked in simulator correctly
    assert logger.log_progress_file == env.simulator._env_working_dir_parent + '/progress.csv'
    assert logger.log_file == env.simulator._eplus_working_dir + '/monitor.csv'

    tmp_log_file = logger.log_file

    # simulating short episode
    for _ in range(10):
        env.step(env.action_space.sample())
    env.reset()

    assert os.path.isfile(logger.log_progress_file)
    assert os.path.isfile(tmp_log_file)

    # If env is wrapped with normalize obs...
    if is_wrapped(env, NormalizeObservation):
        print(logger.log_file[:-4] + '_normalized.csv')
        assert os.path.isfile(tmp_log_file[:-4] + '_normalized.csv')
    else:
        assert not os.path.isfile(tmp_log_file[:-4] + '_normalized.csv')

    # Check headers
    with open(tmp_log_file, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            assert ','.join(row) == logger.monitor_header
            break
    with open(logger.log_progress_file, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            assert ','.join(row) + '\n' == logger.progress_header
            break
    if is_wrapped(env, NormalizeObservation):
        with open(tmp_log_file[:-4] + '_normalized.csv', mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                assert ','.join(row) == logger.monitor_header
                break

    env.close()


def test_env_wrappers(env_all_wrappers):
    # env_wrapper history should be empty at the beginning
    assert env_all_wrappers.history == deque([])
    for i in range(1):  # Only need 1 episode
        obs = env_all_wrappers.reset()
        # This obs should be normalize --> [0,1]
        assert (obs >= 0).all() and (obs <= 1).all()

        done = False
        while not done:
            a = env_all_wrappers.action_space.sample()
            obs, reward, done, info = env_all_wrappers.step(a)

    # Let's check if history has been completed succesfully
    assert len(env_all_wrappers.history) == 5
    assert isinstance(env_all_wrappers.history[0], np.ndarray)
    env_all_wrappers.close()
