import csv
import os
from collections import deque

import numpy as np
import pytest

import gymnasium as gym
from sinergym.utils.common import is_wrapped
from sinergym.utils.wrappers import NormalizeObservation


@pytest.mark.parametrize('env_name',
                         [('env_wrapper_normalization'),
                          ('env_all_wrappers'),
                          ])
def test_normalization_wrapper(env_name, request):
    env = request.getfixturevalue(env_name)

    # Check if new attributes have been created in environment
    assert hasattr(env, 'unwrapped_observation')

    # Check initial values of that attributes
    assert env.unwrapped_observation is None

    # Initialize env
    obs, _ = env.reset()

    # Check observation normalization
    # ...
    # Check original observation recording
    assert env.unwrapped_observation is not None

    # Simulation random step
    a = env.action_space.sample()
    obs, _, _, _, _ = env.step(a)

    # ...
    assert env.unwrapped_observation is not None


@pytest.mark.parametrize('env_name',
                         [('env_wrapper_multiobjective'),
                          ('env_all_wrappers'),
                          ])
def test_multiobjective_wrapper(env_name, request):
    env = request.getfixturevalue(env_name)
    assert hasattr(env, 'reward_terms')
    env.reset()
    action = env.action_space.sample()
    _, reward, _, _, info = env.step(action)
    assert isinstance(reward, list)
    assert len(reward) == len(env.reward_terms)


@pytest.mark.parametrize('env_name',
                         [('env_wrapper_datetime'),
                          ('env_all_wrappers'),
                          ])
def test_datetime_wrapper(env_name, request):
    env = request.getfixturevalue(env_name)

    observation_variables = env.observation_variables
    # Check observation varibles have been updated
    assert 'day' not in observation_variables
    assert 'month' not in observation_variables
    assert 'hour' not in observation_variables
    assert 'is_weekend' in observation_variables
    assert 'month_sin' in observation_variables
    assert 'month_cos' in observation_variables
    assert 'hour_sin' in observation_variables
    assert 'hour_cos' in observation_variables
    # Check new returned observation values are valid
    env.reset()
    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    obs_dict = dict(zip(observation_variables, obs))
    assert obs_dict['is_weekend'] is not None and obs_dict['month_sin'] is not None and obs_dict[
        'month_cos'] is not None and obs_dict['hour_sin'] is not None and obs_dict['hour_cos'] is not None


@pytest.mark.parametrize('env_name',
                         [('env_wrapper_previousobs'),
                          ])
def test_previous_observation_wrapper(env_name, request):

    env = request.getfixturevalue(env_name)
    # Check that the original variable names with previous name added is
    # present
    previous_variable_names = [
        var for var in env.observation_variables if '_previous' in var]

    # Check previous observation stored has the correct len and initial values
    assert len(env.previous_observation) == 3
    assert len(previous_variable_names) == len(
        env.previous_observation)
    # Check reset and np.zeros is added in obs as previous variables
    assert (env.previous_observation == 0.0).all()
    obs1, _ = env.reset()
    original_obs1 = []
    for variable in env.previous_variables:
        original_obs1.append(
            obs1[env.observation_variables.index(variable)])

    # Check step variables is added in obs previous variables
    action = env.action_space.sample()
    obs2, _, _, _, _ = env.step(action)

    # Original obs1 values should be previous variables for obs 2
    assert np.array_equal(
        original_obs1, obs2[-len(env.previous_variables):])


@pytest.mark.parametrize('env_name',
                         [('env_wrapper_incremental'),
                          ('env_all_wrappers'),
                          ])
def test_incremental_wrapper(env_name, request):

    env = request.getfixturevalue(env_name)
    # Check initial setpoints values is initialized
    assert len(env.current_setpoints) > 0
    # Check if action selected is applied correctly
    env.reset()
    action = 16
    _, _, _, _, info = env.step(action)
    assert (env.current_setpoints == info['action']).all()
    # Check environment clip actions(
    for i in range(10):
        env.step(2)  # [1,0]
    assert env.current_setpoints[0] == env.real_space.high[0]


def test_discretize_wrapper(env_wrapper_discretize):

    env = env_wrapper_discretize
    # Check is a discrete env and original env is continuous
    # Wrapped env
    assert env.is_discrete
    assert env.action_space.n == 10
    assert isinstance(env.action_mapping(0), list)
    # Original continuos env
    original_env = env.env
    assert not original_env.is_discrete
    assert not hasattr(original_env, 'action_mapping')


@pytest.mark.parametrize('env_name',
                         [('env_wrapper_multiobs'),
                          ('env_all_wrappers'),
                          ])
def test_multiobs_wrapper(env_name, request):

    env = request.getfixturevalue(env_name)
    # Check attributes exist in wrapped env
    assert hasattr(
        env,
        'n') and hasattr(
        env,
        'ind_flat') and hasattr(
            env,
        'history')

    # Check history
    assert env.history == deque([])

    # Check observation space transformation
    original_shape = env.env.observation_space.shape[0]
    wrapped_shape = env.observation_space.shape[0]
    assert wrapped_shape == original_shape * env.n

    # Check reset obs
    obs, _ = env.reset()
    assert len(obs) == wrapped_shape
    for i in range(env.n - 1):
        # Check store same observation n times
        assert (obs[original_shape * i:original_shape *
                    (i + 1)] == obs[0:original_shape]).all()
        # Check history save same observation n times
        assert (env.history[i] ==
                env.history[i + 1]).all()

    # Check step obs
    a = env.action_space.sample()
    obs, _, _, _, _ = env.step(a)

    # Last observation must be different of the rest of them
    assert (obs[original_shape * (env.n - 1):]
            != obs[0:original_shape]).any()
    assert (env.history[0] !=
            env.history[-1]).any()


@ pytest.mark.parametrize('env_name',
                          [('env_wrapper_logger'), ('env_all_wrappers'), ])
def test_logger_wrapper(env_name, request):

    env = request.getfixturevalue(env_name)
    logger = env.file_logger
    env.reset()

    # Check CSV's have been created and linked in simulator correctly
    assert logger.log_progress_file == env.get_wrapper_attr(
        'workspace_path') + '/progress.csv'
    assert logger.log_file == env.episode_path + '/monitor.csv'

    tmp_log_file = logger.log_file

    # simulating short episode
    for _ in range(10):
        env.step(env.action_space.sample())
    env.reset()

    assert os.path.isfile(logger.log_progress_file)
    assert os.path.isfile(tmp_log_file)

    # If env is wrapped with normalize obs...
    if is_wrapped(env, NormalizeObservation):
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


def test_logger_activation(env_wrapper_logger):
    assert env_wrapper_logger.file_logger.flag
    env_wrapper_logger.deactivate_logger()
    assert not env_wrapper_logger.file_logger.flag
    env_wrapper_logger.activate_logger()
    assert env_wrapper_logger.file_logger.flag


def test_env_wrappers(env_all_wrappers):
    # CHECK ATTRIBUTES
    # MultiObjective
    assert hasattr(env_all_wrappers, 'reward_terms')
    # PreviousObservation
    assert hasattr(env_all_wrappers, 'previous_observation')
    assert hasattr(env_all_wrappers, 'previous_variables')
    # Datetime
    # IncrementalDiscrete
    assert hasattr(env_all_wrappers, 'current_setpoints')
    # Normalization
    assert hasattr(env_all_wrappers, 'unwrapped_observation')
    # Logger
    assert hasattr(env_all_wrappers, 'monitor_header')
    assert hasattr(env_all_wrappers, 'progress_header')
    assert hasattr(env_all_wrappers, 'logger')
    # Multiobs
    assert hasattr(env_all_wrappers, 'n')
    assert hasattr(env_all_wrappers, 'ind_flat')
    assert hasattr(env_all_wrappers, 'history')

    # GENERAL CHECKS
    # Check history multiobs is empty
    assert env_all_wrappers.history == deque([])
    # Start env
    obs, _ = env_all_wrappers.reset()
    # Check history has obs and any more
    assert len(env_all_wrappers.history) == env_all_wrappers.n
    assert (env_all_wrappers._get_obs() == obs).all()

    # obs should be normalized --> [0,1]
    # ...

    # Execute a short episode in order to check logger
    logger = env_all_wrappers.file_logger
    tmp_log_file = logger.log_file
    for _ in range(10):
        _, reward, _, _, info = env_all_wrappers.step(
            env_all_wrappers.action_space.sample())
        # reward should be a vector
        assert isinstance(reward, list)
    env_all_wrappers.reset()

    # Let's check if history has been completed succesfully
    assert len(env_all_wrappers.history) == env_all_wrappers.n
    assert isinstance(env_all_wrappers.history[0], np.ndarray)

    # check logger
    assert logger.log_progress_file == env_all_wrappers.get_wrapper_attr(
        'workspace_path') + '/progress.csv'
    assert logger.log_file == env_all_wrappers.episode_path + '/monitor.csv'
    assert os.path.isfile(logger.log_progress_file)
    assert os.path.isfile(tmp_log_file)
    assert os.path.isfile(tmp_log_file[:-4] + '_normalized.csv')
    # Close env
    env_all_wrappers.close()
