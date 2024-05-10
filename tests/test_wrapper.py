import csv
import os
from collections import deque

import gymnasium as gym
import numpy as np
import pytest

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
    assert env.get_wrapper_attr('unwrapped_observation') is None

    # Initialize env
    obs, _ = env.reset()

    # Check observation normalization
    # ...
    # Check original observation recording
    assert env.get_wrapper_attr('unwrapped_observation') is not None

    # Simulation random step
    a = env.action_space.sample()
    obs, _, _, _, _ = env.step(a)

    # ...
    assert env.get_wrapper_attr('unwrapped_observation') is not None


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
    assert len(reward) == len(env.get_wrapper_attr('reward_terms'))


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
    assert len(env.get_wrapper_attr('previous_observation')) == 3
    assert len(previous_variable_names) == len(
        env.get_wrapper_attr('previous_observation'))
    # Check reset and np.zeros is added in obs as previous variables
    assert (env.get_wrapper_attr('previous_observation') == 0.0).all()
    obs1, _ = env.reset()
    original_obs1 = []
    for variable in env.get_wrapper_attr('previous_variables'):
        original_obs1.append(
            obs1[env.observation_variables.index(variable)])

    # Check step variables is added in obs previous variables
    action = env.action_space.sample()
    obs2, _, _, _, _ = env.step(action)

    # Original obs1 values should be previous variables for obs 2
    assert np.array_equal(
        original_obs1, obs2[-len(env.get_wrapper_attr('previous_variables')):])


def test_incremental_wrapper(env_wrapper_incremental):

    # Check initial values are initialized
    assert hasattr(env_wrapper_incremental, 'values_definition')
    assert len(env_wrapper_incremental.get_wrapper_attr('current_values')) == 2

    old_values = env_wrapper_incremental.get_wrapper_attr(
        'current_values').copy()
    # Check if action selected is applied correctly
    env_wrapper_incremental.reset()
    action = [-0.42, 0.3]
    rounded_action = [-0.5, 0.25]
    _, _, _, _, info = env_wrapper_incremental.step(action)
    assert env_wrapper_incremental.get_wrapper_attr(
        'current_values') == [old_values[i] + rounded_action[i] for i in range(len(old_values))]
    for i, (index, values) in enumerate(
            env_wrapper_incremental.get_wrapper_attr('values_definition').items()):
        assert env_wrapper_incremental.get_wrapper_attr(
            'current_values')[i] == info['action'][index]


@pytest.mark.parametrize('env_name',
                         [('env_discrete_wrapper_incremental'),
                          ('env_all_wrappers'),
                          ])
def test_discrete_incremental_wrapper(env_name, request):

    env = request.getfixturevalue(env_name)
    # Check initial setpoints values is initialized
    assert len(env.get_wrapper_attr('current_setpoints')) > 0
    # Check if action selected is applied correctly
    env.reset()
    action = 16
    _, _, _, _, info = env.step(action)
    assert (env.get_wrapper_attr('current_setpoints') == info['action']).all()
    # Check environment clip actions(
    for i in range(10):
        env.step(2)  # [1,0]
    assert env.unwrapped.action_space.contains(
        list(env.get_wrapper_attr('current_setpoints')))


def test_discretize_wrapper(env_wrapper_discretize):

    env = env_wrapper_discretize
    # Check is a discrete env and original env is continuous
    # Wrapped env
    assert env.get_wrapper_attr('is_discrete')
    assert env.action_space.n == 10
    assert isinstance(env.action_mapping(0), list)
    # Original continuos env
    original_env = env.env
    assert not original_env.get_wrapper_attr('is_discrete')
    assert not hasattr(original_env, 'action_mapping')


def test_normalize_observation_wrapper(env_wrapper_normalization):

    # Spaces
    env = env_wrapper_normalization
    assert not env.get_wrapper_attr('is_discrete')
    assert hasattr(env, 'unwrapped_observation')

    # Normalization calibration
    assert hasattr(env, 'mean')
    old_mean = env.get_wrapper_attr('mean').copy()
    assert hasattr(env, 'var')
    old_var = env.get_wrapper_attr('var').copy()
    assert len(env.get_wrapper_attr('mean')) == env.observation_space.shape[0]
    assert len(env.get_wrapper_attr('var')) == env.observation_space.shape[0]

    # reset
    obs, _ = env.reset()

    # Spaces
    assert (obs != env.get_wrapper_attr('unwrapped_observation')).any()
    assert env.observation_space.contains(
        env.get_wrapper_attr('unwrapped_observation'))

    # Calibration
    assert (old_mean != env.get_wrapper_attr('mean')).any()
    assert (old_var != env.get_wrapper_attr('var')).any()
    old_mean = env.get_wrapper_attr('mean').copy()
    old_var = env.get_wrapper_attr('var').copy()
    env.get_wrapper_attr('deactivate_update')()
    a = env.action_space.sample()
    env.step(a)
    assert (old_mean == env.get_wrapper_attr('mean')).all()
    assert (old_var == env.get_wrapper_attr('var')).all()
    env.get_wrapper_attr('activate_update')()
    env.step(a)
    assert (old_mean != env.get_wrapper_attr('mean')).any()
    assert (old_var != env.get_wrapper_attr('var')).any()
    env.get_wrapper_attr('set_mean')(old_mean)
    env.get_wrapper_attr('set_var')(old_var)
    assert (old_mean == env.get_wrapper_attr('mean')).all()
    assert (old_var == env.get_wrapper_attr('var')).all()

    # Check calibration as been saved as txt
    env.close()
    assert os.path.isfile(env.get_wrapper_attr('workspace_path') + '/mean.txt')
    assert os.path.isfile(env.get_wrapper_attr('workspace_path') + '/var.txt')
    # Check that txt has the same lines than observation space shape
    with open(env.get_wrapper_attr('workspace_path') + '/mean.txt', 'r') as f:
        lines = f.readlines()
        assert len(lines) == env.observation_space.shape[0]
    with open(env.get_wrapper_attr('workspace_path') + '/var.txt', 'r') as f:
        lines = f.readlines()
        assert len(lines) == env.observation_space.shape[0]


def test_normalize_action_wrapper(env_normalize_action_wrapper):

    env = env_normalize_action_wrapper
    assert not env.get_wrapper_attr('is_discrete')
    assert hasattr(env, 'real_space')
    assert hasattr(env, 'normalized_space')
    assert env.get_wrapper_attr(
        'normalized_space') != env.get_wrapper_attr('real_space')
    assert env.get_wrapper_attr('normalized_space') == env.action_space
    assert env.get_wrapper_attr('real_space') == env.unwrapped.action_space
    env.reset()
    action = env.action_space.sample()
    assert env.get_wrapper_attr('normalized_space').contains(action)
    _, _, _, _, info = env.step(action)
    assert env.unwrapped.action_space.contains(info['action'])


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
    assert env.get_wrapper_attr('history') == deque([])

    # Check observation space transformation
    original_shape = env.env.observation_space.shape[0]
    wrapped_shape = env.observation_space.shape[0]
    assert wrapped_shape == original_shape * env.get_wrapper_attr('n')

    # Check reset obs
    obs, _ = env.reset()
    assert len(obs) == wrapped_shape
    for i in range(env.get_wrapper_attr('n') - 1):
        # Check store same observation n times
        assert (obs[original_shape * i:original_shape *
                    (i + 1)] == obs[0:original_shape]).all()
        # Check history save same observation n times
        assert (env.get_wrapper_attr('history')[i] ==
                env.get_wrapper_attr('history')[i + 1]).all()

    # Check step obs
    a = env.action_space.sample()
    obs, _, _, _, _ = env.step(a)

    # Last observation must be different of the rest of them
    assert (obs[original_shape * (env.get_wrapper_attr('n') - 1):]
            != obs[0:original_shape]).any()
    assert (env.get_wrapper_attr('history')[0] !=
            env.get_wrapper_attr('history')[-1]).any()


@ pytest.mark.parametrize('env_name',
                          [('env_wrapper_logger'), ('env_all_wrappers'), ])
def test_logger_wrapper(env_name, request):

    env = request.getfixturevalue(env_name)
    logger = env.get_wrapper_attr('file_logger')
    env.reset()

    # Check CSV's have been created and linked in simulator correctly
    assert logger.log_progress_file == env.get_wrapper_attr(
        'workspace_path') + '/progress.csv'
    assert logger.log_file == env.get_wrapper_attr(
        'episode_path') + '/monitor.csv'

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


def test_reduced_observation_wrapper(env_wrapper_reduce_observation):

    env = env_wrapper_reduce_observation
    # Check that the original variable names has the removed varibles
    # but not in reduced variables
    original_observation_variables = env.env.get_wrapper_attr(
        'observation_variables')
    reduced_observation_variables = env.get_wrapper_attr(
        'observation_variables')
    removed_observation_variables = env.get_wrapper_attr(
        'removed_observation_variables')
    for removed_variable in removed_observation_variables:
        assert removed_variable in original_observation_variables
        assert removed_variable not in reduced_observation_variables

    # Check that the original observation space has a difference with the new
    original_shape = env.env.observation_space.shape[0]
    reduced_shape = env.observation_space.shape[0]
    assert reduced_shape == original_shape - len(removed_observation_variables)

    # Check reset return
    obs1, info1 = env.reset()
    assert len(obs1) == len(reduced_observation_variables)
    assert info1.get('removed_observation', False)
    assert len(info1['removed_observation']) == len(
        removed_observation_variables)
    for removed_variable_name, value in info1['removed_observation'].items():
        assert removed_variable_name in removed_observation_variables
        assert value is not None

    # Check step return
    action = env.action_space.sample()
    obs2, _, _, _, info2 = env.step(action)
    assert len(obs2) == len(reduced_observation_variables)
    assert info2.get('removed_observation', False)
    assert len(info2['removed_observation']) == len(
        removed_observation_variables)
    for removed_variable_name, value in info2['removed_observation'].items():
        assert removed_variable_name in removed_observation_variables
        assert value is not None


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
    # ReduceObservation
    assert hasattr(env_all_wrappers, 'removed_observation_variables')
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
