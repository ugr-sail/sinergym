import json

import gymnasium as gym
import pytest

import sinergym.utils.common as common
from sinergym.utils.wrappers import NormalizeObservation


@pytest.mark.parametrize(
    'st_year,st_mon,st_day,end_year,end_mon,end_day,expected',
    [
        (2000, 10, 1, 2000, 11, 1, 2764800),
        (2002, 1, 10, 2002, 2, 5, 2332800),
        # st_time=00:00:00 and ed_time=24:00:00
        (2021, 5, 5, 2021, 5, 5, 3600 * 24),
        (2004, 7, 1, 2004, 6, 1, -2505600),  # Negative delta secons test
    ]
)
def test_get_delta_seconds(
        st_year,
        st_mon,
        st_day,
        end_year,
        end_mon,
        end_day,
        expected):
    delta_sec = common.get_delta_seconds(
        st_year, st_mon, st_day, end_year, end_mon, end_day)
    assert isinstance(delta_sec, float)
    assert delta_sec == expected


def test_is_wrapped(
        env_5zone,
        env_wrapper_normalization,
        env_all_wrappers):
    # Check returns
    assert not common.is_wrapped(env_5zone, NormalizeObservation)
    assert common.is_wrapped(env_wrapper_normalization, NormalizeObservation)
    assert common.is_wrapped(env_all_wrappers, NormalizeObservation)


def test_unwrap_wrapper(
        env_5zone,
        env_wrapper_normalization,
        env_all_wrappers):
    # Check if env_wrapper_normalization unwrapped is env_5zone
    assert not hasattr(env_5zone, 'unwrapped_observation')
    assert hasattr(env_all_wrappers, 'unwrapped_observation')
    assert hasattr(env_wrapper_normalization, 'unwrapped_observation')
    assert hasattr(env_wrapper_normalization, 'env')
    env = common.unwrap_wrapper(
        env_wrapper_normalization,
        NormalizeObservation)
    assert not hasattr(env, 'unwrapped_observation')
    # Check if trying unwrap a not wrapped environment the result is None
    env = common.unwrap_wrapper(
        env_5zone,
        NormalizeObservation)
    assert env is None
    env = common.unwrap_wrapper(env_all_wrappers, NormalizeObservation)
    assert not hasattr(env, 'unwrapped_observation')


def test_json_to_variables(conf_5zone):

    assert isinstance(conf_5zone['variables'], dict)
    output = common.json_to_variables(conf_5zone['variables'])

    assert isinstance(output, dict)
    assert isinstance(list(output.keys())[0], str)
    assert isinstance(list(output.values())[0], tuple)
    assert len(list(output.values())[0]) == 2


def test_json_to_meters(conf_5zone):

    assert isinstance(conf_5zone['meters'], dict)
    output = common.json_to_meters(conf_5zone['meters'])

    assert isinstance(output, dict)
    assert isinstance(list(output.keys())[0], str)
    assert isinstance(list(output.values())[0], str)


def test_json_to_actuators(conf_5zone):

    assert isinstance(conf_5zone['actuators'], dict)
    output = common.json_to_actuators(conf_5zone['actuators'])

    assert isinstance(output, dict)
    assert isinstance(list(output.keys())[0], str)
    assert isinstance(list(output.values())[0], tuple)
    assert len(list(output.values())[0]) == 3


def test_convert_conf_to_env_parameters(conf_5zone):

    configurations = common.convert_conf_to_env_parameters(conf_5zone)

    # Check if environments are valid
    for env_id, env_kwargs in configurations.items():
        # Added TEST name in env_kwargs
        env_kwargs['env_name'] = 'TESTGYM'
        env = gym.make(env_id, **env_kwargs)
        env.reset()
        env.close()
