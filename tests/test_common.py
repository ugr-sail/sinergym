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
        env_all_wrappers):
    # Check returns
    assert not common.is_wrapped(env_5zone, NormalizeObservation)
    assert common.is_wrapped(env_all_wrappers, NormalizeObservation)


def test_unwrap_wrapper(
        env_5zone,
        env_all_wrappers):
    # Check if env_wrapper_normalization unwrapped is env_5zone
    assert not env_5zone.has_wrapper_attr('unwrapped_observation')
    assert env_all_wrappers.has_wrapper_attr('unwrapped_observation')
    env = common.unwrap_wrapper(
        env_all_wrappers,
        NormalizeObservation)
    assert not env.has_wrapper_attr('unwrapped_observation')
    # Check if trying unwrap a not wrapped environment the result is None
    env = common.unwrap_wrapper(
        env_5zone,
        NormalizeObservation)
    assert env is None


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


def test_json_conf_exceptions(conf_5zone_exceptions):

    assert isinstance(conf_5zone_exceptions, list)
    for conf_5zone_exception in conf_5zone_exceptions:
        with pytest.raises((RuntimeError, AssertionError)):
            common.convert_conf_to_env_parameters(conf_5zone_exception)


def test_ornstein_uhlenbeck_process(weather_data):
    df = weather_data.dataframe
    # Specify variability configuration for each desired column
    variability_conf = {
        'Dry Bulb Temperature': (1.0, 0.0, 0.001),
        'Wind Speed': (3.0, 0.0, 0.01)
    }
    # Calculate dataframe with noise
    noise = common.ornstein_uhlenbeck_process(
        data=df, variability_config=variability_conf)

    # Columns specified in variability_conf should be different
    assert (df['Dry Bulb Temperature'] !=
            noise['Dry Bulb Temperature']).any()
    assert (df['Wind Speed'] !=
            noise['Wind Speed']).any()
    # Columns not specified in variability_conf should be equal
    assert (df['Relative Humidity'] ==
            noise['Relative Humidity']).all()
    assert (df['Wind Direction'] ==
            noise['Wind Direction']).all()
