import shutil

import os
import pytest
from opyplus import Epm, WeatherData

from sinergym.utils.wrappers import NormalizeObservation
import sinergym.utils.common as common


def test_unwrap_wrapper(
        env_demo_continuous,
        env_wrapper_normalization,
        env_all_wrappers):
    # Check if env_wrapper_normalization unwrapped is env_demo_continuous
    assert not hasattr(env_demo_continuous, 'unwrapped_observation')
    assert hasattr(env_all_wrappers, 'unwrapped_observation')
    assert hasattr(env_wrapper_normalization, 'unwrapped_observation')
    assert hasattr(env_wrapper_normalization, 'env')
    env = common.unwrap_wrapper(
        env_wrapper_normalization,
        NormalizeObservation)
    assert not hasattr(env, 'unwrapped_observation')
    # Check if trying unwrap a not wrapped environment the result is None
    env = common.unwrap_wrapper(
        env_demo_continuous,
        NormalizeObservation)
    assert env is None
    env = common.unwrap_wrapper(env_all_wrappers, NormalizeObservation)
    assert not hasattr(env, 'unwrapped_observation')


def test_is_wrapped(
        env_demo_continuous,
        env_wrapper_normalization,
        env_all_wrappers):
    assert not common.is_wrapped(env_demo_continuous, NormalizeObservation)
    assert common.is_wrapped(env_wrapper_normalization, NormalizeObservation)
    assert common.is_wrapped(env_all_wrappers, NormalizeObservation)


def test_to_idf(epm):
    common.to_idf(epm, 'sinergym/data/buildings/TESTepm.idf')
    assert os.path.exists('sinergym/data/buildings/TESTepm.idf')


@pytest.mark.parametrize(
    'year,month,day,expected',
    [(1991, 2, 13, (20.0, 23.5)),
     (1991, 9, 9, (23.0, 26.0))]
)
def test_get_season_comfort_range(year, month, day, expected):
    output_range = common.get_season_comfort_range(year, month, day)
    assert output_range == expected


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


@pytest.mark.parametrize('sec_elapsed,expected_list',
                         [(2764800,
                           [1991,
                            2,
                            2,
                            0]),
                             (0,
                              [1991,
                               1,
                               1,
                               0]),
                             ((2764800 * 4) + (3600 * 10),
                              [1991,
                                 5,
                                 9,
                                 10]),
                          ])
def test_get_current_time_info(epm, sec_elapsed, expected_list):
    output = common.get_current_time_info(epm, sec_elapsed)
    print(output)
    assert isinstance(output, list)
    assert len(output) == 4
    assert output == expected_list


@ pytest.mark.parametrize(
    'variation',
    [
        (None),
        ((1, 0.0, 0.001)),
        ((5, 0.0, 0.01)),
        ((10, 0.0, 0.1)),
    ]
)
def test_create_variable_weather(variation, weather_data, weather_path):
    output = common.create_variable_weather(
        weather_data, weather_path, ['drybulb'], variation)
    if variation is None:
        assert output is None
    else:
        expected = weather_path.split('.epw')[0] + '_Random_' + str(
            variation[0]) + '_' + str(variation[1]) + '_' + str(variation[2]) + '.epw'
        assert output == expected
