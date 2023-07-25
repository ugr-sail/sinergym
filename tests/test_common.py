import os
import shutil

import pytest
from opyplus import Epm, WeatherData

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
        env_5zone_continuous,
        env_wrapper_normalization,
        env_all_wrappers):
    # Check returns
    assert not common.is_wrapped(env_5zone_continuous, NormalizeObservation)
    assert common.is_wrapped(env_wrapper_normalization, NormalizeObservation)
    assert common.is_wrapped(env_all_wrappers, NormalizeObservation)


def test_unwrap_wrapper(
        env_5zone_continuous,
        env_wrapper_normalization,
        env_all_wrappers):
    # Check if env_wrapper_normalization unwrapped is env_5zone_continuous
    assert not hasattr(env_5zone_continuous, 'unwrapped_observation')
    assert hasattr(env_all_wrappers, 'unwrapped_observation')
    assert hasattr(env_wrapper_normalization, 'unwrapped_observation')
    assert hasattr(env_wrapper_normalization, 'env')
    env = common.unwrap_wrapper(
        env_wrapper_normalization,
        NormalizeObservation)
    assert not hasattr(env, 'unwrapped_observation')
    # Check if trying unwrap a not wrapped environment the result is None
    env = common.unwrap_wrapper(
        env_5zone_continuous,
        NormalizeObservation)
    assert env is None
    env = common.unwrap_wrapper(env_all_wrappers, NormalizeObservation)
    assert not hasattr(env, 'unwrapped_observation')


@pytest.mark.parametrize(
    'year,month,day,expected',
    [(1991, 2, 13, (20.0, 23.5)),
     (1991, 9, 9, (23.0, 26.0))]
)
def test_get_season_comfort_range(year, month, day, expected):
    output_range = common.get_season_comfort_range(year, month, day)
    assert output_range == expected
