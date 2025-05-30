import json
import os

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
        (2004, 7, 1, 2004, 6, 1, -2505600),  # Negative delta seconds test
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


def test_import_from_path_all_cases(tmp_path, monkeypatch):
    # Import a standard function from a standard module
    sqrt = common.import_from_path('math:sqrt')
    assert sqrt(4) == 2

    # Import a standard class from a standard module
    dt = common.import_from_path('datetime:datetime')
    assert dt.__name__ == 'datetime'

    # Invalid format (no ':')
    with pytest.raises(ValueError):
        common.import_from_path('mathsqrt')

    # Attribute does not exist in the module
    with pytest.raises(ImportError):
        common.import_from_path('math:not_a_function')

    # Try to import a non-existent attribute from a file
    file_content = '''
def foo():
    return "bar"
'''
    file_path = tmp_path / "my_module.py"
    file_path.write_text(file_content)
    with pytest.raises(ImportError):
        common.import_from_path(f"{file_path}:not_found")

    # Import a function from a file
    foo = common.import_from_path(f"{file_path}:foo")
    assert foo() == "bar"

    # Simulate spec_from_file_location returns None
    file_path2 = tmp_path / "my_module2.py"
    file_path2.write_text('')

    def fake_spec(*args, **kwargs):
        return None
    import importlib.util
    monkeypatch.setattr(importlib.util, "spec_from_file_location", fake_spec)
    with pytest.raises(ImportError):
        common.import_from_path(f"{file_path2}:foo")


def test_deep_update():
    # Simple case
    source = {'a': 1, 'b': 2}
    updates = {'b': 3, 'c': 4}
    result = common.deep_update(source, updates)
    assert result == {'a': 1, 'b': 3, 'c': 4}
    assert source == {'a': 1, 'b': 2}  # source should not be modified

    # Nested case
    source = {'a': {'x': 1, 'y': 2}, 'b': 2}
    updates = {'a': {'y': 3, 'z': 4}, 'c': 5}
    result = common.deep_update(source, updates)
    assert result == {'a': {'x': 1, 'y': 3, 'z': 4}, 'b': 2, 'c': 5}

    # Overwrite dict with simple value
    source = {'a': {'x': 1}, 'b': 2}
    updates = {'a': 5}
    result = common.deep_update(source, updates)
    assert result == {'a': 5, 'b': 2}

    # Empty updates
    source = {'a': 1, 'b': 2}
    updates = {}
    result = common.deep_update(source, updates)
    assert result == {'a': 1, 'b': 2}

    # Empty source
    source = {}
    updates = {'a': 1}
    result = common.deep_update(source, updates)
    assert result == {'a': 1}

    # Deepcopy: lists are independent
    source = {'a': [1, 2]}
    updates = {'a': [3, 4]}
    result = common.deep_update(source, updates)
    assert result == {'a': [3, 4]}
    result['a'].append(5)
    assert source == {'a': [1, 2]}
    assert updates == {'a': [3, 4]}


def test_create_environment_and_wrappers_with_normalize(env_5zone, tmp_path):
    from sinergym.utils.wrappers import NormalizeObservation

    # Prepare wrappers info as dict using NormalizeObservation (no params
    # required)
    wrapper_class = f"{
        NormalizeObservation.__module__}:{
        NormalizeObservation.__name__}"
    wrappers_info = {
        wrapper_class: {'mean': None,
                        'var': None,
                        'automatic_update': False}
    }

    # Test create_environment without wrappers
    env_params = env_5zone.get_wrapper_attr('to_dict')()
    env_params['seed'] = 33
    env = common.create_environment(
        env_id="Eplus-5zone-hot-continuous-v1",
        env_params=env_params,
        wrappers={})
    assert isinstance(env, gym.Env)
    assert env.get_wrapper_attr('seed') == 33

    # Test create_environment with NormalizeObservation wrapper
    env_wrapped = common.create_environment(
        env_id="Eplus-5zone-hot-continuous-v1",
        env_params=env_5zone.get_wrapper_attr('to_dict')(),
        wrappers=wrappers_info)

    assert isinstance(env_wrapped, gym.Wrapper)
    assert isinstance(env_wrapped, NormalizeObservation)
    assert env_wrapped.get_wrapper_attr('automatic_update') is False


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


def test_get_wrappers_info_and_apply_wrappers_info(env_5zone, tmp_path):
    # Use NormalizeObservation, which is importable and simple
    from sinergym.utils.wrappers import NormalizeObservation

    # Wrap environment with NormalizeObservation
    env = NormalizeObservation(env_5zone)

    # Patch has_wrapper_attr and get_wrapper_attr for test
    env.has_wrapper_attr = lambda attr: hasattr(env, attr)
    env.get_wrapper_attr = lambda attr: getattr(env, attr)

    # Test get_wrappers_info and YAML saving
    yaml_path = tmp_path / "wrappers_config.yaml"
    wrappers_dict = common.get_wrappers_info(env, path_to_save=str(yaml_path))
    assert isinstance(wrappers_dict, dict)
    assert any("NormalizeObservation" in k for k in wrappers_dict.keys())
    assert os.path.exists(yaml_path)

    # Test apply_wrappers_info from dict
    applied_env = common.apply_wrappers_info(env_5zone, wrappers_dict)
    assert isinstance(applied_env, gym.Wrapper)

    # Test apply_wrappers_info from YAML file
    applied_env_yaml = common.apply_wrappers_info(env_5zone, str(yaml_path))
    assert isinstance(applied_env_yaml, gym.Wrapper)


def test_parse_variables_settings(conf_5zone):

    assert isinstance(conf_5zone['variables'], dict)
    output = common.parse_variables_settings(conf_5zone['variables'])

    assert isinstance(output, dict)
    assert isinstance(list(output.keys())[0], str)
    assert isinstance(list(output.values())[0], tuple)
    assert len(list(output.values())[0]) == 2


def test_parse_meters_settings(conf_5zone):

    assert isinstance(conf_5zone['meters'], dict)
    output = common.parse_meters_settings(conf_5zone['meters'])

    assert isinstance(output, dict)
    assert isinstance(list(output.keys())[0], str)
    assert isinstance(list(output.values())[0], str)


def test_parse_actuators_settings(conf_5zone):

    assert isinstance(conf_5zone['actuators'], dict)
    output = common.parse_actuators_settings(conf_5zone['actuators'])

    assert isinstance(output, dict)
    assert isinstance(list(output.keys())[0], str)
    assert isinstance(list(output.values())[0], tuple)
    assert len(list(output.values())[0]) == 3


def test_convert_conf_to_env_parameters(conf_5zone):

    configurations = common.convert_conf_to_env_parameters(conf_5zone)

    # Check if environments are valid
    for env_id, env_kwargs in configurations.items():
        # Added TEST name in env_kwargs
        env_kwargs['env_name'] = 'PYTESTGYM'
        env = gym.make(env_id, **env_kwargs)
        env.reset()
        env.close()


def test_yaml_conf_exceptions(conf_5zone_exceptions):

    assert isinstance(conf_5zone_exceptions, list)
    for conf_5zone_exception in conf_5zone_exceptions:
        with pytest.raises((RuntimeError, ValueError)):
            common.convert_conf_to_env_parameters(conf_5zone_exception)


def test_ornstein_uhlenbeck_process(weather_data):
    df = weather_data.dataframe
    # Specify variability configuration for each desired column
    variability_conf = {
        'Dry Bulb Temperature': (1.0, 0.0, 24.0),
        'Wind Speed': (3.0, 0.0, 48.0)
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
