import csv
import os
from collections import deque

import gymnasium as gym
import numpy as np
import pytest

from sinergym.utils.common import is_wrapped
from sinergym.utils.constants import DEFAULT_5ZONE_DISCRETE_FUNCTION
from sinergym.utils.wrappers import *


def test_datetime_wrapper(env_demo):

    env = DatetimeWrapper(env=env_demo)

    observation_variables = env.observation_variables
    # Check if observation variables have been updated
    assert 'day_of_month' not in observation_variables
    assert 'month' not in observation_variables
    assert 'hour' not in observation_variables
    # Check new cyclic encoded variables are present
    assert 'month_sin' in observation_variables
    assert 'month_cos' in observation_variables
    assert 'hour_sin' in observation_variables
    assert 'hour_cos' in observation_variables
    assert 'day_sin' in observation_variables
    assert 'day_cos' in observation_variables
    # Check new returned observation values are valid
    env.reset()
    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    obs_dict = env.get_obs_dict(obs)
    assert (
        obs_dict['month_sin'] is not None
        and obs_dict['month_cos'] is not None
        and obs_dict['hour_sin'] is not None
        and obs_dict['hour_cos'] is not None
        and obs_dict['day_sin'] is not None
        and obs_dict['day_cos'] is not None
    )

    # Check exceptions
    # Delete hour variable from observation variables and observation space
    env_demo.observation_variables.remove('hour')
    with pytest.raises(ValueError):
        DatetimeWrapper(env=env_demo)


def test_previous_observation_wrapper(env_demo):

    env = PreviousObservationWrapper(
        env=env_demo,
        previous_variables=['heating_setpoint', 'cooling_setpoint', 'air_temperature'],
    )
    # Check that the original variable names with previous name added is
    # present
    previous_variable_names = [
        var for var in env.observation_variables if '_previous' in var
    ]

    # Check previous observation stored has the correct len and initial values
    assert len(env.get_wrapper_attr('previous_observation')) == 3
    assert len(previous_variable_names) == len(
        env.get_wrapper_attr('previous_observation')
    )
    # Check reset and np.zeros is added in obs as previous variables
    assert (env.get_wrapper_attr('previous_observation') == 0.0).all()
    obs1, _ = env.reset()
    original_obs1 = []
    for variable in env.get_wrapper_attr('previous_variables'):
        original_obs1.append(obs1[env.observation_variables.index(variable)])

    # Check step variables is added in obs previous variables
    action = env.action_space.sample()
    obs2, _, _, _, _ = env.step(action)

    # Original obs1 values should be previous variables for obs 2
    assert np.array_equal(
        original_obs1, obs2[-len(env.get_wrapper_attr('previous_variables')) :]
    )


def test_multiobs_wrapper(env_demo):

    env = MultiObsWrapper(env=env_demo, n=5, flatten=True)
    # Check attributes exist in wrapped env
    assert (
        env.has_wrapper_attr('n')
        and env.has_wrapper_attr('ind_flat')
        and env.has_wrapper_attr('history')
    )

    # Check history
    assert env.get_wrapper_attr('history') == deque([])

    # Check observation space transformation
    assert env.env.observation_space is not None
    assert env.observation_space is not None
    original_shape = env.env.observation_space.shape[0]  # type: ignore
    wrapped_shape = env.observation_space.shape[0]  # type: ignore
    assert wrapped_shape == original_shape * env.get_wrapper_attr('n')

    # Check reset obs
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert len(obs) == wrapped_shape
    for i in range(env.get_wrapper_attr('n') - 1):
        # Check store same observation n times
        assert (
            obs[original_shape * i : original_shape * (i + 1)] == obs[0:original_shape]
        ).all()
        # Check history save same observation n times
        assert (
            env.get_wrapper_attr('history')[i] == env.get_wrapper_attr('history')[i + 1]
        ).all()

    # Check step obs
    a = env.action_space.sample()
    obs, _, _, _, _ = env.step(a)

    # Last observation must be different of the rest of them
    assert (
        obs[original_shape * (env.get_wrapper_attr('n') - 1) :] != obs[0:original_shape]
    ).any()
    assert (
        env.get_wrapper_attr('history')[0] != env.get_wrapper_attr('history')[-1]
    ).any()

    # Check with flatten=False
    env = MultiObsWrapper(env=env_demo, n=5, flatten=False)
    obs, _ = env.reset()
    # Check obs shape
    assert len(obs) == env.get_wrapper_attr('n')
    assert len(obs[0]) == original_shape


def test_normalize_observation_wrapper(env_demo):
    env = NormalizeObservation(env=env_demo)

    # Check if new attributes have been created in environment
    assert env.has_wrapper_attr('unwrapped_observation')

    # Check initial values of that attributes
    assert env.get_wrapper_attr('unwrapped_observation') is None

    # Initialize env
    obs, _ = env.reset()

    # Check original observation recording
    assert env.get_wrapper_attr('unwrapped_observation') is not None
    # assert env.observation_space.contains(obs)
    assert env.observation_space.contains(env.get_wrapper_attr('unwrapped_observation'))

    # Check observation normalization id done correctly
    env.get_wrapper_attr('deactivate_update')()
    norm_obs = env.get_wrapper_attr('normalize')(
        env.get_wrapper_attr('unwrapped_observation')
    )
    assert all(norm_obs == obs)
    env.get_wrapper_attr('deactivate_update')()

    # Simulation random step
    a = env.action_space.sample()
    obs, _, _, _, _ = env.step(a)

    # Check original observation recording
    assert env.get_wrapper_attr('unwrapped_observation') is not None
    # assert env.observation_space.contains(obs)
    assert env.observation_space.contains(env.get_wrapper_attr('unwrapped_observation'))

    # Check observation normalization id done correctly
    env.get_wrapper_attr('deactivate_update')()
    norm_obs = env.get_wrapper_attr('normalize')(
        env.get_wrapper_attr('unwrapped_observation')
    )
    assert all(norm_obs == obs)
    env.get_wrapper_attr('deactivate_update')()


def test_normalize_observation_calibration(env_demo):

    # Spaces
    env = NormalizeObservation(env=env_demo)
    assert not env.get_wrapper_attr('is_discrete')
    assert env.has_wrapper_attr('unwrapped_observation')

    # Normalization calibration
    assert env.has_wrapper_attr('mean')
    old_mean = env.get_wrapper_attr('mean').copy()
    assert env.has_wrapper_attr('var')
    old_var = env.get_wrapper_attr('var').copy()
    assert env.has_wrapper_attr('count')
    old_count = env.get_wrapper_attr('count')
    assert (
        len(env.get_wrapper_attr('mean')) == env.observation_space.shape[0]  # type: ignore
    )
    assert (
        len(env.get_wrapper_attr('var')) == env.observation_space.shape[0]  # type: ignore
    )
    assert (
        isinstance(env.get_wrapper_attr('count'), float)
        and env.get_wrapper_attr('count') > 0
    )

    # reset
    obs, _ = env.reset()

    # Spaces
    assert (obs != env.get_wrapper_attr('unwrapped_observation')).any()
    assert env.observation_space.contains(env.get_wrapper_attr('unwrapped_observation'))

    # Calibration
    assert (old_mean != env.get_wrapper_attr('mean')).any()
    assert (old_var != env.get_wrapper_attr('var')).any()
    assert old_count != env.get_wrapper_attr('count')
    old_mean = env.get_wrapper_attr('mean').copy()
    old_var = env.get_wrapper_attr('var').copy()
    old_count = env.get_wrapper_attr('count')
    env.get_wrapper_attr('deactivate_update')()
    a = env.action_space.sample()
    env.step(a)
    assert (old_mean == env.get_wrapper_attr('mean')).all()
    assert (old_var == env.get_wrapper_attr('var')).all()
    assert old_count == env.get_wrapper_attr('count')
    env.get_wrapper_attr('activate_update')()
    env.step(a)
    assert (old_mean != env.get_wrapper_attr('mean')).any()
    assert (old_var != env.get_wrapper_attr('var')).any()
    assert old_count != env.get_wrapper_attr('count')
    env.get_wrapper_attr('set_mean')(old_mean)
    env.get_wrapper_attr('set_var')(old_var)
    env.get_wrapper_attr('set_count')(old_count)
    assert (old_mean == env.get_wrapper_attr('mean')).all()
    assert (old_var == env.get_wrapper_attr('var')).all()
    assert old_count == env.get_wrapper_attr('count')

    # Check calibration as been saved as txt
    env.close()
    assert os.path.isfile(env.get_wrapper_attr('workspace_path') + '/mean.txt')
    assert os.path.isfile(env.get_wrapper_attr('workspace_path') + '/var.txt')
    assert os.path.isfile(env.get_wrapper_attr('workspace_path') + '/count.txt')
    # Check that txt has the same lines than observation space shape
    with open(env.get_wrapper_attr('workspace_path') + '/mean.txt', 'r') as f:
        lines = f.readlines()
        assert len(lines) == env.observation_space.shape[0]  # type: ignore
    with open(env.get_wrapper_attr('workspace_path') + '/var.txt', 'r') as f:
        lines = f.readlines()
        assert len(lines) == env.observation_space.shape[0]  # type: ignore
    with open(env.get_wrapper_attr('workspace_path') + '/count.txt', 'r') as f:
        lines = f.readlines()
        assert len(lines) == 1


def test_normalize_observation_exceptions(env_demo):

    # Specify an unknown path file
    with pytest.raises(FileNotFoundError):
        NormalizeObservation(env=env_demo, mean='unknown_path')
    with pytest.raises(FileNotFoundError):
        NormalizeObservation(env=env_demo, var='unknown_path')
    # Specify an unknown value (nor str neither list)
    with pytest.raises(IndexError):
        NormalizeObservation(env=env_demo, mean=56)  # type: ignore
    with pytest.raises(IndexError):
        NormalizeObservation(env=env_demo, var=56)  # type: ignore
    # Specify a list with wrong shape
    with pytest.raises(ValueError):
        NormalizeObservation(env=env_demo, mean=[0.2, 0.1, 0.3])
    with pytest.raises(ValueError):
        NormalizeObservation(env=env_demo, var=[0.2, 0.1, 0.3])


def test_weatherforecasting_wrapper(env_demo):
    env = WeatherForecastingWrapper(env_demo, n=3, delta=1)
    # Check attributes exist in wrapped env
    assert (
        env.has_wrapper_attr('n')
        and env.has_wrapper_attr('delta')
        and env.has_wrapper_attr('columns')
        and env.has_wrapper_attr('forecast_data')
        and env.has_wrapper_attr('forecast_variability')
    )

    # Check observation variables and space transformation
    new_observation_variables = env.get_wrapper_attr('observation_variables')[
        -(len(env.get_wrapper_attr('columns') * env.get_wrapper_attr('n'))) :
    ]
    for i in range(1, env.get_wrapper_attr('n') + 1):
        for column in env.get_wrapper_attr('columns'):
            assert 'forecast_' + str(i) + '_' + column in new_observation_variables
    original_shape = env.env.observation_space.shape[0]  # type: ignore
    wrapped_shape = env.observation_space.shape[0]  # type: ignore
    assert wrapped_shape == (
        original_shape
        + (len(env.get_wrapper_attr('columns')) * env.get_wrapper_attr('n'))
    )

    # Check reset obs
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert len(obs) == wrapped_shape

    # Checks step obs
    a = env.action_space.sample()
    obs, _, _, _, _ = env.step(a)

    assert isinstance(obs, np.ndarray)
    assert len(obs) == wrapped_shape

    # Checks exceptional case 2
    env.forecast_data = env.get_wrapper_attr('forecast_data').head(2)
    a = env.action_space.sample()
    obs, _, _, _, _ = env.step(a)

    assert isinstance(obs, np.ndarray)
    assert len(obs) == wrapped_shape

    # Checks exceptional case 1
    env.forecast_data = env.get_wrapper_attr('forecast_data').head(1)
    a = env.action_space.sample()
    obs, _, _, _, _ = env.step(a)

    assert isinstance(obs, np.ndarray)
    assert len(obs) == wrapped_shape


def test_weatherforecasting_wrapper_forecastdata(env_demo):
    env = WeatherForecastingWrapper(
        env_demo,
        n=3,
        delta=1,
        forecast_variability={
            'Dry Bulb Temperature': (1.0, 0.0, 24.0),
            'Wind Speed': (3.0, 0.0, 48.0),
        },
    )

    # Get original weather_data
    original_weather_data = Weather()
    original_weather_data.read(env.get_wrapper_attr('weather_path'))
    assert original_weather_data.dataframe is not None
    original_weather_data = original_weather_data.dataframe.loc[
        :, ['Month', 'Day', 'Hour'] + env.get_wrapper_attr('columns')
    ]
    # Until first reset, forecast data is None
    assert env.get_wrapper_attr('forecast_data') is None

    # Check that reset create and apply noise in forecast data from original
    # weather data.
    env.reset()
    assert env.get_wrapper_attr('forecast_data') is not None
    noised_weather_data = env.get_wrapper_attr('forecast_data')

    assert not noised_weather_data['Dry Bulb Temperature'].equals(
        original_weather_data['Dry Bulb Temperature']
    )
    assert not noised_weather_data['Wind Speed'].equals(
        original_weather_data['Wind Speed']
    )

    # Columns without noise it should be the same than original weather data
    columns_to_be_same = [
        col
        for col in original_weather_data.columns
        if col not in ['Dry Bulb Temperature', 'Wind Speed']
    ]
    assert noised_weather_data[columns_to_be_same].equals(
        original_weather_data[columns_to_be_same]
    )


def test_weatherforecasting_wrapper_exceptions(env_demo):
    # Specify a tuple with wrong shape (must be 3)
    with pytest.raises(ValueError):
        env = WeatherForecastingWrapper(
            env_demo,
            n=3,
            delta=1,
            forecast_variability={  # type: ignore
                'Dry Bulb Temperature': (1.0, 0.0),
                'Wind Speed': (3.0, 0.0),
            },
        )
        env.reset()

    # Specify a key that it isn't in `columns`
    with pytest.raises(ValueError):
        WeatherForecastingWrapper(
            env_demo,
            n=3,
            delta=1,
            columns=[
                'Dry Bulb Temperature',
                'Relative Humidity',
                'Wind Direction',
                'Wind Speed',
                'Direct Normal Radiation',
                'Diffuse Horizontal Radiation',
            ],
            forecast_variability={
                'Dry Bulb Temperature': (1.0, 0.0, 24.0),
                'Wind Speed': (3.0, 0.0, 48.0),
                'Not in columns': (3.0, 0.0, 48.0),
            },
        )


def test_energycost_wrapper(env_demo):
    env = EnergyCostWrapper(
        env_demo,
        energy_cost_data_path='/workspaces/sinergym/sinergym/data/energy_cost/PVPC_active_energy_billing_Iberian_Peninsula_2023.csv',
    )

    # Check attributes exist in wrapped env
    assert (
        env.has_wrapper_attr('energy_cost_variability')
        and env.has_wrapper_attr('energy_cost_data_path')
        and env.has_wrapper_attr('energy_cost_data')
    )

    # Check observation space transformation
    original_shape = env.env.observation_space.shape[0]  # type: ignore
    wrapped_shape = env.observation_space.shape[0]  # type: ignore
    assert 'energy_cost' in env.get_wrapper_attr('observation_variables')
    assert wrapped_shape == (original_shape + 1)

    # Check reset obs
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert len(obs) == wrapped_shape

    # Checks step obs
    a = env.action_space.sample()
    obs, _, _, _, _ = env.step(a)

    assert isinstance(obs, np.ndarray)
    assert len(obs) == wrapped_shape


def test_energycost_wrapper_energycostdata(env_demo):
    env = EnergyCostWrapper(
        env_demo,
        energy_cost_data_path='/workspaces/sinergym/sinergym/data/energy_cost/PVPC_active_energy_billing_Iberian_Peninsula_2023.csv',
        energy_cost_variability=(1.0, 0.0, 24.0),  # type: ignore
    )

    # Get and preprocess manually original cost data
    original_energy_cost_data = pd.read_csv(env.energy_cost_data_path, sep=';')
    original_energy_cost_data['datetime'] = pd.to_datetime(
        original_energy_cost_data['datetime'], utc=True
    )
    original_energy_cost_data['datetime'] += pd.DateOffset(hours=1)

    original_energy_cost_data['Month'] = original_energy_cost_data['datetime'].dt.month
    original_energy_cost_data['Day'] = original_energy_cost_data['datetime'].dt.day
    original_energy_cost_data['Hour'] = original_energy_cost_data['datetime'].dt.hour
    original_energy_cost_data = original_energy_cost_data[
        ['Month', 'Day', 'Hour', 'value']
    ]
    # Cost data attribute should be None until reset.
    assert env.get_wrapper_attr('energy_cost_data') is None

    # Check that after the reset the energy cost data is recreated.
    env.reset()
    assert env.get_wrapper_attr('energy_cost_data') is not None
    noised_energy_cost_data = env.get_wrapper_attr('energy_cost_data')

    assert not noised_energy_cost_data['value'].equals(
        original_energy_cost_data['value']
    )

    # Columns not affected by noise should be the same
    columns_to_be_same = [
        col for col in original_energy_cost_data.columns if col not in ['value']
    ]
    assert noised_energy_cost_data[columns_to_be_same].equals(
        original_energy_cost_data[columns_to_be_same]
    )


def test_energycost_wrapper_exceptions(env_demo):
    # Specify a tuple with wrong shape (must be 3)
    with pytest.raises(ValueError):
        env = EnergyCostWrapper(
            env_demo,
            energy_cost_data_path='/workspaces/sinergym/sinergym/data/energy_cost/PVPC_active_energy_billing_Iberian_Peninsula_2023.csv',
            energy_cost_variability=(1.0, 0.0),  # type: ignore
        )
        env.reset()

    # Specify a energy cost file that doesn't exist
    with pytest.raises(FileNotFoundError):
        env = EnergyCostWrapper(env_demo, energy_cost_data_path='non-existent-file.csv')
        env.reset()


def test_incremental_wrapper(env_demo):

    env = IncrementalWrapper(
        env=env_demo,
        incremental_variables_definition={
            'heating_setpoint': (2.0, 0.5),
            'cooling_setpoint': (1.0, 0.25),
        },
        initial_values=[21.0, 25.0],
    )

    # Check initial values are initialized
    assert env.has_wrapper_attr('values_definition')
    assert len(env.get_wrapper_attr('current_values')) == 2

    old_values = env.get_wrapper_attr('current_values').copy()
    # Check if action selected is applied correctly
    env.reset()
    action = np.array([-0.42, 0.3], dtype=np.float32)
    rounded_action = [-0.5, 0.25]
    _, _, _, _, info = env.step(action)
    assert (
        env.get_wrapper_attr('current_values')
        == [old_values[i] + rounded_action[i] for i in range(len(old_values))]
    ).all()
    for i, (index, _) in enumerate(env.get_wrapper_attr('values_definition').items()):
        assert env.get_wrapper_attr('current_values')[i] == info['action'][index]


def test_incremental_exceptions(env_demo):

    # Discrete environment exception
    env_discrete = DiscretizeEnv(
        env_demo,
        discrete_space=gym.spaces.Discrete(10),
        action_mapping=DEFAULT_5ZONE_DISCRETE_FUNCTION,
    )
    with pytest.raises(TypeError):
        IncrementalWrapper(
            env=env_discrete,
            incremental_variables_definition={
                'heating_setpoint': (2.0, 0.5),
                'cooling_setpoint': (1.0, 0.25),
            },
            initial_values=[21.0, 25.0],
        )

    # Unknown variable exception
    with pytest.raises(ValueError):
        IncrementalWrapper(
            env=env_demo,
            incremental_variables_definition={
                'heating_setpoint': (2.0, 0.5),
                'cooling_setpoint': (1.0, 0.25),
                'Unknown_Variable': (1.0, 0.25),
            },
            initial_values=[21.0, 25.0, 3.0],
        )

    # Wrong initial values exception
    with pytest.raises(ValueError):
        IncrementalWrapper(
            env=env_demo,
            incremental_variables_definition={
                'heating_setpoint': (2.0, 0.5),
                'cooling_setpoint': (1.0, 0.25),
            },
            initial_values=[21.0, 25.0, 3.0],
        )


def test_discrete_incremental_wrapper(env_demo):

    # Check environment is continuous
    assert not env_demo.is_discrete
    env = DiscreteIncrementalWrapper(
        env=env_demo, initial_values=[21.0, 25.0], delta_temp=2, step_temp=0.5
    )
    # Check initial setpoints values is initialized
    assert len(env.get_wrapper_attr('current_setpoints')) > 0
    # Check if action selected is applied correctly
    env.reset()
    action = 16
    _, _, _, _, info = env.step(action)
    assert (env.get_wrapper_attr('current_setpoints') == info['action']).all()
    # Check environment clip actions(
    for _ in range(10):
        env.step(2)  # [1,0]
    assert env.unwrapped.action_space.contains(
        env.get_wrapper_attr('current_setpoints')
    )
    # Check environment is discrete now
    assert env.is_discrete


def test_discrete_incremental_exceptions(env_demo):

    # Discrete environment exception
    env_discrete = DiscretizeEnv(
        env_demo,
        discrete_space=gym.spaces.Discrete(10),
        action_mapping=DEFAULT_5ZONE_DISCRETE_FUNCTION,
    )
    with pytest.raises(TypeError):
        DiscreteIncrementalWrapper(
            env=env_discrete, initial_values=[21.0, 25.0], delta_temp=2, step_temp=0.5
        )
    # Number of initial values different than number of action_variables
    with pytest.raises(ValueError):
        DiscreteIncrementalWrapper(
            env=env_demo, initial_values=[21.0, 25.0, 3.0], delta_temp=2, step_temp=0.5
        )


def test_discretize_wrapper(env_demo):

    assert not env_demo.is_discrete
    env = DiscretizeEnv(
        env=env_demo,
        discrete_space=gym.spaces.Discrete(10),
        action_mapping=DEFAULT_5ZONE_DISCRETE_FUNCTION,
    )
    # Check is a discrete env and original env is continuous
    # Wrapped env
    assert env.get_wrapper_attr('is_discrete')
    assert env.action_space.n == 10  # type: ignore
    assert isinstance(env.action_mapping(0), np.ndarray)
    # Original continuos env
    original_env = env.env
    assert not original_env.get_wrapper_attr('is_discrete')
    assert not original_env.has_wrapper_attr('action_mapping')


def test_normalize_action_wrapper(env_demo):

    env = NormalizeAction(env=env_demo)
    # Check if new attributes have been created in environment
    assert not env.get_wrapper_attr('is_discrete')
    assert env.has_wrapper_attr('real_space')
    assert env.has_wrapper_attr('normalized_space')
    assert env.get_wrapper_attr('normalized_space') != env.get_wrapper_attr(
        'real_space'
    )
    assert env.get_wrapper_attr('normalized_space') == env.action_space
    assert env.get_wrapper_attr('real_space') == env.unwrapped.action_space
    env.reset()
    action = env.action_space.sample()
    assert env.get_wrapper_attr('normalized_space').contains(action)
    _, _, _, _, info = env.step(action)
    assert env.unwrapped.action_space.contains(
        np.array(info['action'], dtype=np.float32)
    )


def test_deltatemp_wrapper(env_datacenter):

    old_observation_space = deepcopy(
        env_datacenter.get_wrapper_attr('observation_space')
    )
    old_observation_variables = deepcopy(
        env_datacenter.get_wrapper_attr('observation_variables')
    )

    assert not env_datacenter.has_wrapper_attr('delta_temperatures')
    assert not env_datacenter.has_wrapper_attr('delta_setpoints')

    # Same setpoint values than temperature values
    env = DeltaTempWrapper(
        env_datacenter,
        temperature_variables=[
            'west_zone_air_temperature',
            'east_zone_air_temperature',
        ],
        setpoint_variables=['cooling_setpoint'],
    )

    # Check attributes exist in wrapped env
    assert env.has_wrapper_attr('delta_temperatures')
    assert env.has_wrapper_attr('delta_setpoints')
    # Check new space
    assert env.observation_space.shape is not None
    assert env.observation_space.shape[0] == old_observation_space.shape[0] + 2
    assert (
        len(env.get_wrapper_attr('observation_variables'))
        == len(old_observation_variables) + 2
    )
    assert env.get_wrapper_attr('observation_variables')[-2:] == [
        'delta_' + env.delta_temperatures[0],
        'delta_' + env.delta_temperatures[1],
    ]

    # Check observation values
    obs, _ = env.reset()
    obs_dict = env.get_obs_dict(obs)
    assert len(obs) == len(env.get_wrapper_attr('observation_variables'))
    assert (
        obs_dict['delta_west_zone_air_temperature']
        == obs_dict['west_zone_air_temperature'] - obs_dict['cooling_setpoint']
    )
    assert (
        obs_dict['delta_east_zone_air_temperature']
        == obs_dict['east_zone_air_temperature'] - obs_dict['cooling_setpoint']
    )

    a = env.action_space.sample()
    obs, _, _, _, _ = env.step(a)
    obs_dict = env.get_obs_dict(obs)
    assert (
        obs_dict['delta_west_zone_air_temperature']
        == obs_dict['west_zone_air_temperature'] - obs_dict['cooling_setpoint']
    )
    assert (
        obs_dict['delta_east_zone_air_temperature']
        == obs_dict['east_zone_air_temperature'] - obs_dict['cooling_setpoint']
    )

    env.close()


def test_normalize_action_exceptions(env_demo):
    # Environment cannot be discrete
    env_discrete = DiscretizeEnv(
        env_demo,
        discrete_space=gym.spaces.Discrete(10),
        action_mapping=DEFAULT_5ZONE_DISCRETE_FUNCTION,
    )
    with pytest.raises(TypeError):
        NormalizeAction(env=env_discrete)


def test_multiobjective_wrapper(env_demo):
    env = MultiObjectiveReward(
        env=env_demo, reward_terms=['energy_term', 'comfort_term']
    )
    assert env.has_wrapper_attr('reward_terms')
    env.reset()
    action = env.action_space.sample()
    _, reward, _, _, _ = env.step(action)
    assert isinstance(reward, list)
    assert len(reward) == len(env.get_wrapper_attr('reward_terms'))


def test_logger_wrapper(env_demo):

    assert not env_demo.has_wrapper_attr('data_logger')
    env = LoggerWrapper(env=env_demo)
    assert env.has_wrapper_attr('data_logger')
    logger = env.get_wrapper_attr('data_logger')
    # Check logger is empty
    assert len(logger.observations) == 0
    assert len(logger.actions) == 0
    assert len(logger.rewards) == 0
    assert len(logger.infos) == 0
    assert len(logger.terminateds) == 0
    assert len(logger.truncateds) == 0
    assert len(logger.custom_metrics) == 0
    assert logger.interactions == 0

    env.reset()
    # Reset is not logged
    assert len(logger.observations) == 1
    assert len(logger.actions) == 0
    assert len(logger.rewards) == 0
    assert len(logger.infos) == 1
    assert len(logger.terminateds) == 0
    assert len(logger.truncateds) == 0
    assert len(logger.custom_metrics) == 0
    assert logger.interactions == 0

    # Make 3 steps
    for _ in range(3):
        a = env.action_space.sample()
        env.step(a)

    # Check that the logger has stored the data
    assert len(logger.observations) == 4
    assert len(logger.actions) == 3
    assert len(logger.rewards) == 3
    assert len(logger.infos) == 4
    assert len(logger.terminateds) == 3
    assert len(logger.truncateds) == 3
    assert len(logger.custom_metrics) == 0
    assert logger.interactions == 3

    # Check summary data is done
    summary = env.get_wrapper_attr('get_episode_summary')()
    assert env.get_wrapper_attr('summary_metrics') == list(summary.keys())
    assert isinstance(summary, dict)
    assert len(summary) > 0
    assert summary.get('mean_reward', False) and summary.get('std_reward', False)
    assert summary['mean_reward'] == np.mean(logger.rewards)
    assert summary['std_reward'] == np.std(logger.rewards)

    # Check if reset method reset logger data too
    env.reset()
    assert len(logger.observations) == 1
    assert len(logger.actions) == 0
    assert len(logger.rewards) == 0
    assert len(logger.infos) == 1
    assert len(logger.terminateds) == 0
    assert len(logger.truncateds) == 0
    assert len(logger.custom_metrics) == 0
    assert logger.interactions == 0

    # Make 3 steps
    for _ in range(3):
        a = env.action_space.sample()
        env.step(a)

    # Check closing environment reset logger
    env.close()

    assert len(logger.observations) == 0
    assert len(logger.actions) == 0
    assert len(logger.rewards) == 0
    assert len(logger.infos) == 0
    assert len(logger.terminateds) == 0
    assert len(logger.truncateds) == 0
    assert len(logger.custom_metrics) == 0
    assert logger.interactions == 0


def test_custom_loggers(env_demo, custom_logger_wrapper):

    assert not env_demo.has_wrapper_attr('data_logger')
    env = custom_logger_wrapper(env=env_demo)
    assert env.has_wrapper_attr('data_logger')
    assert env.has_wrapper_attr('custom_variables')
    assert len(env.get_wrapper_attr('custom_variables')) > 0
    logger = env.get_wrapper_attr('data_logger')
    env.reset()

    # Make 3 steps
    for _ in range(3):
        a = env.action_space.sample()
        env.step(a)

    # Check that the logger has stored the data (custom metrics too)
    assert len(logger.observations) == 4
    assert len(logger.actions) == 3
    assert len(logger.rewards) == 3
    assert len(logger.infos) == 4
    assert len(logger.terminateds) == 3
    assert len(logger.truncateds) == 3
    assert len(logger.custom_metrics) == 3
    assert logger.interactions == 3

    # Check summary data is done
    summary = env.get_wrapper_attr('get_episode_summary')()
    assert env.get_wrapper_attr('summary_variables') == list(summary.keys())
    assert isinstance(summary, dict)
    assert len(summary) > 0

    # Check if reset method reset logger data too (custom_metrics too)
    env.reset()
    assert len(logger.observations) == 1
    assert len(logger.actions) == 0
    assert len(logger.rewards) == 0
    assert len(logger.infos) == 1
    assert len(logger.terminateds) == 0
    assert len(logger.truncateds) == 0
    assert len(logger.custom_metrics) == 0
    assert logger.interactions == 0

    # Make 3 steps
    for _ in range(3):
        a = env.action_space.sample()
        env.step(a)

    # Check closing environment reset logger
    env.close()

    assert len(logger.observations) == 0
    assert len(logger.actions) == 0
    assert len(logger.rewards) == 0
    assert len(logger.infos) == 0
    assert len(logger.terminateds) == 0
    assert len(logger.truncateds) == 0
    assert len(logger.custom_metrics) == 0
    assert logger.interactions == 0


@pytest.mark.parametrize('env_name', [('env_demo'), ('env_5zone_stochastic')])
def test_CSVlogger_wrapper(env_name, request):
    env = request.getfixturevalue(env_name)

    env = CSVLogger(env=LoggerWrapper(env=NormalizeObservation(env=env)))
    # Check progress CSV path
    assert (
        env.get_wrapper_attr('progress_file_path')
        == env.get_wrapper_attr('workspace_path') + '/progress.csv'
    )

    # First reset should not create new files
    env.reset()

    # Assert logger files are not created
    assert not os.path.isfile(env.get_wrapper_attr('progress_file_path'))
    assert not os.path.isfile(env.get_wrapper_attr('weather_variability_config_path'))
    assert not os.path.isdir(env.get_wrapper_attr('episode_path') + '/monitor')

    # simulating short episode
    for _ in range(10):
        env.step(env.action_space.sample())
    episode_path = env.get_wrapper_attr('episode_path')
    env.reset()

    # Now logger files about first episode should be created
    assert os.path.isfile(env.get_wrapper_attr('progress_file_path'))
    assert os.path.isdir(episode_path + '/monitor')
    # Check csv files has been generated in monitor
    assert len(os.listdir(episode_path + '/monitor')) > 0

    # Read CSV files and check row nums
    with open(
        env.get_wrapper_attr('progress_file_path'), mode='r', newline=''
    ) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # Header row and episode summary
        assert len(list(reader)) == 2
    if env_name == 'env_demo':
        # File not exists
        assert not os.path.isfile(
            env.get_wrapper_attr('weather_variability_config_path')
        )
    else:
        with open(
            env.get_wrapper_attr('weather_variability_config_path'),
            mode='r',
            newline='',
        ) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            # Header row and episode config
            assert len(list(reader)) == 2
    # Check csv in monitor is created correctly (only check with observations)
    with open(
        episode_path + '/monitor/observations.csv', mode='r', newline=''
    ) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # Header row, reset and 10 steps (12)
        assert len(list(reader)) == 12

    # If env is wrapped with normalize obs...
    if is_wrapped(env, NormalizeObservation):
        assert os.path.isfile(episode_path + '/monitor/normalized_observations.csv')
    env.close()


def test_CSVlogger_custom_logger(env_demo, custom_logger_wrapper):
    env = CSVLogger(env=custom_logger_wrapper(env=env_demo))

    # First reset should not create new files
    env.reset()

    # simulating short episode
    for _ in range(10):
        env.step(env.action_space.sample())
    episode_path = env.get_wrapper_attr('episode_path')
    env.reset()

    # Read CSV files and check row nums
    with open(
        env.get_wrapper_attr('progress_file_path'), mode='r', newline=''
    ) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # Header row and episode summary (2)
        assert len(list(reader)) == 2
    # Check csv in monitor is created correctly (only check with observations)
    with open(
        episode_path + '/monitor/custom_metrics.csv', mode='r', newline=''
    ) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # Header row and 10 steps (11)
        assert len(list(reader)) == 11

    # Since we have not normalized observation, this file should not exist
    assert not os.path.isfile(episode_path + '/monitor/normalized_observations.csv')

    env.close()


def test_logger_exceptions(env_demo):
    # Use a Logger without previous BaseLoggerWrapper child class should raise
    # exception
    with pytest.raises(ValueError):
        CSVLogger(env=env_demo)


def test_reduced_observation_wrapper(env_demo):

    env = ReduceObservationWrapper(
        env=env_demo,
        obs_reduction=['outdoor_temperature', 'outdoor_humidity', 'air_temperature'],
    )
    # Check that the original variable names has the removed variables
    # but not in reduced variables
    original_observation_variables = env.env.get_wrapper_attr('observation_variables')
    reduced_observation_variables = env.get_wrapper_attr('observation_variables')
    removed_observation_variables = env.get_wrapper_attr(
        'removed_observation_variables'
    )
    for removed_variable in removed_observation_variables:
        assert removed_variable in original_observation_variables
        assert removed_variable not in reduced_observation_variables

    # Check that the original observation space has a difference with the new
    original_shape = env.env.observation_space.shape[0]  # type: ignore
    reduced_shape = env.observation_space.shape[0]  # type: ignore
    assert reduced_shape == original_shape - len(removed_observation_variables)

    # Check reset return
    obs1, _ = env.reset()
    assert len(obs1) == len(reduced_observation_variables)

    # Check step return
    action = env.action_space.sample()
    obs2, _, _, _, _ = env.step(action)
    assert len(obs2) == len(reduced_observation_variables)


def test_reduced_observation_exceptions(env_demo):

    # We force an unknown variable to be removed
    with pytest.raises(ValueError):
        ReduceObservationWrapper(
            env=env_demo,
            obs_reduction=[
                'outdoor_temperature',
                'outdoor_humidity',
                'air_temperature',
                'unknown_variable',
            ],
        )


def test_env_wrappers(env_all_wrappers):
    # CHECK ATTRIBUTES
    # MultiObjective
    assert env_all_wrappers.has_wrapper_attr('reward_terms')
    # PreviousObservation
    assert env_all_wrappers.has_wrapper_attr('previous_observation')
    assert env_all_wrappers.has_wrapper_attr('previous_variables')
    # Datetime
    # IncrementalDiscrete
    assert env_all_wrappers.has_wrapper_attr('current_setpoints')
    # Normalization
    assert env_all_wrappers.has_wrapper_attr('unwrapped_observation')
    # Logger
    assert env_all_wrappers.has_wrapper_attr('data_logger')
    # CSVLogger
    assert env_all_wrappers.has_wrapper_attr('progress_file_path')
    # ReduceObservation
    assert env_all_wrappers.has_wrapper_attr('removed_observation_variables')
    # Multi obs
    assert env_all_wrappers.has_wrapper_attr('n')
    assert env_all_wrappers.has_wrapper_attr('ind_flat')
    assert env_all_wrappers.has_wrapper_attr('history')

    # GENERAL CHECKS
    # Check history multi obs is empty
    assert env_all_wrappers.history == deque([])
    # Start env
    obs, _ = env_all_wrappers.reset()
    # Check history has obs and any more
    assert len(env_all_wrappers.history) == env_all_wrappers.n
    assert (env_all_wrappers._get_obs() == obs).all()

    # Execute a short episode in order to check logger
    for _ in range(10):
        _, reward, _, _, _ = env_all_wrappers.step(
            env_all_wrappers.action_space.sample()
        )
        # reward should be a vector
        assert isinstance(reward, list)
    env_all_wrappers.reset()

    # Let's check if history has been completed successfully
    assert len(env_all_wrappers.history) == env_all_wrappers.n
    assert isinstance(env_all_wrappers.history[0], np.ndarray)

    # Close env
    env_all_wrappers.close()


def test_scheduled_context_wrapper(env_5zone):
    """Test ScheduledContextWrapper: initialization, updates, and edge cases."""
    scheduled_context = {
        '01-15 10': [0.8, 0.6],
        '01-20 14': [0.5, 0.7],
        '02-10 09': [0.9, 0.4],
    }

    # Test initialization
    env = ScheduledContextWrapper(env=env_5zone, scheduled_context=scheduled_context)
    assert env.get_wrapper_attr('scheduled_context') == scheduled_context

    # Test context updates at scheduled times
    env.reset()
    action = env.action_space.sample()
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(action)
        dt_str = f"{info['month']:02d}-{info['day']:02d} {info['hour']:02d}"
        if dt_str in scheduled_context:
            assert env.get_wrapper_attr('last_context') == scheduled_context[dt_str]
            assert len(env.get_wrapper_attr('last_context')) == 2

    env.close()

    # Test no matches (dates outside run period)
    env2 = ScheduledContextWrapper(
        env=env_5zone, scheduled_context={'12-15 10': [0.8, 0.6]}
    )
    env2.reset()
    initial_context = env2.get_wrapper_attr('last_context')
    action = env2.action_space.sample()
    for _ in range(50):
        _, _, terminated, truncated, _ = env2.step(action)
        assert env2.get_wrapper_attr('last_context') == initial_context
        if terminated or truncated:
            break
    env2.close()


# ============================================================================
# Tests for ProbabilisticContextWrapper
# ============================================================================


def test_probabilistic_context_wrapper_basic(env_5zone):
    """Test ProbabilisticContextWrapper: initialization, float/list modes, reset, clipping."""
    context_space = gym.spaces.Box(
        low=np.array([0.3, 0.3], dtype=np.float32),
        high=np.array([0.9, 0.9], dtype=np.float32),
        shape=(2,),
        dtype=np.float32,
    )

    # Test initialization with float probability
    env = ProbabilisticContextWrapper(
        env=env_5zone,
        context_space=context_space,
        update_probability=0.1,
        global_value=False,
        delta_update=False,
    )
    assert env.get_wrapper_attr('update_probability') == 0.1
    assert env.get_wrapper_attr('prob_per_variable') is False

    # Test initialization with list probability
    env2 = ProbabilisticContextWrapper(
        env=env_5zone,
        context_space=context_space,
        update_probability=[0.2, 0.3],
        global_value=False,
        delta_update=False,
    )
    assert env2.get_wrapper_attr('prob_per_variable') is True
    assert len(env2.get_wrapper_attr('update_probability')) == 2

    # Test initial context clipping (initial_context=[1.0, 0.5] should clip to [0.9, 0.5])
    env.reset()
    assert context_space.contains(env.current_context)
    assert env.current_context[0] == 0.9  # Clipped from 1.0
    assert env.current_context[1] == 0.5  # Within range

    # Test reset reinitializes context
    context1 = env.current_context.copy()
    for _ in range(5):
        env.step(env.action_space.sample())
    env.reset()
    context2 = env.current_context.copy()
    assert context_space.contains(context1)
    assert context_space.contains(context2)

    # Test valid context values during steps
    action = env.action_space.sample()
    for _ in range(50):
        _, _, terminated, truncated, _ = env.step(action)
        assert context_space.contains(env.current_context)
        if terminated or truncated:
            break

    env.close()
    env2.close()


def test_probabilistic_context_wrapper_update_modes(env_5zone):
    """Test ProbabilisticContextWrapper: global_value, delta_update, and combinations."""
    context_space = gym.spaces.Box(
        low=np.array([0.5, 0.5], dtype=np.float32),
        high=np.array([0.8, 0.8], dtype=np.float32),
        shape=(2,),
        dtype=np.float32,
    )

    # Test global_value=True (same value for all variables)
    env = ProbabilisticContextWrapper(
        env=env_5zone,
        context_space=context_space,
        update_probability=1.0,
        global_value=True,
        delta_update=False,
    )
    env.reset()
    action = env.action_space.sample()
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(action)
        assert context_space.contains(env.current_context)
        assert env.current_context[0] == env.current_context[1]  # Same value
        if terminated or truncated:
            break
    env.close()

    # Test delta_update=True
    context_space2 = gym.spaces.Box(
        low=np.array([0.4, 0.4], dtype=np.float32),
        high=np.array([0.8, 0.8], dtype=np.float32),
        shape=(2,),
        dtype=np.float32,
    )
    env2 = ProbabilisticContextWrapper(
        env=env_5zone,
        context_space=context_space2,
        update_probability=1.0,
        global_value=False,
        delta_update=True,
        delta_value=0.1,
    )
    env2.reset()
    action = env2.action_space.sample()
    for _ in range(20):
        _, _, terminated, truncated, _ = env2.step(action)
        assert context_space2.contains(env2.current_context)
        if terminated or truncated:
            break
    env2.close()

    # Test global_value=True + delta_update=True (same delta for all variables)
    context_space3 = gym.spaces.Box(
        low=np.array([0.4, 0.4], dtype=np.float32),
        high=np.array([0.8, 0.8], dtype=np.float32),
        shape=(2,),
        dtype=np.float32,
    )
    env3 = ProbabilisticContextWrapper(
        env=env_5zone,
        context_space=context_space3,
        update_probability=1.0,
        global_value=True,
        delta_update=True,
        delta_value=0.05,
    )
    env3.reset()
    previous_context = env3.current_context.copy()
    action = env3.action_space.sample()
    for _ in range(10):
        _, _, terminated, truncated, _ = env3.step(action)
        assert context_space3.contains(env3.current_context)
        current_context = env3.current_context
        delta_applied = current_context - previous_context
        # With global_value=True and delta_update=True, the same delta is applied to all variables
        # However, after clipping, deltas may differ if variables hit bounds
        # So we only verify same delta when both variables can move freely
        if not np.allclose(delta_applied, 0.0):
            # Check if both variables are within bounds (not at limits) and can move
            at_low = np.isclose(current_context, context_space3.low, atol=0.01)
            at_high = np.isclose(current_context, context_space3.high, atol=0.01)
            at_bound = at_low | at_high
            # If neither variable is at a bound, deltas should be same
            if not at_bound.any():
                # Both variables can move freely, so deltas should be same
                assert np.allclose(delta_applied, delta_applied[0], atol=1e-4)
        previous_context = current_context.copy()
        if terminated or truncated:
            break
    env3.close()

    # Test update_probability as list (probabilistic per-variable mode)
    context_space4 = gym.spaces.Box(
        low=np.array([0.3, 0.3], dtype=np.float32),
        high=np.array([0.9, 0.9], dtype=np.float32),
        shape=(2,),
        dtype=np.float32,
    )
    env4 = ProbabilisticContextWrapper(
        env=env_5zone,
        context_space=context_space4,
        update_probability=[0.3, 0.5],  # 30% and 50% probability per step per variable
        global_value=False,
        delta_update=False,
    )
    assert env4.get_wrapper_attr('prob_per_variable') is True
    assert isinstance(env4.get_wrapper_attr('update_probability'), np.ndarray)
    env4.reset()
    action = env4.action_space.sample()
    for _ in range(100):
        _, _, terminated, truncated, _ = env4.step(action)
        assert context_space4.contains(env4.current_context)
        if terminated or truncated:
            break
    env4.close()


def test_probabilistic_context_wrapper_exceptions(env_5zone):
    """Test ProbabilisticContextWrapper: all validation exceptions."""
    context_space_base = gym.spaces.Box(
        low=np.array([0.3, 0.3], dtype=np.float32),
        high=np.array([0.9, 0.9], dtype=np.float32),
        shape=(2,),
        dtype=np.float32,
    )

    # Invalid context_space type
    with pytest.raises(
        TypeError, match='context_space must be an instance of gym.spaces.Box'
    ):
        ProbabilisticContextWrapper(
            env=env_5zone,
            context_space='invalid',  # type: ignore
            update_probability=0.1,
        )

    # Shape mismatch
    with pytest.raises(ValueError, match='Context space shape.*must match'):
        context_space_wrong = gym.spaces.Box(
            low=np.array([0.3, 0.3, 0.3], dtype=np.float32),
            high=np.array([0.9, 0.9, 0.9], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        ProbabilisticContextWrapper(
            env=env_5zone,
            context_space=context_space_wrong,
            update_probability=0.1,
        )

    # update_probability out of range
    with pytest.raises(ValueError, match='update_probability must be in'):
        ProbabilisticContextWrapper(
            env=env_5zone,
            context_space=context_space_base,
            update_probability=-0.1,
        )
    with pytest.raises(ValueError, match='update_probability must be in'):
        ProbabilisticContextWrapper(
            env=env_5zone,
            context_space=context_space_base,
            update_probability=1.5,
        )

    # List length mismatch
    with pytest.raises(ValueError, match='update_probability list length.*must match'):
        ProbabilisticContextWrapper(
            env=env_5zone,
            context_space=context_space_base,
            update_probability=[0.1, 0.1, 0.1],
        )

    # List values out of range
    with pytest.raises(
        ValueError, match='All values in update_probability list must be in'
    ):
        ProbabilisticContextWrapper(
            env=env_5zone,
            context_space=context_space_base,
            update_probability=[-0.1, 0.2],
        )
    with pytest.raises(
        ValueError, match='All values in update_probability list must be in'
    ):
        ProbabilisticContextWrapper(
            env=env_5zone,
            context_space=context_space_base,
            update_probability=[0.5, 1.5],
        )

    # delta_value missing
    with pytest.raises(
        ValueError, match='delta_value is required when delta_update=True'
    ):
        ProbabilisticContextWrapper(
            env=env_5zone,
            context_space=context_space_base,
            update_probability=0.1,
            delta_update=True,
            delta_value=None,
        )

    # delta_value <= 0
    with pytest.raises(ValueError, match='delta_value must be > 0'):
        ProbabilisticContextWrapper(
            env=env_5zone,
            context_space=context_space_base,
            update_probability=0.1,
            delta_update=True,
            delta_value=0.0,
        )
    with pytest.raises(ValueError, match='delta_value must be > 0'):
        ProbabilisticContextWrapper(
            env=env_5zone,
            context_space=context_space_base,
            update_probability=0.1,
            delta_update=True,
            delta_value=-0.1,
        )

    # global_value=True with different ranges
    with pytest.raises(
        ValueError, match='all dimensions of context_space must have the same range'
    ):
        context_space_diff = gym.spaces.Box(
            low=np.array([0.3, 0.5], dtype=np.float32),
            high=np.array([0.9, 0.7], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        ProbabilisticContextWrapper(
            env=env_5zone,
            context_space=context_space_diff,
            update_probability=0.1,
            global_value=True,
        )
