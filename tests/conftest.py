import json
import os
import shutil
from glob import glob  # to find directories with patterns

import pkg_resources
import pytest
from opyplus import WeatherData

import sinergym
import gymnasium as gym
from sinergym.config.modeling import ModelJSON
from sinergym.envs.eplus_env import EplusEnv
from sinergym.simulators.eplus import EnergyPlus
from sinergym.utils.constants import *
from sinergym.utils.controllers import *
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *

# ---------------------------------------------------------------------------- #
#                                Root Directory                                #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='session')
def sinergym_path():
    return os.path.abspath(
        os.path.join(
            pkg_resources.resource_filename(
                'sinergym',
                ''),
            os.pardir))

# ---------------------------------------------------------------------------- #
#                                     Paths                                    #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='session')
def pkg_data_path():
    return PKG_DATA_PATH


@pytest.fixture(scope='session')
def json_path_5zone(pkg_data_path):
    return os.path.join(pkg_data_path, 'buildings', '5ZoneAutoDXVAV.epJSON')


@pytest.fixture(scope='session')
def weather_path_pittsburgh(pkg_data_path):
    return os.path.join(
        pkg_data_path,
        'weather',
        'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw')

# ---------------------------------------------------------------------------- #
#                                 Environments                                 #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def env_5zone_discrete():
    env = EplusEnv(
        building_file='5ZoneAutoDXVAV.epJSON',
        weather_files='USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        action_space=DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        time_variables=DEFAULT_TIME_VARIABLES,
        variables=DEFAULT_5ZONE_VARIABLES,
        meters=DEFAULT_5ZONE_METERS,
        actuators=DEFAULT_5ZONE_ACTUATORS,
        action_mapping=DEFAULT_5ZONE_ACTION_MAPPING,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variable': 'air_temperature',
            'energy_variable': 'HVAC_electricity_demand_rate',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        env_name='TESTGYM',
        config_params={
            'runperiod': (1, 1, 1991, 31, 3, 1991)
        }
    )
    return env


@pytest.fixture(scope='function')
def env_5zone_continuous():
    env = EplusEnv(
        building_file='5ZoneAutoDXVAV.epJSON',
        weather_files='USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        action_space=DEFAULT_5ZONE_ACTION_SPACE_CONTINUOUS,
        time_variables=DEFAULT_TIME_VARIABLES,
        variables=DEFAULT_5ZONE_VARIABLES,
        meters=DEFAULT_5ZONE_METERS,
        actuators=DEFAULT_5ZONE_ACTUATORS,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variable': 'air_temperature',
            'energy_variable': 'HVAC_electricity_demand_rate',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        env_name='TESTGYM',
        config_params={
            'runperiod': (1, 1, 1991, 31, 3, 1991)
        }
    )
    return env


@pytest.fixture(scope='function')
def env_5zopne_continuous_stochastic():
    env = EplusEnv(
        building_file='5ZoneAutoDXVAV.epJSON',
        weather_files='USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        action_space=DEFAULT_5ZONE_ACTION_SPACE_CONTINUOUS,
        time_variables=DEFAULT_TIME_VARIABLES,
        variables=DEFAULT_5ZONE_VARIABLES,
        meters=DEFAULT_5ZONE_METERS,
        actuators=DEFAULT_5ZONE_ACTUATORS,
        weather_variability=(1.0, 0.0, 0.001),
        reward=LinearReward,
        reward_kwargs={
            'temperature_variable': 'air_temperature',
            'energy_variable': 'HVAC_electricity_demand_rate',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        env_name='TESTGYM',
        config_params={
            'runperiod': (1, 1, 1991, 31, 3, 1991)
        }
    )
    return env


@pytest.fixture(scope='function')
def env_datacenter_discrete():
    env = EplusEnv(
        building_file='2ZoneDataCenterHVAC_wEconomizer.epJSON',
        weather_files='USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        action_space=DEFAULT_DATACENTER_ACTION_SPACE_DISCRETE,
        time_variables=DEFAULT_TIME_VARIABLES,
        variables=DEFAULT_DATACENTER_VARIABLES,
        meters=DEFAULT_DATACENTER_METERS,
        actuators=DEFAULT_DATACENTER_ACTUATORS,
        action_mapping=DEFAULT_DATACENTER_ACTION_MAPPING,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variable': [
                'west_zone_temperature',
                'east_zone_temperature'],
            'energy_variable': 'HVAC_electricity_demand_rate',
            'range_comfort_winter': (
                18,
                27),
            'range_comfort_summer': (
                18,
                27)},
        env_name='TESTGYM',
        config_params={
            'runperiod': (1, 1, 1991, 31, 3, 1991)
        }
    )
    return env


@pytest.fixture(scope='function')
def env_datacenter_continuous():
    env = EplusEnv(
        building_file='2ZoneDataCenterHVAC_wEconomizer.epJSON',
        weather_files='USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        action_space=DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        time_variables=DEFAULT_TIME_VARIABLES,
        variables=DEFAULT_DATACENTER_VARIABLES,
        meters=DEFAULT_DATACENTER_METERS,
        actuators=DEFAULT_DATACENTER_ACTUATORS,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variable': [
                'west_zone_temperature',
                'east_zone_temperature'],
            'energy_variable': 'HVAC_electricity_demand_rate',
            'range_comfort_winter': (
                18,
                27),
            'range_comfort_summer': (
                18,
                27)},
        env_name='TESTGYM',
        config_params={
            'runperiod': (1, 1, 1991, 31, 3, 1991)
        }
    )
    return env

# ---------------------------------------------------------------------------- #
#                                  Simulators                                  #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def simulator_5zone(env_5zone_continuous):
    return env_5zone_continuous.energyplus_simulator


@pytest.fixture(scope='function')
def simulator_datacenter(env_datacenter_continuous):
    return env_datacenter_continuous.energyplus_simulator

# ---------------------------------------------------------------------------- #
#                               Modeling classes                               #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def model_5zone():

    return ModelJSON(
        env_name='TESTCONFIG',
        json_file='5ZoneAutoDXVAV.epJSON',
        weather_files=['USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw'],
        variables=DEFAULT_5ZONE_VARIABLES,
        meters=DEFAULT_5ZONE_METERS,
        actuators=DEFAULT_5ZONE_ACTUATORS,
        max_ep_store=10,
        extra_config={
            'timesteps_per_hour': 2,
            'runperiod': (1, 2, 1993, 2, 3, 1993),
        })


@pytest.fixture(scope='function')
def model_5zone_several_weathers():
    return ModelJSON(
        env_name='TESTCONFIG',
        json_file='5ZoneAutoDXVAV.epJSON',
        weather_files=[
            'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
            'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw'],
        variables=DEFAULT_5ZONE_VARIABLES,
        meters=DEFAULT_5ZONE_METERS,
        actuators=DEFAULT_5ZONE_ACTUATORS,
        max_ep_store=10,
        extra_config={
            'timesteps_per_hour': 2,
            'runperiod': (
                1,
                2,
                1993,
                2,
                3,
                1993),
        })

# ---------------------------------------------------------------------------- #
#                          Environments with Wrappers                          #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def env_wrapper_normalization(env_5zone_continuous):
    return NormalizeObservation(env=env_5zone_continuous, ranges=RANGES_5ZONE)


@pytest.fixture(scope='function')
def env_wrapper_multiobjective(env_5zone_continuous):
    return MultiObjectiveReward(
        env=env_5zone_continuous, reward_terms=[
            'reward_energy', 'reward_comfort'])


@pytest.fixture(scope='function')
def env_wrapper_logger(env_5zone_continuous):
    return LoggerWrapper(env=env_5zone_continuous, flag=True)


@pytest.fixture(scope='function')
def env_wrapper_multiobs(env_5zone_continuous):
    return MultiObsWrapper(env=env_5zone_continuous, n=5, flatten=True)


@pytest.fixture(scope='function')
def env_wrapper_datetime(env_5zone_continuous):
    return DatetimeWrapper(
        env=env_5zone_continuous)


@pytest.fixture(scope='function')
def env_wrapper_previousobs(env_5zone_continuous):
    return PreviousObservationWrapper(
        env=env_5zone_continuous,
        previous_variables=[
            'htg_setpoint',
            'clg_setpoint',
            'air_temperature'])


@pytest.fixture(scope='function')
def env_wrapper_incremental(env_5zone_continuous):
    return DiscreteIncrementalWrapper(
        env=env_5zone_continuous,
        delta_temp=2,
        step_temp=0.5
    )


@pytest.fixture(scope='function')
def env_all_wrappers(env_5zone_continuous):
    env = MultiObjectiveReward(
        env=env_5zone_continuous,
        reward_terms=[
            'reward_energy',
            'reward_comfort'])
    env = PreviousObservationWrapper(env, previous_variables=[
        'htg_setpoint',
        'clg_setpoint',
        'air_temperature'])
    env = DatetimeWrapper(env)
    env = DiscreteIncrementalWrapper(
        env, initial_values=[
            21.0, 25.0], delta_temp=2, step_temp=0.5)
    env = NormalizeObservation(
        env=env,
        ranges=RANGES_5ZONE,
        variables=[
            'htg_setpoint',
            'clg_setpoint',
            'air_temperature',
            'htg_setpoint_previous',
            'clg_setpoint_previous',
            'air_temperature_previous'])
    env = LoggerWrapper(env=env, flag=True)
    env = MultiObsWrapper(env=env, n=5, flatten=True)
    return env

# ---------------------------------------------------------------------------- #
#                                  Controllers                                 #
# ---------------------------------------------------------------------------- #


@ pytest.fixture(scope='function')
def random_controller(env_5zone_continuous):
    env_5zone_continuous.update_flag_normalization(False)
    return RandomController(env=env_5zone_continuous)


@ pytest.fixture(scope='function')
def zone5_controller(env_5zone_continuous):
    env_5zone_continuous.update_flag_normalization(False)
    return RBC5Zone(env=env_5zone_continuous)


@ pytest.fixture(scope='function')
def datacenter_controller(env_datacenter_continuous):
    env_datacenter_continuous.update_flag_normalization(False)
    return RBCDatacenter(env=env_datacenter_continuous)

# ---------------------------------------------------------------------------- #
#                      Building and weather python models                      #
# ---------------------------------------------------------------------------- #


@ pytest.fixture(scope='function')
def building(json_path_5zone):
    with open(json_path_5zone) as json_f:
        building_model = json.load(json_f)
    return building_model


@ pytest.fixture(scope='function')
def weather_data(weather_path_pittsburgh):
    return WeatherData.from_epw(weather_path_pittsburgh)

# ---------------------------------------------------------------------------- #
#                                    Rewards                                   #
# ---------------------------------------------------------------------------- #


@ pytest.fixture(scope='function')
def base_reward():
    return BaseReward()


@ pytest.fixture(scope='function')
def custom_reward():
    class CustomReward(BaseReward):
        def __init__(self):
            super(CustomReward, self).__init__()

        def __call__(self):
            return -1.0, {}

    return CustomReward()


@ pytest.fixture(scope='function')
def linear_reward():
    return LinearReward(
        temperature_variable='air_temperature',
        energy_variable='HVAC_electricity_demand_rate',
        range_comfort_winter=(
            20.0,
            23.5),
        range_comfort_summer=(
            23.0,
            26.0))


@ pytest.fixture(scope='function')
def exponential_reward():
    return ExpReward(
        temperature_variable=[
            'air_temperature1',
            'air_temperature2'],
        energy_variable='HVAC_electricity_demand_rate',
        range_comfort_winter=(
            20.0,
            23.5),
        range_comfort_summer=(
            23.0,
            26.0))


@ pytest.fixture(scope='function')
def hourly_linear_reward():
    return HourlyLinearReward(
        temperature_variable='air_temperature',
        energy_variable='HVAC_electricity_demand_rate',
        range_comfort_winter=(
            20.0,
            23.5),
        range_comfort_summer=(
            23.0,
            26.0))

# ---------------------------------------------------------------------------- #
#                         WHEN TESTS HAVE BEEN FINISHED                        #
# ---------------------------------------------------------------------------- #


def pytest_sessionfinish(session, exitstatus):
    """ whole test run finishes. """
    # Deleting all temporal directories generated during tests
    directories = glob('Eplus-env-TEST*/')
    for directory in directories:
        shutil.rmtree(directory)

    # Deleting all temporal files generated during tests
    files = glob('./TEST*.xlsx')
    for file in files:
        os.remove(file)

    # Deleting new JSON files generated during tests
    files = glob('sinergym/data/buildings/TEST*.epJSON')
    for file in files:
        os.remove(file)

    # Deleting new random weather files generated during tests
    files = glob('sinergym/data/weather/*Random*.epw')
    for file in files:
        os.remove(file)
