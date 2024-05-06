import json
import os
import shutil
from glob import glob  # to find directories with patterns

import pkg_resources
import pytest
from opyplus import WeatherData

import sinergym
from sinergym.config.modeling import ModelJSON
from sinergym.envs.eplus_env import EplusEnv
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


@pytest.fixture(scope='session')
def configuration_path_5zone(pkg_data_path):
    return os.path.join(
        pkg_data_path,
        'default_configuration',
        '5ZoneAutoDXVAV.json')

# ---------------------------------------------------------------------------- #
#                         Default Environment Arguments                        #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='session')
def TIME_VARIABLES():
    return ['month', 'day_of_month', 'hour']


@pytest.fixture(scope='session')
def ACTION_SPACE_5ZONE():
    return gym.spaces.Box(
        low=np.array([15.0, 22.5], dtype=np.float32),
        high=np.array([22.5, 30.0], dtype=np.float32),
        shape=(2,),
        dtype=np.float32
    )


@pytest.fixture(scope='session')
def VARIABLES_5ZONE():
    variables = {}
    return {
        'outdoor_temperature': (
            'Site Outdoor Air DryBulb Temperature',
            'Environment'),
        'outdoor_humidity': (
            'Site Outdoor Air Relative Humidity',
            'Environment'),
        'wind_speed': (
            'Site Wind Speed',
            'Environment'),
        'wind_direction': (
            'Site Wind Direction',
            'Environment'),
        'diffuse_solar_radiation': (
            'Site Diffuse Solar Radiation Rate per Area',
            'Environment'),
        'direct_solar_radiation': (
            'Site Direct Solar Radiation Rate per Area',
            'Environment'),
        'htg_setpoint': (
            'Zone Thermostat Heating Setpoint Temperature',
            'SPACE5-1'),
        'clg_setpoint': (
            'Zone Thermostat Cooling Setpoint Temperature',
            'SPACE5-1'),
        'air_temperature': (
            'Zone Air Temperature',
            'SPACE5-1'),
        'air_humidity': (
            'Zone Air Relative Humidity',
            'SPACE5-1'),
        'people_occupant': (
            'Zone People Occupant Count',
            'SPACE5-1'),
        'co2_emission': (
            'Environmental Impact Total CO2 Emissions Carbon Equivalent Mass',
            'site'),
        'HVAC_electricity_demand_rate': (
            'Facility Total HVAC Electricity Demand Rate',
            'Whole Building')
    }


@pytest.fixture(scope='session')
def METERS_5ZONE():
    return {'total_electricity_HVAC': 'Electricity:HVAC'}


@pytest.fixture(scope='session')
def ACTUATORS_5ZONE():
    return {
        'Heating_Setpoint_RL': (
            'Schedule:Compact',
            'Schedule Value',
            'HTG-SETP-SCH'),
        'Cooling_Setpoint_RL': (
            'Schedule:Compact',
            'Schedule Value',
            'CLG-SETP-SCH')
    }


@pytest.fixture(scope='session')
def ACTION_SPACE_DISCRETE_5ZONE():
    return gym.spaces.Discrete(10)


@pytest.fixture(scope='session')
def ACTION_SPACE_DATACENTER():
    return gym.spaces.Box(
        low=np.array([15.0, 22.5], dtype=np.float32),
        high=np.array([22.5, 30.0], dtype=np.float32),
        shape=(2,),
        dtype=np.float32
    )


@pytest.fixture(scope='session')
def VARIABLES_DATACENTER():
    return {
        'outdoor_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
        'outdoor_humidity': ('Site Outdoor Air Relative Humidity', 'Environment'),
        'wind_speed': ('Site Wind Speed', 'Environment'),
        'wind_direction': ('Site Wind Direction', 'Environment'),
        'diffuse_solar_radiation': ('Site Diffuse Solar Radiation Rate per Area', 'Environment'),
        'direct_solar_radiation': ('Site Direct Solar Radiation Rate per Area', 'Environment'),
        'west_zone_htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'West Zone'),
        'east_zone_htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'East Zone'),
        'west_zone_clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'West Zone'),
        'east_zone_clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'East Zone'),
        'west_zone_air_temperature': ('Zone Air Temperature', 'West Zone'),
        'east_zone_air_temperature': ('Zone Air Temperature', 'East Zone'),
        'west_zone_thermal_comfort_mean_radiant_temperature': ('Zone Thermal Comfort Mean Radiant Temperature', 'West Zone PEOPLE'),
        'east_zone_thermal_comfort_mean_radiant_temperature': ('Zone Thermal Comfort Mean Radiant Temperature', 'East Zone PEOPLE'),
        'west_zone_air_humidity': ('Zone Air Relative Humidity', 'West Zone'),
        'east_zone_air_humidity': ('Zone Air Relative Humidity', 'East Zone'),
        'west_zone_thermal_comfort_clothing_value': ('Zone Thermal Comfort Clothing Value', 'West Zone PEOPLE'),
        'east_zone_thermal_comfort_clothing_value': ('Zone Thermal Comfort Clothing Value', 'East Zone PEOPLE'),
        'west_zone_thermal_comfort_fanger_model_ppd': ('Zone Thermal Comfort Fanger Model PPD', 'West Zone PEOPLE'),
        'east_zone_thermal_comfort_fanger_model_ppd': ('Zone Thermal Comfort Fanger Model PPD', 'East Zone PEOPLE'),
        'west_zone_people_occupant': ('Zone People Occupant Count', 'West Zone'),
        'east_zone_people_occupant': ('Zone People Occupant Count', 'East Zone'),
        'west_zone_people_air_temperature': ('People Air Temperature', 'West Zone PEOPLE'),
        'east_zone_people_air_temperature': ('People Air Temperature', 'East Zone PEOPLE'),
        'HVAC_electricity_demand_rate': ('Facility Total HVAC Electricity Demand Rate', 'Whole Building')
    }


@pytest.fixture(scope='session')
def METERS_DATACENTER():
    return {}


@pytest.fixture(scope='session')
def ACTUATORS_DATACENTER():
    return {
        'Heating_Setpoint_RL': (
            'Schedule:Compact',
            'Schedule Value',
            'Heating Setpoints'),
        'Cooling_Setpoint_RL': (
            'Schedule:Compact',
            'Schedule Value',
            'Cooling Setpoints')
    }

# ---------------------------------------------------------------------------- #
#                       Default environment configuration                      #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='session')
def conf_5zone(configuration_path_5zone):
    with open(configuration_path_5zone) as json_f:
        conf = json.load(json_f)
    return conf

# ---------------------------------------------------------------------------- #
#                                 Environments                                 #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def env_5zone(
        ACTION_SPACE_5ZONE,
        TIME_VARIABLES,
        VARIABLES_5ZONE,
        METERS_5ZONE,
        ACTUATORS_5ZONE):
    env = EplusEnv(
        building_file='5ZoneAutoDXVAV.epJSON',
        weather_files='USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        action_space=ACTION_SPACE_5ZONE,
        time_variables=TIME_VARIABLES,
        variables=VARIABLES_5ZONE,
        meters=METERS_5ZONE,
        actuators=ACTUATORS_5ZONE,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
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
def env_5zone_stochastic(
        ACTION_SPACE_5ZONE,
        TIME_VARIABLES,
        VARIABLES_5ZONE,
        METERS_5ZONE,
        ACTUATORS_5ZONE):
    env = EplusEnv(
        building_file='5ZoneAutoDXVAV.epJSON',
        weather_files='USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        action_space=ACTION_SPACE_5ZONE,
        time_variables=TIME_VARIABLES,
        variables=VARIABLES_5ZONE,
        meters=METERS_5ZONE,
        actuators=ACTUATORS_5ZONE,
        weather_variability=(1.0, 0.0, 0.001),
        reward=LinearReward,
        reward_kwargs={
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
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
def env_datacenter(
        ACTION_SPACE_DATACENTER,
        TIME_VARIABLES,
        VARIABLES_DATACENTER,
        METERS_DATACENTER,
        ACTUATORS_DATACENTER):
    env = EplusEnv(
        building_file='2ZoneDataCenterHVAC_wEconomizer.epJSON',
        weather_files='USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        action_space=ACTION_SPACE_DATACENTER,
        time_variables=TIME_VARIABLES,
        variables=VARIABLES_DATACENTER,
        meters=METERS_DATACENTER,
        actuators=ACTUATORS_DATACENTER,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variables': [
                'west_zone_air_temperature',
                'east_zone_air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
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
def simulator_5zone(env_5zone):
    return env_5zone.energyplus_simulator


@pytest.fixture(scope='function')
def simulator_datacenter(env_datacenter):
    return env_datacenter.energyplus_simulator

# ---------------------------------------------------------------------------- #
#                               Modeling classes                               #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def building_5zone(json_path_5zone):
    with open(json_path_5zone) as json_f:
        building = json.load(json_f)
        return building


@pytest.fixture(scope='function')
def model_5zone(VARIABLES_5ZONE, METERS_5ZONE, ACTUATORS_5ZONE):

    return ModelJSON(
        env_name='TESTCONFIG',
        json_file='5ZoneAutoDXVAV.epJSON',
        weather_files=['USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw'],
        variables=VARIABLES_5ZONE,
        meters=METERS_5ZONE,
        actuators=ACTUATORS_5ZONE,
        max_ep_store=10,
        extra_config={
            'timesteps_per_hour': 2,
            'runperiod': (1, 2, 1993, 2, 3, 1993),
        })


@pytest.fixture(scope='function')
def model_5zone_several_weathers(
        VARIABLES_5ZONE,
        METERS_5ZONE,
        ACTUATORS_5ZONE):
    return ModelJSON(
        env_name='TESTCONFIG',
        json_file='5ZoneAutoDXVAV.epJSON',
        weather_files=[
            'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
            'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw'],
        variables=VARIABLES_5ZONE,
        meters=METERS_5ZONE,
        actuators=ACTUATORS_5ZONE,
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
def env_wrapper_normalization(env_5zone):
    return NormalizeObservation(env=env_5zone)


@pytest.fixture(scope='function')
def env_wrapper_multiobjective(env_5zone):
    return MultiObjectiveReward(
        env=env_5zone, reward_terms=[
            'energy_term', 'comfort_term'])


@pytest.fixture(scope='function')
def env_wrapper_logger(env_5zone):
    return LoggerWrapper(env=env_5zone, flag=True)


@pytest.fixture(scope='function')
def env_wrapper_multiobs(env_5zone):
    return MultiObsWrapper(env=env_5zone, n=5, flatten=True)


@pytest.fixture(scope='function')
def env_wrapper_datetime(env_5zone):
    return DatetimeWrapper(
        env=env_5zone)


@pytest.fixture(scope='function')
def env_wrapper_previousobs(env_5zone):
    return PreviousObservationWrapper(
        env=env_5zone,
        previous_variables=[
            'htg_setpoint',
            'clg_setpoint',
            'air_temperature'])


@pytest.fixture(scope='function')
def env_wrapper_incremental(env_5zone):
    return IncrementalWrapper(
        env=env_5zone,
        incremental_variables_definition={
            'Heating_Setpoint_RL': (2.0, 0.5),
            'Cooling_Setpoint_RL': (1.0, 0.25)
        },
        initial_values=[21.0, 25.0],
    )


@pytest.fixture(scope='function')
def env_discrete_wrapper_incremental(env_5zone):
    return DiscreteIncrementalWrapper(
        env=env_5zone,
        initial_values=[21.0, 25.0],
        delta_temp=2,
        step_temp=0.5
    )


@pytest.fixture(scope='function')
def env_normalize_action_wrapper(env_5zone):
    return NormalizeAction(env=env_5zone)


@pytest.fixture(scope='function')
def env_wrapper_discretize(env_5zone, ACTION_SPACE_DISCRETE_5ZONE):
    return DiscretizeEnv(
        env=env_5zone,
        discrete_space=ACTION_SPACE_DISCRETE_5ZONE,
        action_mapping=DEFAULT_5ZONE_DISCRETE_FUNCTION
    )


@pytest.fixture(scope='function')
def env_wrapper_reduce_observation(env_5zone):
    return ReduceObservationWrapper(
        env=env_5zone,
        obs_reduction=[
            'outdoor_temperature',
            'outdoor_humidity',
            'air_temperature'])


@pytest.fixture(scope='function')
def env_all_wrappers(env_5zone):
    env = MultiObjectiveReward(
        env=env_5zone,
        reward_terms=[
            'energy_term',
            'comfort_term'])
    env = PreviousObservationWrapper(env, previous_variables=[
        'htg_setpoint',
        'clg_setpoint',
        'air_temperature'])
    env = DatetimeWrapper(env)
    env = DiscreteIncrementalWrapper(
        env, initial_values=[
            21.0, 25.0], delta_temp=2, step_temp=0.5)
    env = NormalizeObservation(
        env=env)
    env = LoggerWrapper(env=env, flag=True)
    env = ReduceObservationWrapper(
        env=env,
        obs_reduction=[
            'outdoor_temperature',
            'outdoor_humidity',
            'air_temperature'])
    env = MultiObsWrapper(env=env, n=5, flatten=True)
    return env

# ---------------------------------------------------------------------------- #
#                                  Controllers                                 #
# ---------------------------------------------------------------------------- #


@ pytest.fixture(scope='function')
def random_controller(env_5zone):
    return RandomController(env=env_5zone)


@ pytest.fixture(scope='function')
def zone5_controller(env_5zone):
    return RBC5Zone(env=env_5zone)


@ pytest.fixture(scope='function')
def datacenter_controller(env_datacenter):
    return RBCDatacenter(env=env_datacenter)


@ pytest.fixture(scope='function')
def datacenter_incremental_controller(env_datacenter):
    return RBCIncrementalDatacenter(env=env_datacenter)

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
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        range_comfort_winter=(
            20.0,
            23.5),
        range_comfort_summer=(
            23.0,
            26.0))


@ pytest.fixture(scope='function')
def exponential_reward():
    return ExpReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        range_comfort_winter=(
            20.0,
            23.5),
        range_comfort_summer=(
            23.0,
            26.0))


@ pytest.fixture(scope='function')
def hourly_linear_reward():
    return HourlyLinearReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
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
    # Deleting all temporal directories generated during tests (environments)
    directories = glob('Eplus-env-TEST*/')
    for directory in directories:
        shutil.rmtree(directory)

    # Deleting all temporal directories generated during tests (simulators)
    directories = glob('Eplus-TESTSIMULATOR*/')
    for directory in directories:
        shutil.rmtree(directory)

    # Deleting all temporal files generated during tests
    files = glob('./TEST*.xlsx')
    for file in files:
        os.remove(file)
    files = glob('./data_available*')
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
