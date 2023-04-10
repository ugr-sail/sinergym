import os
import shutil
from glob import glob  # to find directories with patterns

import pkg_resources
import pytest
from opyplus import Epm, Idd, WeatherData

from sinergym.envs.eplus_env import EplusEnv
from sinergym.simulators.eplus import EnergyPlus
from sinergym.utils.config import Config
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
def eplus_path():
    return os.environ['EPLUS_PATH']


@pytest.fixture(scope='session')
def bcvtb_path():
    return os.environ['BCVTB_PATH']


@pytest.fixture(scope='session')
def pkg_data_path():
    return PKG_DATA_PATH


@pytest.fixture(scope='session')
def idf_path(pkg_data_path):
    return os.path.join(pkg_data_path, 'buildings', '5ZoneAutoDXVAV.idf')


@pytest.fixture(scope='session')
def weather_path(pkg_data_path):
    return os.path.join(
        pkg_data_path,
        'weather',
        'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw')


@pytest.fixture(scope='session')
def idf_file():
    return '5ZoneAutoDXVAV.idf'


@pytest.fixture(scope='session')
def weather_file():
    return 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw'


@pytest.fixture(scope='session')
def idf_file2():
    return '2ZoneDataCenterHVAC_wEconomizer.idf'


@pytest.fixture(scope='session')
def weather_file2():
    return 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw'


# 5zones variables


@pytest.fixture(scope='session')
def variables_5zone():
    variables = {}
    variables['observation'] = DEFAULT_5ZONE_OBSERVATION_VARIABLES
    variables['action'] = DEFAULT_5ZONE_ACTION_VARIABLES
    return variables

# datacenter variables


@pytest.fixture(scope='session')
def variables_datacenter():
    variables = {}
    variables['observation'] = DEFAULT_DATACENTER_OBSERVATION_VARIABLES
    variables['action'] = DEFAULT_DATACENTER_ACTION_VARIABLES
    return variables

# ---------------------------------------------------------------------------- #
#                                 Environments                                 #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def env_demo(idf_file, weather_file):

    return EplusEnv(
        env_name='TESTGYM',
        idf_file=idf_file,
        weather_file=weather_file,
        observation_space=DEFAULT_5ZONE_OBSERVATION_SPACE,
        observation_variables=DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        action_space=DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        action_variables=DEFAULT_5ZONE_ACTION_VARIABLES,
        action_mapping=DEFAULT_5ZONE_ACTION_MAPPING,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        weather_variability=None,
        action_definition=DEFAULT_5ZONE_ACTION_DEFINITION)


@pytest.fixture(scope='function')
def env_demo_continuous(idf_file, weather_file):

    return EplusEnv(
        env_name='TESTGYM',
        idf_file=idf_file,
        weather_file=weather_file,
        observation_space=DEFAULT_5ZONE_OBSERVATION_SPACE,
        observation_variables=DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        action_space=DEFAULT_5ZONE_ACTION_SPACE_CONTINUOUS,
        action_variables=DEFAULT_5ZONE_ACTION_VARIABLES,
        action_mapping=DEFAULT_5ZONE_ACTION_MAPPING,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        weather_variability=None,
        action_definition=DEFAULT_5ZONE_ACTION_DEFINITION)


@pytest.fixture(scope='function')
def env_demo_continuous_stochastic(idf_file, weather_file):

    return EplusEnv(
        env_name='TESTGYM',
        idf_file=idf_file,
        weather_file=weather_file,
        observation_space=DEFAULT_5ZONE_OBSERVATION_SPACE,
        observation_variables=DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        action_space=DEFAULT_5ZONE_ACTION_SPACE_CONTINUOUS,
        action_variables=DEFAULT_5ZONE_ACTION_VARIABLES,
        action_mapping=DEFAULT_5ZONE_ACTION_MAPPING,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        weather_variability=(1.0, 0.0, 0.001),
        action_definition=DEFAULT_5ZONE_ACTION_DEFINITION)


@pytest.fixture(scope='function')
def env_datacenter(idf_file2, weather_file):

    return EplusEnv(
        env_name='TESTGYM',
        idf_file=idf_file2,
        weather_file=weather_file,
        observation_space=DEFAULT_DATACENTER_OBSERVATION_SPACE,
        observation_variables=DEFAULT_DATACENTER_OBSERVATION_VARIABLES,
        action_space=DEFAULT_DATACENTER_ACTION_SPACE_DISCRETE,
        action_variables=DEFAULT_DATACENTER_ACTION_VARIABLES,
        action_mapping=DEFAULT_DATACENTER_ACTION_MAPPING,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (
                18,
                27),
            'range_comfort_summer': (
                18,
                27)},
        weather_variability=None,
        action_definition=DEFAULT_DATACENTER_ACTION_DEFINITION)


@pytest.fixture(scope='function')
def env_datacenter_continuous(
        idf_file2,
        weather_file):

    return EplusEnv(
        env_name='TESTGYM',
        idf_file=idf_file2,
        weather_file=weather_file,
        observation_space=DEFAULT_DATACENTER_OBSERVATION_SPACE,
        observation_variables=DEFAULT_DATACENTER_OBSERVATION_VARIABLES,
        action_space=DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        action_variables=DEFAULT_DATACENTER_ACTION_VARIABLES,
        action_mapping=DEFAULT_DATACENTER_ACTION_MAPPING,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (
                18,
                27),
            'range_comfort_summer': (
                18,
                27)},
        weather_variability=None,
        action_definition=DEFAULT_DATACENTER_ACTION_DEFINITION)

# ---------------------------------------------------------------------------- #
#                                  Simulators                                  #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def simulator(
        eplus_path,
        bcvtb_path,
        idf_file,
        weather_file,
        variables_5zone):
    env_name = 'TEST'
    return EnergyPlus(
        eplus_path=eplus_path,
        bcvtb_path=bcvtb_path,
        weather_files=[weather_file],
        idf_file=idf_file,
        env_name=env_name,
        variables=variables_5zone,
        act_repeat=1,
        max_ep_data_store_num=10,
        action_definition=DEFAULT_5ZONE_ACTION_DEFINITION)

# ---------------------------------------------------------------------------- #
#                            Simulator Config class                            #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def config(idf_file, weather_file2, variables_5zone):
    env_name = 'TESTCONFIG'
    max_ep_store = 10
    return Config(
        idf_file=idf_file,
        weather_files=[weather_file2],
        env_name=env_name,
        variables=variables_5zone,
        max_ep_store=max_ep_store,
        action_definition=DEFAULT_5ZONE_ACTION_DEFINITION,
        extra_config={
            'timesteps_per_hour': 2,
            'runperiod': (1, 2, 1993, 2, 3, 1993),
        })


@pytest.fixture(scope='function')
def config_several_weathers(
        idf_file,
        weather_file,
        weather_file2,
        variables_5zone):
    env_name = 'TESTCONFIG'
    max_ep_store = 10
    return Config(
        idf_file=idf_file,
        weather_files=[weather_file, weather_file2],
        env_name=env_name,
        variables=variables_5zone,
        max_ep_store=max_ep_store,
        action_definition=DEFAULT_5ZONE_ACTION_DEFINITION,
        extra_config={
            'timesteps_per_hour': 2,
            'runperiod': (1, 2, 1993, 2, 3, 1993),
        })

# ---------------------------------------------------------------------------- #
#                          Environments with Wrappers                          #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def env_wrapper_normalization(env_demo_continuous):
    return NormalizeObservation(env=env_demo_continuous, ranges=RANGES_5ZONE)


@pytest.fixture(scope='function')
def env_wrapper_multiobjective(env_demo_continuous):
    return MultiObjectiveReward(
        env=env_demo_continuous, reward_terms=[
            'reward_energy', 'reward_comfort'])


@pytest.fixture(scope='function')
def env_wrapper_logger(env_demo_continuous):
    return LoggerWrapper(env=env_demo_continuous, flag=True)


@pytest.fixture(scope='function')
def env_wrapper_multiobs(env_demo_continuous):
    return MultiObsWrapper(env=env_demo_continuous, n=5, flatten=True)


@pytest.fixture(scope='function')
def env_wrapper_datetime(env_demo_continuous):
    return DatetimeWrapper(
        env=env_demo_continuous)


@pytest.fixture(scope='function')
def env_wrapper_previousobs(env_demo_continuous):
    return PreviousObservationWrapper(
        env=env_demo_continuous,
        previous_variables=[
            'Zone Thermostat Heating Setpoint Temperature(SPACE1-1)',
            'Zone Thermostat Cooling Setpoint Temperature(SPACE1-1)',
            'Zone Air Temperature(SPACE1-1)'])


@pytest.fixture(scope='function')
def env_wrapper_incremental(env_demo):
    return DiscreteIncrementalWrapper(
        env=env_demo,
        max_values=[22.0, 34.0],
        min_values=[10.0, 22.0],
        delta_temp=2,
        step_temp=0.5
    )


@pytest.fixture(scope='function')
def env_all_wrappers(env_demo):
    env = NormalizeObservation(env=env_demo, ranges=RANGES_5ZONE)
    env = MultiObjectiveReward(
        env=env,
        reward_terms=[
            'reward_energy',
            'reward_comfort'])
    env = LoggerWrapper(env=env, flag=True)
    env = MultiObsWrapper(env=env, n=5, flatten=True)
    return env

# ---------------------------------------------------------------------------- #
#                                  Controllers                                 #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def random_controller(env_demo):
    return RandomController(env=env_demo)


@pytest.fixture(scope='function')
def zone5_controller(env_demo):
    return RBC5Zone(env=env_demo)


@pytest.fixture(scope='function')
def datacenter_controller(env_datacenter):
    return RBCDatacenter(env=env_datacenter)

# ---------------------------------------------------------------------------- #
#                      Building and weather python models                      #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def epm(idf_path, eplus_path):
    idd = Idd(os.path.join(eplus_path, 'Energy+.idd'))
    return Epm.from_idf(idf_path, idd_or_version=idd)


@pytest.fixture(scope='function')
def weather_data(weather_path):
    return WeatherData.from_epw(weather_path)

# ---------------------------------------------------------------------------- #
#                                    Rewards                                   #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def base_reward(env_demo):
    return BaseReward()


@pytest.fixture(scope='function')
def custom_reward(env_demo):
    class CustomReward(BaseReward):
        def __init__(self):
            super(CustomReward, self).__init__()

        def __call__(self):
            return -1.0, {}

    return CustomReward()


@pytest.fixture(scope='function')
def linear_reward():
    return LinearReward(
        temperature_variable='Zone Air Temperature(SPACE1-1)',
        energy_variable='Facility Total HVAC Electricity Demand Rate(Whole Building)',
        range_comfort_winter=(
            20.0,
            23.5),
        range_comfort_summer=(
            23.0,
            26.0))


@pytest.fixture(scope='function')
def exponential_reward():
    return ExpReward(
        temperature_variable=[
            'Zone Air Temperature(SPACE1-1)',
            'Zone Air Temperature(SPACE1-2)'],
        energy_variable='Facility Total HVAC Electricity Demand Rate(Whole Building)',
        range_comfort_winter=(
            20.0,
            23.5),
        range_comfort_summer=(
            23.0,
            26.0))


@pytest.fixture(scope='function')
def hourly_linear_reward():
    return HourlyLinearReward(
        temperature_variable='Zone Air Temperature(SPACE1-1)',
        energy_variable='Facility Total HVAC Electricity Demand Rate(Whole Building)',
        range_comfort_winter=(
            20.0,
            23.5),
        range_comfort_summer=(
            23.0,
            26.0))


@pytest.fixture(scope='function')
def env_custom_reward(
        idf_file,
        weather_file,
        custom_reward):

    return EplusEnv(
        env_name='TESTGYM',
        idf_file=idf_file,
        weather_file=weather_file,
        observation_space=DEFAULT_5ZONE_OBSERVATION_SPACE,
        observation_variables=DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        action_space=DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        action_variables=DEFAULT_5ZONE_ACTION_VARIABLES,
        action_mapping=DEFAULT_5ZONE_ACTION_MAPPING,
        reward=custom_reward,
        weather_variability=None,
        action_definition=DEFAULT_5ZONE_ACTION_DEFINITION
    )


@pytest.fixture(scope='function')
def env_linear_reward(idf_file, weather_file):

    return EplusEnv(
        env_name='TESTGYM',
        idf_file=idf_file,
        weather_file=weather_file,
        observation_space=DEFAULT_5ZONE_OBSERVATION_SPACE,
        observation_variables=DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        action_space=DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        action_variables=DEFAULT_5ZONE_ACTION_VARIABLES,
        action_mapping=DEFAULT_5ZONE_ACTION_MAPPING,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        weather_variability=None,
        action_definition=DEFAULT_5ZONE_ACTION_DEFINITION)


@pytest.fixture(scope='function')
def env_linear_reward_args(idf_file, weather_file):

    return EplusEnv(
        env_name='TESTGYM',
        idf_file=idf_file,
        weather_file=weather_file,
        observation_space=DEFAULT_5ZONE_OBSERVATION_SPACE,
        observation_variables=DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        action_space=DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        action_variables=DEFAULT_5ZONE_ACTION_VARIABLES,
        action_mapping=DEFAULT_5ZONE_ACTION_MAPPING,
        reward=LinearReward,
        reward_kwargs={
            'energy_weight': 0.2,
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (18.0, 20.0)},
        weather_variability=None,
        action_definition=DEFAULT_5ZONE_ACTION_DEFINITION)


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

    # Deleting new IDF files generated during tests
    files = glob('sinergym/data/buildings/TEST*.idf')
    for file in files:
        os.remove(file)

    # Deleting new random weather files generated during tests
    files = glob('sinergym/data/weather/*Random*.epw')
    for file in files:
        os.remove(file)
